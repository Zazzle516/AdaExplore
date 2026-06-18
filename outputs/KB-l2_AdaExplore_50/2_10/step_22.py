import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused conv (ConvTranspose2d with stride=1, pad=1, k=3 == regular conv with flipped weights, transposed in/out)
# + 2x2 maxpool + hardtanh + spatial mean.
# Output (N, C_out, 1, 1). Final tanh applied separately on the small tensor.

@triton.jit
def fused_conv_pool_htanh_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC, H, W,
    H_out, W_out,  # H/2, W/2
    HTMIN: tl.constexpr,
    HTMAX: tl.constexpr,
    BLOCK_SP: tl.constexpr,  # number of pooled output positions per program
):
    # grid: (N * OC, num_sp_blocks)
    # We accumulate partial sums per program, then atomic add into out[(n,oc)]
    pid_nc = tl.program_id(0)
    pid_sp = tl.program_id(1)

    n = pid_nc // OC
    oc = pid_nc % OC

    total_sp = H_out * W_out
    inv = 1.0 / total_sp

    sp_start = pid_sp * BLOCK_SP
    offs = sp_start + tl.arange(0, BLOCK_SP)
    mask_sp = offs < total_sp

    ho = offs // W_out
    wo = offs % W_out
    # output H position (after pool, each pool covers 2x2 of conv output)
    # conv output positions for the 2x2 window:
    h_conv0 = ho * 2  # rows h_conv0, h_conv0+1
    w_conv0 = wo * 2  # cols w_conv0, w_conv0+1

    # We compute conv at 4 positions: (h0,w0), (h0,w0+1), (h0+1,w0), (h0+1,w0+1)
    # Conv: y[oc, h, w] = sum_{ic, kh, kw} x[ic, h+kh-1, w+kw-1] * W_eff[oc, ic, kh, kw] + b[oc]
    # where W_eff[oc, ic, kh, kw] = weight[ic, oc, 2-kh, 2-kw]   (ConvTranspose2d weight shape: (IC, OC, K, K))
    # With padding=1, kernel=3.

    acc0 = tl.zeros([BLOCK_SP], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_SP], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_SP], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_SP], dtype=tl.float32)

    x_base = n * IC * H * W
    # weight layout: (IC, OC, 3, 3). flipped index: kh' = 2-kh, kw' = 2-kw
    # W_eff[oc, ic, kh, kw] = w_ptr[ic*OC*9 + oc*9 + (2-kh)*3 + (2-kw)]

    for ic in range(0, IC):
        x_ic_base = x_base + ic * H * W
        w_ic_base = ic * OC * 9 + oc * 9
        # preload 9 weights for this (ic, oc)
        w00 = tl.load(w_ptr + w_ic_base + (2 * 3 + 2))  # kh=0,kw=0 -> flipped (2,2)
        w01 = tl.load(w_ptr + w_ic_base + (2 * 3 + 1))  # kh=0,kw=1 -> (2,1)
        w02 = tl.load(w_ptr + w_ic_base + (2 * 3 + 0))  # kh=0,kw=2 -> (2,0)
        w10 = tl.load(w_ptr + w_ic_base + (1 * 3 + 2))  # kh=1,kw=0 -> (1,2)
        w11 = tl.load(w_ptr + w_ic_base + (1 * 3 + 1))
        w12 = tl.load(w_ptr + w_ic_base + (1 * 3 + 0))
        w20 = tl.load(w_ptr + w_ic_base + (0 * 3 + 2))
        w21 = tl.load(w_ptr + w_ic_base + (0 * 3 + 1))
        w22 = tl.load(w_ptr + w_ic_base + (0 * 3 + 0))

        # For 2x2 conv outputs, we need x at rows h0-1, h0, h0+1, h0+2 and cols w0-1, w0, w0+1, w0+2
        # That's a 4x4 patch per pooled position.
        # Load all 16 positions (with bounds masking for padding).

        # Helper: load x at (h_conv0 + dh, w_conv0 + dw) where dh in {-1,0,1,2}, dw in {-1,0,1,2}
        # Use mask for padding.

        def load_x(dh, dw):
            hh = h_conv0 + dh
            ww = w_conv0 + dw
            valid = (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W) & mask_sp
            idx = x_ic_base + hh * W + ww
            return tl.load(x_ptr + idx, mask=valid, other=0.0)

        x_m1_m1 = load_x(-1, -1)
        x_m1_0  = load_x(-1, 0)
        x_m1_1  = load_x(-1, 1)
        x_m1_2  = load_x(-1, 2)
        x_0_m1  = load_x(0, -1)
        x_0_0   = load_x(0, 0)
        x_0_1   = load_x(0, 1)
        x_0_2   = load_x(0, 2)
        x_1_m1  = load_x(1, -1)
        x_1_0   = load_x(1, 0)
        x_1_1   = load_x(1, 1)
        x_1_2   = load_x(1, 2)
        x_2_m1  = load_x(2, -1)
        x_2_0   = load_x(2, 0)
        x_2_1   = load_x(2, 1)
        x_2_2   = load_x(2, 2)

        # conv output at (h0, w0): uses x at rows h0-1..h0+1, cols w0-1..w0+1
        acc0 += (x_m1_m1 * w00 + x_m1_0 * w01 + x_m1_1 * w02 +
                 x_0_m1  * w10 + x_0_0  * w11 + x_0_1  * w12 +
                 x_1_m1  * w20 + x_1_0  * w21 + x_1_1  * w22)

        # conv output at (h0, w0+1): rows h0-1..h0+1, cols w0..w0+2
        acc1 += (x_m1_0 * w00 + x_m1_1 * w01 + x_m1_2 * w02 +
                 x_0_0  * w10 + x_0_1  * w11 + x_0_2  * w12 +
                 x_1_0  * w20 + x_1_1  * w21 + x_1_2  * w22)

        # conv output at (h0+1, w0): rows h0..h0+2, cols w0-1..w0+1
        acc2 += (x_0_m1 * w00 + x_0_0 * w01 + x_0_1 * w02 +
                 x_1_m1 * w10 + x_1_0 * w11 + x_1_1 * w12 +
                 x_2_m1 * w20 + x_2_0 * w21 + x_2_1 * w22)

        # conv output at (h0+1, w0+1): rows h0..h0+2, cols w0..w0+2
        acc3 += (x_0_0 * w00 + x_0_1 * w01 + x_0_2 * w02 +
                 x_1_0 * w10 + x_1_1 * w11 + x_1_2 * w12 +
                 x_2_0 * w20 + x_2_1 * w21 + x_2_2 * w22)

    # add bias
    bias = tl.load(b_ptr + oc)
    acc0 += bias
    acc1 += bias
    acc2 += bias
    acc3 += bias

    # 2x2 max pool
    m01 = tl.maximum(acc0, acc1)
    m23 = tl.maximum(acc2, acc3)
    m = tl.maximum(m01, m23)

    # hardtanh
    m = tl.minimum(tl.maximum(m, HTMIN), HTMAX)
    m = tl.where(mask_sp, m, 0.0)

    partial = tl.sum(m, axis=0) * inv

    # atomic add into out[n, oc]
    tl.atomic_add(out_ptr + (n * OC + oc), partial)


def fused_conv_pool_htanh_mean(x, weight, bias, hardtanh_min, hardtanh_max):
    N, IC, H, W = x.shape
    OC = weight.shape[1]
    H_out = H // 2
    W_out = W // 2

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

    BLOCK_SP = 256
    total_sp = H_out * W_out
    num_sp_blocks = (total_sp + BLOCK_SP - 1) // BLOCK_SP
    grid = (N * OC, num_sp_blocks)

    fused_conv_pool_htanh_mean_kernel[grid](
        x, weight, bias, out,
        N, IC, OC, H, W,
        H_out, W_out,
        float(hardtanh_min), float(hardtanh_max),
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out.view(N, OC, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        # Fast path: stride=1, pad=1, k=3, pool=2x2 stride 2
        if (self.stride == 1 and self.padding == 1 and self.kernel_size == 3
                and self.maxpool_kernel_size == 2 and self.maxpool_stride == 2
                and x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0
                and x.is_cuda):
            x = x.contiguous()
            mean_out = fused_conv_pool_htanh_mean(
                x, self.conv_transpose.weight, self.conv_transpose.bias,
                self.hardtanh_min, self.hardtanh_max
            )
            return torch.tanh(mean_out)
        else:
            x = self.conv_transpose(x)
            x = F.max_pool2d(x, self.maxpool_kernel_size, self.maxpool_stride)
            x = F.hardtanh(x, self.hardtanh_min, self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x