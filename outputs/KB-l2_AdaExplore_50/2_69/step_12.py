import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
    ],
    key=['C_out', 'H_out', 'W_out', 'C_in_K'],
)
@triton.jit
def conv2d_nhwc_im2col_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    C_in_K: tl.constexpr,  # C_in * KH * KW
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N * cdiv(HW, BLOCK_HW), cdiv(C_out, BLOCK_OC))
    pid_nhw = tl.program_id(0)
    pid_oc = tl.program_id(1)

    num_hw_blocks = tl.cdiv(H_out * W_out, BLOCK_HW)
    pid_n = pid_nhw // num_hw_blocks
    pid_hw = pid_nhw % num_hw_blocks

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = hw_offs // W_out
    ow = hw_offs % W_out

    oc_mask = oc_offs < C_out
    hw_mask = hw_offs < (H_out * W_out)

    # x is NHWC: shape (N, H_in, W_in, C_in), so addressing:
    # x[n, ih, iw, ci] = n*H_in*W_in*C_in + ih*W_in*C_in + iw*C_in + ci
    # K = C_in * KH * KW; we treat k = (kh*KW + kw)*C_in + ci so input access stride along K is contiguous (ci dim)
    K = C_in_K
    k_offs = tl.arange(0, K)  # [K]
    # decompose k -> (kh, kw, ci)
    ci = k_offs % C_in
    khw = k_offs // C_in
    kh = khw // KW
    kw = khw % KW

    # input pixel locations per hw
    ih = oh[:, None] + kh[None, :]  # [BLOCK_HW, K]
    iw = ow[:, None] + kw[None, :]

    x_base = pid_n * (H_in * W_in * C_in)
    x_idx = x_base + ih * (W_in * C_in) + iw * C_in + ci[None, :]  # [BLOCK_HW, K]
    x_mask = hw_mask[:, None]

    x_tile = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)  # [BLOCK_HW, K]

    # weight is layout: (C_out, K) where K = (kh*KW+kw)*C_in + ci
    # original conv weight is (C_out, C_in, KH, KW). We need to permute to (C_out, KH, KW, C_in) and reshape.
    # That permutation is done in python beforehand and passed as w_ptr (C_out, K) contiguous.
    w_idx = oc_offs[:, None] * K + k_offs[None, :]  # [BLOCK_OC, K]
    w_mask = oc_mask[:, None]
    w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)  # [BLOCK_OC, K]

    # GEMM: acc = w_tile @ x_tile.T -> [BLOCK_OC, BLOCK_HW]
    acc = tl.dot(w_tile, tl.trans(x_tile), out_dtype=tl.float32)

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[:, None]

    # fused hardswish + relu
    hs_mid = acc * (acc + 3.0) * (1.0 / 6.0)
    res = tl.where(acc <= 0.0, 0.0, tl.where(acc >= 3.0, acc, hs_mid))

    # store as NHWC: out[n, oh, ow, oc]
    out_base = pid_n * (H_out * W_out * C_out)
    out_idx = out_base + hw_offs[None, :] * C_out + oc_offs[:, None]
    mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_idx, res, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to (C_out, KH, KW, C_in) contiguous = (C_out, K) row-major
        with torch.no_grad():
            w = self.conv.weight.detach()  # (C_out, C_in, KH, KW)
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # (C_out, KH, KW, C_in)
            self.register_buffer('w_packed', w_nhwc.view(out_channels, -1).cuda())
            self.register_buffer('b_packed', self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.cuda()
        N, C_in, H_in, W_in = x.shape
        KH = self.kernel_size
        KW = self.kernel_size
        C_out = self.out_channels
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H_in, W_in, C_in)

        out_nhwc = torch.empty((N, H_out, W_out, C_out), device=x.device, dtype=x.dtype)

        C_in_K = C_in * KH * KW

        grid = lambda META: (
            N * triton.cdiv(H_out * W_out, META['BLOCK_HW']),
            triton.cdiv(C_out, META['BLOCK_OC']),
        )

        conv2d_nhwc_im2col_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, out_nhwc,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            KH, KW,
            C_in_K,
        )

        # Permute back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out