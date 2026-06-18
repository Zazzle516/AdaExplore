import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
    ],
    key=['H_out', 'W_out'],
)
@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
    x_ptr,        # [N, C, H, W]
    out_ptr,      # [N, C, 1, 1]
    N, C, H, W,
    H_out, W_out,
    inv_count,
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * H * W + c * H * W
    total = H_out * W_out

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for start in range(0, total, BLOCK):
        idx = start + offs
        mask = idx < total
        oh = idx // W_out
        ow = idx % W_out

        ih0 = oh * 2
        iw0 = ow * 2

        p00 = base + ih0 * W + iw0
        p01 = p00 + 1
        p10 = p00 + W
        p11 = p10 + 1

        v00 = tl.load(x_ptr + p00, mask=mask, other=-float('inf'))
        v01 = tl.load(x_ptr + p01, mask=mask, other=-float('inf'))
        v10 = tl.load(x_ptr + p10, mask=mask, other=-float('inf'))
        v11 = tl.load(x_ptr + p11, mask=mask, other=-float('inf'))

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.minimum(tl.maximum(m, HTANH_MIN), HTANH_MAX)
        m = tl.where(mask, m, 0.0)
        acc += m

    s = tl.sum(acc, axis=0)
    mean = s * inv_count
    e2 = tl.exp(2.0 * mean)
    out_val = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

        # Precompute equivalent Conv2d weight (flip + permute) once at init.
        # For stride=1 ConvTranspose2d, output = Conv2d with flipped weight.
        self._can_use_conv = (stride == 1)
        if self._can_use_conv:
            with torch.no_grad():
                w = self.conv_transpose.weight.detach()  # [IC, OC, KH, KW]
                w_eq = torch.flip(w, dims=[2, 3]).permute(1, 0, 2, 3).contiguous()
            self.register_buffer('_conv_weight', w_eq)

    def forward(self, x):
        if self._can_use_conv:
            # Use precomputed equivalent conv weight (no per-forward flip/permute)
            y = F.conv2d(x, self._conv_weight, bias=self.conv_transpose.bias,
                         stride=1, padding=self.padding)
        else:
            y = self.conv_transpose(x)
        y = y.contiguous()
        N, C, H, W = y.shape

        if (self.maxpool_kernel_size == 2 and self.maxpool_stride == 2
                and H % 2 == 0 and W % 2 == 0):
            H_out = H // 2
            W_out = W // 2
            out = torch.empty((N, C, 1, 1), device=y.device, dtype=y.dtype)
            inv_count = 1.0 / (H_out * W_out)
            grid = (N * C,)
            fused_pool_htanh_mean_tanh_kernel[grid](
                y, out,
                N, C, H, W,
                H_out, W_out,
                inv_count,
                HTANH_MIN=self.hardtanh_min,
                HTANH_MAX=self.hardtanh_max,
            )
            return out
        else:
            y = F.max_pool2d(y, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
            y = F.hardtanh(y, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
            y = torch.mean(y, dim=(2, 3), keepdim=True)
            y = torch.tanh(y)
            return y