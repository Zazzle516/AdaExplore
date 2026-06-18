import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 512}, num_warps=2, num_stages=2),
    ],
    key=['C', 'D', 'H', 'W'],
)
@triton.jit
def _fused_maxpool_mean_kernel(
    x_ptr, out_ptr,
    C, D, H, W,
    PD, PH, PW,
    clamp_min, clamp_max, inv_count_scale,
    BLOCK: tl.constexpr,
    PK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * D * H * W

    pool_total = PD * PH * PW
    offs = tl.arange(0, BLOCK)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    HW = H * W
    PHPW = PH * PW
    NEG_INF = -3.4e38

    for start in range(0, pool_total, BLOCK):
        idx = start + offs
        mask = idx < pool_total
        pd = idx // PHPW
        rem = idx % PHPW
        ph = rem // PW
        pw = rem % PW
        d0 = pd * PK
        h0 = ph * PK
        w0 = pw * PK

        m = tl.zeros((BLOCK,), dtype=tl.float32) + NEG_INF
        for ddd in tl.static_range(0, PK):
            for hhh in tl.static_range(0, PK):
                for www in tl.static_range(0, PK):
                    addr = base + (d0 + ddd) * HW + (h0 + hhh) * W + (w0 + www)
                    v = tl.load(x_ptr + addr, mask=mask, other=NEG_INF)
                    m = tl.maximum(m, v)
        acc += tl.where(mask, m, 0.0)

    total_sum = tl.sum(acc, axis=0)
    mean = total_sum * inv_count_scale
    mean = tl.minimum(tl.maximum(mean, clamp_min), clamp_max)
    tl.store(out_ptr + pid, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = float(scale)
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, D, H, W = x.shape
        PK = self.maxpool_kernel_size
        PD = D // PK
        PH = H // PK
        PW = W // PK
        pool_total = PD * PH * PW
        inv_count_scale = self.scale / pool_total
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

        grid = (N * C,)
        _fused_maxpool_mean_kernel[grid](
            x, out,
            C, D, H, W,
            PD, PH, PW,
            self.clamp_min, self.clamp_max,
            inv_count_scale,
            PK=PK,
        )
        return out