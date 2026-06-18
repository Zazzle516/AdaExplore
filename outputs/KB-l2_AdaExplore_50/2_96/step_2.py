import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_maxpool_mean_kernel(
    x_ptr, out_ptr,
    N, C,
    D, H, W,                # conv-transpose output spatial dims
    PD, PH, PW,             # pooled spatial dims
    scale: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    inv_count,
    K: tl.constexpr,        # pool kernel size
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    spatial_in = D * H * W
    base_in = (n * C + c) * spatial_in
    pooled_total = PD * PH * PW

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    NEG_INF = float("-inf")

    for start in range(0, pooled_total, BLOCK):
        idx = start + offs
        mask = idx < pooled_total
        pw = idx % PW
        tmp = idx // PW
        ph = tmp % PH
        pd = tmp // PH
        d0 = pd * K
        h0 = ph * K
        w0 = pw * K

        m = tl.full((BLOCK,), NEG_INF, dtype=tl.float32)
        # unrolled 2x2x2 (K=2)
        for kd in tl.static_range(0, K):
            for kh in tl.static_range(0, K):
                for kw in tl.static_range(0, K):
                    in_idx = (d0 + kd) * (H * W) + (h0 + kh) * W + (w0 + kw)
                    v = tl.load(x_ptr + base_in + in_idx, mask=mask, other=NEG_INF)
                    m = tl.maximum(m, v)
        m = tl.where(mask, m, 0.0)
        acc += m

    total_sum = tl.sum(acc, axis=0)
    mean = total_sum * inv_count * scale
    mean = tl.minimum(tl.maximum(mean, clamp_min), clamp_max)
    tl.store(out_ptr + n * C + c, mean)


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
        K = self.maxpool_kernel_size
        PD = D // K
        PH = H // K
        PW = W // K
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        pooled_total = PD * PH * PW
        inv_count = 1.0 / pooled_total
        # BLOCK chosen relative to pooled_total
        if pooled_total <= 1024:
            BLOCK = 1024
        elif pooled_total <= 2048:
            BLOCK = 2048
        else:
            BLOCK = 4096
        grid = (N * C,)
        _fused_maxpool_mean_kernel[grid](
            x, out,
            N, C,
            D, H, W,
            PD, PH, PW,
            float(self.scale),
            float(self.clamp_min),
            float(self.clamp_max),
            inv_count,
            K=K,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=2,
        )
        return out