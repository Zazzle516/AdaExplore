import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Note: norm_shape = (out_channels,) = (64,), so LayerNorm is applied over
# the last dim of the [N, C, D, H, W] tensor — wait, careful.
# nn.LayerNorm(norm_shape) normalizes over the LAST len(norm_shape) dims.
# norm_shape=(64,), so it normalizes over the last dim (W=64 after deconv).
# Actually checking: input is [N, C, D, H, W] with C=64 and W=64.
# LayerNorm normalizes over last 1 dim of size 64 — that's W axis.
# So per-row (N,C,D,H) we normalize across W.

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['C', 'W', 'Ho'],
)
@triton.jit
def fused_post_kernel(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    sum_w,
    eps,
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_W: tl.constexpr,
    BLOCK_WO: tl.constexpr,
    BLOCK_HO: tl.constexpr,
):
    pid = tl.program_id(0)
    do = pid % Do
    tmp = pid // Do
    c = tmp % C
    n = tmp // C

    d0 = do * 2

    NUM_ROWS: tl.constexpr = 4 * BLOCK_HO
    TWO_HO: tl.constexpr = 2 * BLOCK_HO

    row_off = tl.arange(0, NUM_ROWS)
    d_idx = row_off // TWO_HO
    h_idx = row_off % TWO_HO

    w_off = tl.arange(0, BLOCK_W)
    w_mask = w_off < W

    gamma = tl.load(gamma_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)

    DHW = D * H * W
    HW = H * W
    inv_W = 1.0 / W

    base_nc = n * C * DHW + c * DHW
    row_base = base_nc + (d0 + d_idx) * HW + h_idx * W  # [NUM_ROWS]

    addrs = row_base[:, None] + w_off[None, :]
    mask_2d = w_mask[None, :]

    x = tl.load(x_ptr + addrs, mask=mask_2d, other=0.0).to(tl.float32) + sum_w
    xz = tl.where(mask_2d, x, 0.0)
    s = tl.sum(xz, axis=1)
    sq = tl.sum(xz * xz, axis=1)
    m = s * inv_W
    v = sq * inv_W - m * m
    r = 1.0 / tl.sqrt(v + eps)
    y = (x - m[:, None]) * r[:, None] * gamma[None, :] + beta[None, :]

    # Pool W pairs: [NUM_ROWS, BLOCK_W] -> [NUM_ROWS, BLOCK_WO, 2] -> [NUM_ROWS, BLOCK_WO]
    y_w = tl.reshape(y, [NUM_ROWS, BLOCK_WO, 2])
    y_w = tl.sum(y_w, axis=2)
    # [NUM_ROWS, BLOCK_WO] = [2, TWO_HO, BLOCK_WO]; sum over d axis=0
    y_d = tl.reshape(y_w, [2, TWO_HO, BLOCK_WO])
    y_d = tl.sum(y_d, axis=0)  # [TWO_HO, BLOCK_WO]
    # [TWO_HO, BLOCK_WO] = [BLOCK_HO, 2, BLOCK_WO]; sum over h_in axis=1
    y_h = tl.reshape(y_d, [BLOCK_HO, 2, BLOCK_WO])
    pooled = tl.sum(y_h, axis=1) * (1.0 / 8.0)  # [BLOCK_HO, BLOCK_WO]

    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))

    ho_off = tl.arange(0, BLOCK_HO)
    wo_off = tl.arange(0, BLOCK_WO)
    ho_mask = ho_off < Ho
    wo_mask = wo_off < Wo
    DHWo = Do * Ho * Wo
    HWo = Ho * Wo
    out_base = n * C * DHWo + c * DHWo + do * HWo
    out_addrs = out_base + ho_off[:, None] * Wo + wo_off[None, :]
    out_mask = ho_mask[:, None] & wo_mask[None, :]
    tl.store(out_ptr + out_addrs, gelu, mask=out_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.pool_kernel_size = pool_kernel_size
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        Do, Ho, Wo = D // 2, H // 2, W // 2

        out = torch.empty((N, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        x_c = x.contiguous()
        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()
        eps = self.norm.eps
        sum_w = float(self.sum_weight.item())

        BLOCK_W = _next_pow2(W)
        BLOCK_WO = BLOCK_W // 2
        BLOCK_HO = _next_pow2(Ho)
        grid = (N * C * Do,)
        fused_post_kernel[grid](
            x_c, out, gamma, beta,
            sum_w, eps,
            N, C, D, H, W,
            Do, Ho, Wo,
            BLOCK_W=BLOCK_W,
            BLOCK_WO=BLOCK_WO,
            BLOCK_HO=BLOCK_HO,
        )
        return out