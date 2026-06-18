import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=3),
    ],
    key=['GROUP_SIZE', 'CPG'],
)
@triton.jit
def swish_groupnorm_hardswish_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    N, C, S,
    G,
    eps,
    GROUP_SIZE: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * C * S + g * CPG * S

    offs = tl.arange(0, BLOCK)

    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    num_iters = (GROUP_SIZE + BLOCK - 1) // BLOCK

    for i in range(num_iters):
        idx = i * BLOCK + offs
        mask = idx < GROUP_SIZE
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sw = x * tl.sigmoid(x)
        sw = tl.where(mask, sw, 0.0)
        sum_val += tl.sum(sw)
        sum_sq += tl.sum(sw * sw)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pre-load per-group channel weights/biases (CPG values)
    cpg_offs = tl.arange(0, CPG)
    w_vec = tl.load(weight_ptr + g * CPG + cpg_offs).to(tl.float32)
    b_vec = tl.load(bias_ptr + g * CPG + cpg_offs).to(tl.float32)

    for i in range(num_iters):
        idx = i * BLOCK + offs
        mask = idx < GROUP_SIZE
        c_in_g = idx // S  # [BLOCK]
        # gather w,b via where over CPG
        # Use broadcast comparison + sum trick
        w = tl.sum(tl.where(cpg_offs[None, :] == c_in_g[:, None], w_vec[None, :], 0.0), axis=1)
        b = tl.sum(tl.where(cpg_offs[None, :] == c_in_g[:, None], b_vec[None, :], 0.0), axis=1)

        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sw = x * tl.sigmoid(x)
        norm = (sw - mean) * rstd
        out = norm * w + b
        t = out + 3.0
        t = tl.minimum(tl.maximum(t, 0.0), 6.0)
        res = out * t * (1.0 / 6.0)
        tl.store(y_ptr + base + idx, res, mask=mask)


def fused_swish_gn_hswish(x, weight, bias, G, eps):
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // G
    GROUP_SIZE = CPG * S

    x_flat = x.contiguous().view(N, C, S)
    y = torch.empty_like(x_flat)

    grid = (N * G,)
    swish_groupnorm_hardswish_kernel[grid](
        x_flat, y, weight, bias,
        N, C, S, G,
        eps,
        GROUP_SIZE=GROUP_SIZE,
        CPG=CPG,
    )
    return y.view(N, C, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_swish_gn_hswish(x, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return x