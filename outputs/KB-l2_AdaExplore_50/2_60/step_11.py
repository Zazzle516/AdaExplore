import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_swish_gn_hardswish_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, S, G, C_per_G, group_elems, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * C * S + g * C_per_G * S

    sum_val = 0.0
    sum_sq = 0.0
    offs = tl.arange(0, BLOCK_SIZE)

    # Pass 1: apply swish on the fly, accumulate sum & sum_sq
    for start in range(0, group_elems, BLOCK_SIZE):
        idx = start + offs
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sw = x * tl.sigmoid(x)
        sw = tl.where(mask, sw, 0.0)
        sum_val += tl.sum(sw, axis=0)
        sum_sq += tl.sum(sw * sw, axis=0)

    inv_n = 1.0 / group_elems
    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: recompute swish, normalize, affine, hardswish
    for start in range(0, group_elems, BLOCK_SIZE):
        idx = start + offs
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sw = x * tl.sigmoid(x)

        c_in_g = idx // S
        c_global = g * C_per_G + c_in_g
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)

        y = (sw - mean) * rstd * w + b
        t = y + 3.0
        t = tl.minimum(tl.maximum(t, 0.0), 6.0)
        out = y * t * (1.0 / 6.0)
        tl.store(out_ptr + base + idx, out, mask=mask)


def triton_fused_swish_gn_hardswish(x, weight, bias, G, eps):
    x = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    C_per_G = C // G
    out = torch.empty_like(x)
    group_elems = C_per_G * S

    if group_elems >= 4096:
        BLOCK = 2048
        num_warps = 8
    elif group_elems >= 1024:
        BLOCK = 1024
        num_warps = 4
    elif group_elems >= 512:
        BLOCK = 512
        num_warps = 4
    else:
        BLOCK = 256
        num_warps = 2

    grid = (N * G,)
    fused_swish_gn_hardswish_kernel[grid](
        x, weight, bias, out,
        N, C, S, G, C_per_G, group_elems, eps,
        BLOCK_SIZE=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_fused_swish_gn_hardswish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return x