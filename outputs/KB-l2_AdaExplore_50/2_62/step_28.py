import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_leaky_double_row_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    C, G, CH_PER_G: tl.constexpr, NUM_GROUPS: tl.constexpr,
    eps, negative_slope,
):
    # One program per batch row. Process all G groups in a 2D tile.
    b = tl.program_id(0)

    # 2D layout: [NUM_GROUPS, CH_PER_G]
    g_idx = tl.arange(0, NUM_GROUPS)[:, None]
    c_idx = tl.arange(0, CH_PER_G)[None, :]
    offs = g_idx * CH_PER_G + c_idx  # [NUM_GROUPS, CH_PER_G]

    base = b * C
    x = tl.load(x_ptr + base + offs).to(tl.float32)

    # reduce along channel axis per group
    sum_x = tl.sum(x, axis=1)         # [NUM_GROUPS]
    sum_x2 = tl.sum(x * x, axis=1)    # [NUM_GROUPS]
    mean = sum_x / CH_PER_G
    var = sum_x2 / CH_PER_G - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs).to(tl.float32)
    beta = tl.load(beta_ptr + offs).to(tl.float32)

    y = (x - mean[:, None]) * rstd[:, None] * gamma + beta
    y = tl.where(y >= 0, y, y * negative_slope)
    y = y + y

    tl.store(out_ptr + base + offs, y)


def gn_leaky_double(x, gamma, beta, num_groups, eps, negative_slope):
    B, C = x.shape
    ch_per_g = C // num_groups
    out = torch.empty_like(x)

    # Pick num_warps based on total tile size
    total = num_groups * ch_per_g
    if total <= 1024:
        nw = 4
    elif total <= 4096:
        nw = 8
    else:
        nw = 16 if total >= 16384 else 8

    grid = (B,)
    gn_leaky_double_row_kernel[grid](
        x, gamma, beta, out,
        C, num_groups, ch_per_g, num_groups,
        eps, negative_slope,
        num_warps=nw,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.fc(x)
        x = gn_leaky_double(
            x.contiguous(),
            self.gn.weight.contiguous(),
            self.gn.bias.contiguous(),
            self.num_groups,
            self.eps,
            self.negative_slope,
        )
        return x