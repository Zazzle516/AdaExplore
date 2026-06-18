import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_leaky_double_row_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    C, G, CH_PER_G: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
    eps, negative_slope,
    BLOCK: tl.constexpr,  # = GROUPS_PER_PROG * CH_PER_G
):
    pid = tl.program_id(0)  # batch
    gid = tl.program_id(1)  # group block index

    base = pid * C + gid * GROUPS_PER_PROG * CH_PER_G

    offs = tl.arange(0, BLOCK)
    # Reshape into [GROUPS_PER_PROG, CH_PER_G]
    x = tl.load(x_ptr + base + offs).to(tl.float32)
    x2d = tl.reshape(x, (GROUPS_PER_PROG, CH_PER_G))

    sum_x = tl.sum(x2d, axis=1)
    sum_x2 = tl.sum(x2d * x2d, axis=1)
    mean = sum_x / CH_PER_G
    var = sum_x2 / CH_PER_G - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    g_offs = gid * GROUPS_PER_PROG * CH_PER_G + offs
    gamma = tl.load(gamma_ptr + g_offs).to(tl.float32)
    beta = tl.load(beta_ptr + g_offs).to(tl.float32)
    gamma2d = tl.reshape(gamma, (GROUPS_PER_PROG, CH_PER_G))
    beta2d = tl.reshape(beta, (GROUPS_PER_PROG, CH_PER_G))

    y = (x2d - mean[:, None]) * rstd[:, None] * gamma2d + beta2d
    y = tl.where(y >= 0, y, y * negative_slope)
    y = y + y

    y_flat = tl.reshape(y, (BLOCK,))
    tl.store(out_ptr + base + offs, y_flat)


def gn_leaky_double(x, gamma, beta, num_groups, eps, negative_slope):
    B, C = x.shape
    ch_per_g = C // num_groups
    # choose groups per program
    if ch_per_g <= 16:
        gpp = 16
    elif ch_per_g <= 32:
        gpp = 8
    elif ch_per_g <= 64:
        gpp = 4
    else:
        gpp = 1
    while num_groups % gpp != 0:
        gpp //= 2
    if gpp < 1:
        gpp = 1

    BLOCK = gpp * ch_per_g
    num_g_blocks = num_groups // gpp

    out = torch.empty_like(x)
    grid = (B, num_g_blocks)

    if BLOCK <= 64:
        nw = 1
    elif BLOCK <= 256:
        nw = 2
    elif BLOCK <= 1024:
        nw = 4
    else:
        nw = 8

    gn_leaky_double_row_kernel[grid](
        x, gamma, beta, out,
        C, num_groups, ch_per_g,
        gpp,
        eps, negative_slope,
        BLOCK=BLOCK,
        num_warps=nw,
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
        x = x.contiguous()
        # cuBLAS matmul + bias
        x = torch.addmm(self.fc.bias, x, self.fc.weight.t())
        x = gn_leaky_double(
            x,
            self.gn.weight.contiguous(),
            self.gn.bias.contiguous(),
            self.num_groups,
            self.eps,
            self.negative_slope,
        )
        return x