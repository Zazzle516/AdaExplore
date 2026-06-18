import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_leaky_double_row_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    C, G, CH_PER_G: tl.constexpr, GROUPS_PER_PROG: tl.constexpr,
    eps, negative_slope,
    BLOCK: tl.constexpr,  # = CH_PER_G * GROUPS_PER_PROG
):
    pid = tl.program_id(0)  # B
    pid_g = tl.program_id(1)  # G // GROUPS_PER_PROG

    # Load a chunk of GROUPS_PER_PROG groups for this row
    offs = tl.arange(0, BLOCK)
    base = pid * C + pid_g * BLOCK

    x = tl.load(x_ptr + base + offs).to(tl.float32)
    # Reshape into (GROUPS_PER_PROG, CH_PER_G) implicitly via group reduction
    # We use 2D layout
    x2d = tl.reshape(x, (GROUPS_PER_PROG, CH_PER_G))

    sum_x = tl.sum(x2d, axis=1)  # (GROUPS_PER_PROG,)
    sum_x2 = tl.sum(x2d * x2d, axis=1)
    mean = sum_x / CH_PER_G
    var = sum_x2 / CH_PER_G - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + pid_g * BLOCK + offs).to(tl.float32)
    beta = tl.load(beta_ptr + pid_g * BLOCK + offs).to(tl.float32)
    g2d = tl.reshape(gamma, (GROUPS_PER_PROG, CH_PER_G))
    b2d = tl.reshape(beta, (GROUPS_PER_PROG, CH_PER_G))

    y2d = (x2d - mean[:, None]) * rstd[:, None] * g2d + b2d
    y2d = tl.where(y2d >= 0, y2d, y2d * negative_slope)
    y2d = y2d + y2d

    y = tl.reshape(y2d, (BLOCK,))
    tl.store(out_ptr + base + offs, y)


def gn_leaky_double(x, gamma, beta, num_groups, eps, negative_slope):
    B, C = x.shape
    ch_per_g = C // num_groups
    out = torch.empty_like(x)

    # Choose GROUPS_PER_PROG so block stays reasonable
    if ch_per_g <= 16:
        groups_per_prog = 64  # block = 1024
    elif ch_per_g <= 32:
        groups_per_prog = 32  # block = 1024
    elif ch_per_g <= 64:
        groups_per_prog = 16
    elif ch_per_g <= 128:
        groups_per_prog = 8
    else:
        groups_per_prog = 1

    while num_groups % groups_per_prog != 0:
        groups_per_prog //= 2
    if groups_per_prog < 1:
        groups_per_prog = 1

    BLOCK = ch_per_g * groups_per_prog
    num_prog_g = num_groups // groups_per_prog
    grid = (B, num_prog_g)

    if BLOCK <= 128:
        nw = 2
    elif BLOCK <= 512:
        nw = 4
    else:
        nw = 8

    gn_leaky_double_row_kernel[grid](
        x, gamma, beta, out,
        C, num_groups, ch_per_g, groups_per_prog,
        eps, negative_slope,
        BLOCK=BLOCK,
        num_warps=nw,
        num_stages=3,
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
        self._wt_cache = None

    def _get_wt(self):
        w = self.fc.weight
        if self._wt_cache is None or self._wt_cache.data_ptr() != w.data_ptr():
            self._wt_cache = w.t().contiguous()
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous()
        wt = self._get_wt()
        x = torch.addmm(self.fc.bias, x, wt)
        x = gn_leaky_double(
            x.contiguous(),
            self.gn.weight.contiguous(),
            self.gn.bias.contiguous(),
            self.num_groups,
            self.eps,
            self.negative_slope,
        )
        return x