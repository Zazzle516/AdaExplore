import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_leaky_double_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    B, C, G, CH_PER_G: tl.constexpr,
    eps, negative_slope,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # B * G
    b = pid // G
    g = pid % G

    offs = tl.arange(0, BLOCK)
    mask = offs < CH_PER_G

    base = b * C + g * CH_PER_G
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x / CH_PER_G
    var = sum_x2 / CH_PER_G - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + g * CH_PER_G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + g * CH_PER_G + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * gamma + beta
    # leaky relu
    y = tl.where(y >= 0, y, y * negative_slope)
    # x + x
    y = y + y

    tl.store(out_ptr + base + offs, y, mask=mask)


def gn_leaky_double(x, gamma, beta, num_groups, eps, negative_slope):
    B, C = x.shape
    ch_per_g = C // num_groups
    # next power of 2 for BLOCK
    BLOCK = 1
    while BLOCK < ch_per_g:
        BLOCK *= 2
    out = torch.empty_like(x)
    grid = (B * num_groups,)
    gn_leaky_double_kernel[grid](
        x, gamma, beta, out,
        B, C, num_groups, ch_per_g,
        eps, negative_slope,
        BLOCK=BLOCK,
        num_warps=4 if BLOCK <= 256 else 8,
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