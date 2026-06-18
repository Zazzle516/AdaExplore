import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gn_lrelu_double_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    N, C, G, CH_PER_GROUP,
    eps, neg_slope,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * C + g * CH_PER_GROUP
    offs = tl.arange(0, BLOCK)
    mask = offs < CH_PER_GROUP

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x / CH_PER_GROUP
    var = sum_x2 / CH_PER_GROUP - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + g * CH_PER_GROUP + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + g * CH_PER_GROUP + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * gamma + beta
    y = tl.where(y >= 0, y, y * neg_slope)
    y = y + y

    tl.store(out_ptr + base + offs, y, mask=mask)


def fused_gn_lrelu_double(x, gamma, beta, num_groups, eps, neg_slope):
    N, C = x.shape
    CH_PER_GROUP = C // num_groups
    BLOCK = triton.next_power_of_2(CH_PER_GROUP)
    out = torch.empty_like(x)
    grid = (N * num_groups,)
    gn_lrelu_double_kernel[grid](
        x, gamma, beta, out,
        N, C, num_groups, CH_PER_GROUP,
        eps, neg_slope,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.num_groups = num_groups
        self.eps = eps
        self.neg_slope = negative_slope

    def forward(self, x):
        x = self.fc(x)
        x = fused_gn_lrelu_double(
            x.contiguous(), self.gn.weight, self.gn.bias,
            self.num_groups, self.eps, self.neg_slope,
        )
        return x