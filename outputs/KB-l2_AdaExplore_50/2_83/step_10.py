import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Note: min(x, min_value) followed by clamp(x, min=min_value, max=max_value)
# When min_value <= max_value:
#   y = min(x, min_value) -> y <= min_value
#   clamp(y, min_value, max_value) -> max(y, min_value) since y <= min_value <= max_value
#   = max(min(x, min_value), min_value) = min_value (when min_value <= max_value)
# Wait: y = min(x, min_value). If x >= min_value, y = min_value. If x < min_value, y = x.
# Then clamp(y, min_value, max_value): if y < min_value -> min_value. So result is always min_value.
# So the output is constant min_value everywhere (before dropout).
# But we should still execute the conv and groupnorm at runtime per safety contract.
# Actually re-read: "Every operator in the reference forward must execute at runtime"
# So we need to actually run conv and groupnorm. The min/clamp result is mathematically min_value.
# Then dropout: in eval mode it's identity, in train mode it scales/zeros.
# We'll execute conv + groupnorm but the elementwise tail can be simplified since result is constant.
# Actually we still need to run them since they have side effects... no, they're pure. But the safety
# contract says ops must execute at runtime. So let's run conv+groupnorm, then apply the fused tail.


@triton.jit
def fused_min_clamp_kernel(
    x_ptr, out_ptr, n_elements,
    MIN_VAL: tl.constexpr, MAX_VAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # min(x, MIN_VAL)
    y = tl.minimum(x, MIN_VAL)
    # clamp(y, MIN_VAL, MAX_VAL)
    y = tl.maximum(y, MIN_VAL)
    y = tl.minimum(y, MAX_VAL)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def groupnorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, S, G, CPG,
    eps,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * S
    base = n * C * S + g * CPG * S

    # compute mean and var
    sum_val = 0.0
    sum_sq = 0.0
    count = group_size

    for off in range(0, group_size, BLOCK_S):
        idx = off + tl.arange(0, BLOCK_S)
        mask = idx < group_size
        vals = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)

    mean = sum_val / count
    var = sum_sq / count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # apply normalization
    for off in range(0, group_size, BLOCK_S):
        idx = off + tl.arange(0, BLOCK_S)
        mask = idx < group_size
        vals = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        # channel index within group
        c_local = idx // S  # 0..CPG-1
        c_global = g * CPG + c_local
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)
        normalized = (vals - mean) * rstd * w + b
        tl.store(out_ptr + base + idx, normalized, mask=mask)


def triton_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    S = 1
    for s in spatial:
        S *= s
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty_like(x_flat)
    CPG = C // groups
    group_size = CPG * S
    # pick BLOCK_S
    BLOCK_S = 1024
    grid = (N * groups,)
    groupnorm_kernel[grid](
        x_flat, weight, bias, out,
        N, C, S, groups, CPG,
        eps,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )
    return out.view(x.shape)


def fused_min_clamp(x, min_value, max_value):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_min_clamp_kernel[grid](
        x, out, n,
        MIN_VAL=float(min_value),
        MAX_VAL=float(max_value),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        x = triton_groupnorm(x, self.norm.weight, self.norm.bias, self.groups, self.eps)
        x = fused_min_clamp(x, self.min_value, self.max_value)
        x = self.dropout(x)
        return x