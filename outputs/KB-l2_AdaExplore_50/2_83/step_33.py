import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Note: After conv with kernel_size=3 on (16,64,64), output is (14,62,62).
# Since min(x, 0) then clamp(x, 0, 1) == 0 if x <= 0 else 0 (since x <= 0 from min).
# Wait: min(x, 0) gives values <= 0. Then clamp(x, 0, 1) clamps them to >= 0, so result is 0.
# Actually: min(x, 0) -> y where y <= 0. clamp(y, 0, 1) -> max(min(y,1), 0) = max(y, 0) = 0 (since y<=0).
# So the output is always 0! Then dropout(0) = 0.
# But we must execute the operators. Let's still compute, but we can skip the conv/gn entirely
# at runtime... no wait, safety contract says every op must execute.
# 
# Actually re-reading: "Every operator in the reference forward must execute at runtime on the actual input tensor."
# So we must run conv, gn, min, clamp, dropout. But we still need to produce correct output.
# 
# We'll do the full pipeline but optimize the GN+min+clamp into a fused kernel.
# Since the output is always 0 mathematically, we just need correct numerics.
# But we cannot shortcut. Let's keep the fused kernel approach but optimize it.


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    S, G,
    min_value, max_value, eps,
    C: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    c_start = g * CPG
    group_size = CPG * S

    s_offs = tl.arange(0, BLOCK_S)
    base = n * C * S + c_start * S

    # Pass 1
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for blk in tl.static_range(0, NUM_BLOCKS):
        s_start = blk * BLOCK_S
        offs = s_start + s_offs
        mask = offs < S
        for c_off in tl.static_range(0, CPG):
            ptr = x_ptr + base + c_off * S + offs
            v = tl.load(ptr, mask=mask, other=0.0)
            sum_val += tl.sum(v, axis=0)
            sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2
    for c_off in tl.static_range(0, CPG):
        c = c_start + c_off
        w = tl.load(weight_ptr + c)
        b = tl.load(bias_ptr + c)
        scale = rstd * w
        shift = b - mean * scale
        for blk in tl.static_range(0, NUM_BLOCKS):
            s_start = blk * BLOCK_S
            offs = s_start + s_offs
            mask = offs < S
            ptr = x_ptr + base + c_off * S + offs
            v = tl.load(ptr, mask=mask, other=0.0)
            norm = v * scale + shift
            norm = tl.minimum(norm, min_value)
            norm = tl.maximum(norm, min_value)
            norm = tl.minimum(norm, max_value)
            tl.store(out_ptr + base + c_off * S + offs, norm, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.groups = groups
        self.out_channels = out_channels
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(N, C, S)
        out = torch.empty_like(x_flat)

        G = self.groups
        CPG = C // G

        BLOCK_S = 8192
        NUM_BLOCKS = (S + BLOCK_S - 1) // BLOCK_S

        grid = (N * G,)
        fused_gn_min_clamp_kernel[grid](
            x_flat, out,
            self.norm.weight, self.norm.bias,
            S, G,
            self.min_value, self.max_value,
            self.eps,
            C=C,
            CPG=CPG,
            BLOCK_S=BLOCK_S,
            NUM_BLOCKS=NUM_BLOCKS,
            num_warps=8,
            num_stages=3,
        )
        out = out.view(N, C, D, H, W)
        out = self.dropout(out)
        return out