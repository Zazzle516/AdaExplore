import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_scale_sigmoid_gn_kernel(
    x_ptr,           # [N, C, H, W]
    bias_ptr,        # [C]
    scale_ptr,       # [C]
    gn_w_ptr,        # [C]
    gn_b_ptr,        # [C]
    out_ptr,         # [N, C, H, W]
    N, C, H, W,
    G,
    CPG,             # channels per group
    eps,
    BLOCK: tl.constexpr,
    CPG_C: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    HW = H * W
    group_size = CPG * HW

    # base offset for (n, g*CPG, 0, 0)
    base = n * C * HW + g * CPG * HW

    # load per-channel params for this group as vectors of length CPG_C
    c_offs = g * CPG + tl.arange(0, CPG_C)
    c_mask = tl.arange(0, CPG_C) < CPG
    bias_v = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0)   # [CPG_C]
    scale_v = tl.load(scale_ptr + c_offs, mask=c_mask, other=0.0) # [CPG_C]
    gn_w_v = tl.load(gn_w_ptr + c_offs, mask=c_mask, other=0.0)   # [CPG_C]
    gn_b_v = tl.load(gn_b_ptr + c_offs, mask=c_mask, other=0.0)   # [CPG_C]

    num_chunks = (HW + BLOCK - 1) // BLOCK

    # Reshape per-channel vectors as [CPG_C, 1] for broadcasting
    bias_2d = bias_v[:, None]
    scale_2d = scale_v[:, None]
    cmask_2d = c_mask[:, None]

    # ci offsets [CPG_C]
    ci_arange = tl.arange(0, CPG_C)

    sum_acc = tl.zeros([], dtype=tl.float32)
    sumsq_acc = tl.zeros([], dtype=tl.float32)

    # First pass: compute y = sigmoid((x+b)*s), store to out, accumulate stats
    for k in range(0, num_chunks):
        offs = k * BLOCK + tl.arange(0, BLOCK)              # [BLOCK]
        hw_mask = offs < HW                                  # [BLOCK]
        # 2D pointer: [CPG_C, BLOCK]
        ptrs = x_ptr + base + ci_arange[:, None] * HW + offs[None, :]
        mask2d = cmask_2d & hw_mask[None, :]
        x = tl.load(ptrs, mask=mask2d, other=0.0)
        y = (x + bias_2d) * scale_2d
        y = tl.sigmoid(y)
        y_masked = tl.where(mask2d, y, 0.0)
        sum_acc += tl.sum(y_masked)
        sumsq_acc += tl.sum(y_masked * y_masked)
        # store post-sigmoid as scratch
        out_ptrs = out_ptr + base + ci_arange[:, None] * HW + offs[None, :]
        tl.store(out_ptrs, y, mask=mask2d)

    mean = sum_acc / group_size
    var = sumsq_acc / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    w_2d = gn_w_v[:, None]
    bn_2d = gn_b_v[:, None]

    # Second pass: load y from out, normalize, write back
    for k in range(0, num_chunks):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        hw_mask = offs < HW
        out_ptrs = out_ptr + base + ci_arange[:, None] * HW + offs[None, :]
        mask2d = cmask_2d & hw_mask[None, :]
        y = tl.load(out_ptrs, mask=mask2d, other=0.0)
        z = (y - mean) * rstd
        z = z * w_2d + bn_2d
        tl.store(out_ptrs, z, mask=mask2d)


def fused_post_conv(x, bias, scale, gn_w, gn_b, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    assert C % num_groups == 0
    CPG = C // num_groups
    out = torch.empty_like(x)

    # pick BLOCK based on HW
    HW = H * W
    if HW >= 2048:
        BLOCK = 2048
    else:
        BLOCK = triton.next_power_of_2(HW)
        BLOCK = max(BLOCK, 32)

    # CPG_C must be a power of 2 >= CPG for tl.arange
    CPG_C = triton.next_power_of_2(CPG)

    grid = (N * num_groups,)
    fused_bias_scale_sigmoid_gn_kernel[grid](
        x, bias, scale, gn_w, gn_b, out,
        N, C, H, W,
        num_groups, CPG, eps,
        BLOCK=BLOCK,
        CPG_C=CPG_C,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps
        x = x.contiguous()
        out = fused_post_conv(x, bias_flat, scale_flat, gn_w, gn_b, self.num_groups, eps)
        return out