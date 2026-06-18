import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, y_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    S_out = D_out * H_out * W_out
    HW_out = H_out * W_out

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S_out

    od = s_offs // HW_out
    rem = s_offs % HW_out
    oh = rem // W_out
    ow = rem % W_out

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_S, BLOCK_OC), dtype=tl.float32)

    # iterate over kernel positions
    for kd in tl.static_range(0, KD):
        id_ = od - kd
        valid_d = (id_ >= 0) & (id_ < D_in)
        for kh in tl.static_range(0, KH):
            ih = oh - kh
            valid_h = (ih >= 0) & (ih < H_in)
            for kw in tl.static_range(0, KW):
                iw = ow - kw
                valid_w = (iw >= 0) & (iw < W_in)
                valid = valid_d & valid_h & valid_w & s_mask
                # input offset for spatial: id_*H_in*W_in + ih*W_in + iw
                spatial_in = id_ * H_in * W_in + ih * W_in + iw  # [BLOCK_S]
                # iterate over IC
                for ic in range(0, IC):
                    # load input scalar per spatial - but it depends on n,ic,spatial_in
                    x_offset = pid_n * IC * D_in * H_in * W_in + ic * D_in * H_in * W_in + spatial_in
                    x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)  # [BLOCK_S]
                    # weight: [IC, OC, KD, KH, KW]
                    w_offset = ic * OC * KD * KH * KW + oc_offs * KD * KH * KW + kd * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    acc += x_val[:, None] * w_val[None, :]

    # store: y is [N, OC, D_out, H_out, W_out]
    y_offset = pid_n * OC * S_out + oc_offs[None, :] * S_out + s_offs[:, None]
    store_mask = s_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptr + y_offset, acc, mask=store_mask)


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, S, CPG,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = CPG * S
    base = pid_n * C * S + pid_g * CPG * S

    sum_val = 0.0
    sum_sq = 0.0

    for c_off in range(0, CPG):
        row_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            sum_val += tl.sum(tl.where(mask, x, 0.0), axis=0)
            sum_sq += tl.sum(tl.where(mask, x * x, 0.0), axis=0)

    inv_n = 1.0 / group_size
    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c_off in range(0, CPG):
        c_idx = pid_g * CPG + c_off
        w = tl.load(weight_ptr + c_idx)
        b = tl.load(bias_ptr + c_idx)
        row_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            y = (x - mean) * rstd * w + b
            tl.store(y_ptr + row_base + offs, y, mask=mask)


def conv_transpose3d_triton(x, weight):
    N, IC, D_in, H_in, W_in = x.shape
    IC2, OC, KD, KH, KW = weight.shape
    assert IC == IC2
    D_out = D_in + KD - 1
    H_out = H_in + KH - 1
    W_out = W_in + KW - 1
    S_out = D_out * H_out * W_out

    y = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_S = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(S_out, BLOCK_S))
    conv_transpose3d_kernel[grid](
        x, weight, y,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )
    return y


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // groups
    x_c = x.contiguous()
    y = torch.empty_like(x_c)

    BLOCK_S = 2048
    grid = (N, groups)
    fused_relu_groupnorm_kernel[grid](
        x_c, y, weight, bias,
        N, C, S, CPG,
        eps,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.eps = 1e-5
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        # Use torch's conv_transpose since custom kernel is for demo; fallback
        # Actually use our custom kernel
        x = conv_transpose3d_triton(x, self.conv_transpose.weight)
        x = fused_relu_groupnorm(x, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return x