import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
    ],
    key=['B', 'OC_K', 'S_K'],
)
@triton.jit
def conv3d_kernel(
    x_ptr,         # (B, IC, ID, IH, IW)
    w_ptr,         # (OC, IC, KD, KH, KW)
    b_ptr,         # (OC,)
    out_ptr,       # (B, OC, OD, OH, OW)  -> stored as (B, OC, S)
    B, OC_K, S_K,
    ID, IH, IW,
    OD, OH, OW,
    IC: tl.constexpr,
    OC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    S = OD * OH * OW
    mask_s = s_off < S

    # decompose s -> (od, oh, ow)
    od = s_off // (OH * OW)
    rem = s_off - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    x_batch_base = pid_b * IC * ID * IH * IW
    w_oc_base = pid_oc * IC * KD * KH * KW

    for ic in tl.static_range(0, IC):
        x_ic_base = x_batch_base + ic * ID * IH * IW
        w_ic_base = w_oc_base + ic * KD * KH * KW
        for kd in tl.static_range(0, KD):
            id_ = od + kd
            for kh in tl.static_range(0, KH):
                ih = oh + kh
                for kw in tl.static_range(0, KW):
                    iw = ow + kw
                    x_idx = x_ic_base + id_ * (IH * IW) + ih * IW + iw
                    w_idx = w_ic_base + kd * (KH * KW) + kh * KW + kw
                    x_val = tl.load(x_ptr + x_idx, mask=mask_s, other=0.0)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val

    bias = tl.load(b_ptr + pid_oc)
    acc += bias

    out_base = pid_b * OC * S + pid_oc * S
    tl.store(out_ptr + out_base + s_off, acc, mask=mask_s)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
    ],
    key=['B', 'C', 'S'],
)
@triton.jit
def fused_post_conv_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    B, C, S,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = CHANNELS_PER_GROUP * S
    base = pid_b * C * S + pid_g * group_size

    sum_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    sumsq_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)

    c_off = tl.arange(0, CHANNELS_PER_GROUP)

    for s_start in range(0, S, BLOCK_S):
        s_off = s_start + tl.arange(0, BLOCK_S)
        mask = s_off < S
        ptrs = base + c_off[:, None] * S + s_off[None, :]
        x = tl.load(x_ptr + ptrs, mask=mask[None, :], other=0.0).to(tl.float32)
        t = x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = x * t * (1.0 / 6.0)
        sum_acc += tl.sum(hs, axis=1)
        sumsq_acc += tl.sum(hs * hs, axis=1)

    total = CHANNELS_PER_GROUP * S
    group_sum = tl.sum(sum_acc, axis=0)
    group_sumsq = tl.sum(sumsq_acc, axis=0)
    mean = group_sum / total
    var = group_sumsq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    c_global = pid_g * CHANNELS_PER_GROUP + c_off
    gamma = tl.load(gamma_ptr + c_global).to(tl.float32)
    beta = tl.load(beta_ptr + c_global).to(tl.float32)

    per_ch_hs_mean = sum_acc / S
    out_vals = (per_ch_hs_mean - mean) * rstd * gamma + beta

    out_ptrs = pid_b * C + c_global
    tl.store(out_ptr + out_ptrs, out_vals)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        B, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kd, self.kh, self.kw
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        S = OD * OH * OW

        x = x.contiguous()
        w = self.conv.weight.contiguous()
        if self.conv.bias is not None:
            b = self.conv.bias.contiguous()
        else:
            b = torch.zeros(OC, device=x.device, dtype=x.dtype)

        conv_out = torch.empty((B, OC, S), device=x.device, dtype=x.dtype)

        grid = lambda META: (B, OC, (S + META['BLOCK_S'] - 1) // META['BLOCK_S'])
        conv3d_kernel[grid](
            x, w, b, conv_out,
            B, OC, S, ID, IH, IW, OD, OH, OW,
            IC=IC, OC=OC, KD=KD, KH=KH, KW=KW,
        )

        out = torch.empty((B, OC), device=x.device, dtype=x.dtype)
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        groups = self.num_groups
        channels_per_group = OC // groups

        grid2 = (B, groups)
        fused_post_conv_kernel[grid2](
            conv_out, gamma, beta, out,
            B, OC, S,
            GROUPS=groups,
            CHANNELS_PER_GROUP=channels_per_group,
            EPS=self.eps,
        )
        return out