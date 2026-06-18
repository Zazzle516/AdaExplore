import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['B', 'OC', 'OD', 'OH', 'OW', 'IC_CONST'],
)
@triton.jit
def conv3d_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_CONST: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    OHW = OH * OW
    OSP = OD * OHW
    sp_mask = sp_offs < OSP

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = tl.arange(0, BLOCK_OC)
    bias = tl.load(b_ptr + oc_offs, mask=oc_offs < OC, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    KHW = KH * KW
    KDHW = KD * KHW

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_TOTAL

        ic_i = k_offs // KDHW
        rem_k = k_offs % KDHW
        kd_i = rem_k // KHW
        rem_kh = rem_k % KHW
        kh_i = rem_kh // KW
        kw_i = rem_kh % KW

        # weight (OC, K_TOTAL)
        w_ptrs = w_ptr + oc_offs[:, None] * K_TOTAL + k_offs[None, :]
        w_mask = (oc_offs[:, None] < OC) & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # x: (BLOCK_K, BLOCK_SP)
        id_ = od[None, :] + kd_i[:, None]
        ih_ = oh[None, :] + kh_i[:, None]
        iw_ = ow[None, :] + kw_i[:, None]
        x_idx = (((pid_b * IC_CONST + ic_i[:, None]) * ID + id_) * IH + ih_) * IW + iw_
        x_mask = k_mask[:, None] & sp_mask[None, :]
        x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    acc += bias[:, None]
    t = acc + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    out = acc * t * (1.0 / 6.0)

    out_mask = (oc_offs[:, None] < OC) & sp_mask[None, :]
    out_idx = (pid_b * OC + oc_offs[:, None]) * OSP + sp_offs[None, :]
    tl.store(out_ptr + out_idx, out, mask=out_mask)


@triton.jit
def groupnorm_mean_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    B, C, S,
    NUM_GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    S_CONST: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    GROUP_SIZE = CHANNELS_PER_GROUP * S_CONST

    sum_x = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    sum_x2 = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)

    c_base = pid_g * CHANNELS_PER_GROUP
    s_offs = tl.arange(0, BLOCK_S)
    c_range = tl.arange(0, CHANNELS_PER_GROUP)

    for s_start in range(0, S_CONST, BLOCK_S):
        s_idx = s_start + s_offs
        mask = s_idx < S_CONST
        # load (CPG, BLOCK_S)
        ptrs = x_ptr + ((pid_b * C + c_base + c_range[:, None]) * S) + s_idx[None, :]
        v = tl.load(ptrs, mask=mask[None, :], other=0.0).to(tl.float32)
        sum_x += tl.sum(v, axis=1)
        sum_x2 += tl.sum(v * v, axis=1)

    total_sum = tl.sum(sum_x, axis=0)
    total_sum2 = tl.sum(sum_x2, axis=0)
    mean = total_sum / GROUP_SIZE
    var = total_sum2 / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # per-channel mean of normalized: (sum_x_c/S - mean) * rstd * gamma + beta
    inv_S = 1.0 / S_CONST
    chan_mean_x = sum_x * inv_S
    gamma = tl.load(gamma_ptr + c_base + c_range).to(tl.float32)
    beta = tl.load(beta_ptr + c_base + c_range).to(tl.float32)
    chan_out = (chan_mean_x - mean) * rstd * gamma + beta

    tl.store(out_ptr + pid_b * C + c_base + c_range, chan_out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.weight = nn.Parameter(conv.weight.data.clone())
        if bias:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        gn = nn.GroupNorm(num_groups, out_channels)
        self.gn_weight = nn.Parameter(gn.weight.data.clone())
        self.gn_bias = nn.Parameter(gn.bias.data.clone())

        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        B, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        OSP = OD * OH * OW
        K_TOTAL = IC * KD * KH * KW

        conv_out = torch.empty((B, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

        # BLOCK_OC must be power of two >= 16 for tl.dot
        BLOCK_OC = 16
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        # weight is (OC, IC, KD, KH, KW) contiguous -> view as (OC, K_TOTAL)
        w_flat = self.weight.view(OC, K_TOTAL)

        grid = lambda meta: (B, (OSP + meta['BLOCK_SP'] - 1) // meta['BLOCK_SP'])
        conv3d_hardswish_kernel[grid](
            x, w_flat, self.bias, conv_out,
            B, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            IC,
            BLOCK_OC,
            K_TOTAL,
        )

        out = torch.empty((B, OC), device=x.device, dtype=torch.float32)
        CPG = OC // self.num_groups

        # Use smaller power-of-two BLOCK_S with runtime tile loop
        if OSP <= 1024:
            BS = 1
            while BS < OSP:
                BS *= 2
            BLOCK_S = BS
        else:
            BLOCK_S = 1024

        nw = 8

        grid2 = (B, self.num_groups)
        groupnorm_mean_kernel[grid2](
            conv_out, self.gn_weight, self.gn_bias, out,
            B, OC, OSP,
            self.num_groups, CPG, OSP,
            BLOCK_S,
            self.eps,
            num_warps=nw,
            num_stages=2,
        )
        return out