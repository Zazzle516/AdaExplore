import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def post_conv_kernel(
    x_ptr,           # [N, C, S] conv output
    mult_ptr,        # [C]
    out_ptr,         # [N, S]
    N, C, S,
    clamp_min,
    clamp_max,
    eps,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, s_block)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # load multiplier [C]
    mult = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)

    # We need per-channel mean and var across S for this batch n.
    # That means we have to loop over s for each c. Better: load full [C, S] tile.
    # For each channel, compute mean and var over all S.
    # Let's do it with a separate loop: outer over channel block, inner over S.

    # Compute per-channel mean and variance over the full S dimension
    # Strategy: for each channel c in [0, BLOCK_C), iterate over s in chunks
    # But here BLOCK_S covers a portion of S. Need full S reduction.
    # Reset: this kernel handles ONE n. We need mean/var over all S per channel.

    # Recompute approach: ignore BLOCK_S tiling; do full S in one program per n.
    # Re-implement: program per (n, c_block), reduce full S.
    pass


@triton.jit
def fused_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    out_ptr,         # [N, S]
    N, C, S,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
    C_NEXT: tl.constexpr,
):
    # one program per n
    n = tl.program_id(0)
    s_block = tl.program_id(1)

    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, C_NEXT)
    c_mask = c_offs < C

    # load multiplier [C]
    mult = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)  # [C_NEXT]

    # First: compute per-channel mean and var over full S for this n.
    # We need to iterate over all S regardless of s_block. So each program
    # redundantly computes stats. To avoid that, we precompute stats in a
    # separate kernel. But for simplicity, do it here.

    # Compute sum and sumsq per channel by iterating over S in BLOCK_S chunks.
    sum_c = tl.zeros([C_NEXT], dtype=tl.float32)
    sumsq_c = tl.zeros([C_NEXT], dtype=tl.float32)

    num_blocks = (S + BLOCK_S - 1) // BLOCK_S
    for sb in range(0, num_blocks):
        s_idx = sb * BLOCK_S + tl.arange(0, BLOCK_S)
        sm = s_idx < S
        # load tile [C_NEXT, BLOCK_S]
        ptrs = x_ptr + n * C * S + c_offs[:, None] * S + s_idx[None, :]
        mask2 = c_mask[:, None] & sm[None, :]
        vals = tl.load(ptrs, mask=mask2, other=0.0)
        # multiply by multiplier
        vals = vals * mult[:, None]
        sum_c += tl.sum(vals, axis=1)
        sumsq_c += tl.sum(vals * vals, axis=1)

    mean_c = sum_c / S
    var_c = sumsq_c / S - mean_c * mean_c
    inv_std = 1.0 / tl.sqrt(var_c + eps)

    # Now process this s_block: load tile, normalize, clamp, multiply, max over C.
    ptrs = x_ptr + n * C * S + c_offs[:, None] * S + s_offs[None, :]
    mask2 = c_mask[:, None] & s_mask[None, :]
    vals = tl.load(ptrs, mask=mask2, other=0.0)
    vals = vals * mult[:, None]
    vals = (vals - mean_c[:, None]) * inv_std[:, None]
    vals = tl.minimum(tl.maximum(vals, clamp_min), clamp_max)
    vals = vals * mult[:, None]
    # mask out invalid channels with -inf
    neg_inf = float('-inf')
    vals = tl.where(c_mask[:, None], vals, neg_inf)
    out = tl.max(vals, axis=0)  # [BLOCK_S]
    tl.store(out_ptr + n * S + s_offs, out, mask=s_mask)


@triton.jit
def conv3d_kernel(
    x_ptr,       # [N, IC, ID, IH, IW]
    w_ptr,       # [OC, IC, KD, KH, KW]
    b_ptr,       # [OC]
    out_ptr,     # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OS = OD * OH * OW
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < OS

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros([BLOCK_OC, BLOCK_SP], dtype=tl.float32)

    for ic in range(0, IC):
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    # input ptr: [n, ic, id_, ih_, iw_]
                    in_idx = pid_n * IC * ID * IH * IW + ic * ID * IH * IW + id_ * IH * IW + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + in_idx, mask=sp_mask, other=0.0)  # [BLOCK_SP]
                    # weight: [oc, ic, kd, kh, kw]
                    w_idx = oc_offs * IC * KD * KH * KW + ic * KD * KH * KW + kd * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # store
    out_idx = pid_n * OC * OS + oc_offs[:, None] * OS + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 16 if OC <= 16 else 32
        # ensure power of two
        if OC <= 16:
            BLOCK_OC = 16
        else:
            BLOCK_OC = 32
        BLOCK_SP = 128

        OS = OD * OH * OW
        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (OS + BLOCK_SP - 1) // BLOCK_SP)
        conv3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
        )

        # Now fused: multiply, instance norm, clamp, multiply, max over C
        S = OD * OH * OW
        x_flat = conv_out.view(N, OC, S)
        mult_flat = self.multiplier.view(-1).contiguous()
        out = torch.empty((N, S), device=x.device, dtype=x.dtype)

        # Choose C_NEXT as next power of 2 >= OC
        C_NEXT = 1
        while C_NEXT < OC:
            C_NEXT *= 2

        BLOCK_S2 = 256
        grid2 = (N, (S + BLOCK_S2 - 1) // BLOCK_S2)
        fused_kernel[grid2](
            x_flat, mult_flat, out,
            N, OC, S,
            self.clamp_min, self.clamp_max, 1e-5,
            BLOCK_S=BLOCK_S2,
            C_NEXT=C_NEXT,
            num_warps=4,
        )

        return out.view(N, OD, OH, OW)