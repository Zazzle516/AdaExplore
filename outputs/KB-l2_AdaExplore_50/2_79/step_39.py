import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    OS = OD * OH * OW
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < OS

    oc_offs = tl.arange(0, OC)

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros([OC, BLOCK_SP], dtype=tl.float32)

    x_base = pid_n * IC * ID * IH * IW

    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    in_idx = x_base + ic * ID * IH * IW + id_ * IH * IW + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + in_idx, mask=sp_mask, other=0.0)
                    w_idx = oc_offs * IC * KD * KH * KW + ic * KD * KH * KW + kd * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx)
                    acc += w_val[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + oc_offs)
    acc += bias[:, None]

    out_idx = pid_n * OC * OS + oc_offs[:, None] * OS + sp_offs[None, :]
    mask = sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask)


@triton.jit
def fused_norm_max_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    out_ptr,         # [N, S]
    S: tl.constexpr,
    C: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    n = tl.program_id(0)

    c_offs = tl.arange(0, C)
    mult = tl.load(mult_ptr + c_offs)  # [C]

    # compute per-channel mean and var over full S
    sum_c = tl.zeros([C], dtype=tl.float32)
    sumsq_c = tl.zeros([C], dtype=tl.float32)

    base = n * C * S
    for sb in range(NUM_BLOCKS):
        s_idx = tl.arange(0, BLOCK_S) + sb * BLOCK_S
        sm = s_idx < S
        ptrs = x_ptr + base + c_offs[:, None] * S + s_idx[None, :]
        vals = tl.load(ptrs, mask=sm[None, :], other=0.0)
        vals = vals * mult[:, None]
        sum_c += tl.sum(vals, axis=1)
        sumsq_c += tl.sum(vals * vals, axis=1)

    inv_S = 1.0 / S
    mean_c = sum_c * inv_S
    var_c = sumsq_c * inv_S - mean_c * mean_c
    inv_std = 1.0 / tl.sqrt(var_c + eps)

    # second pass: process each block, normalize, clamp, multiply, max over C, store
    for sb in range(NUM_BLOCKS):
        s_offs = tl.arange(0, BLOCK_S) + sb * BLOCK_S
        s_mask = s_offs < S
        ptrs = x_ptr + base + c_offs[:, None] * S + s_offs[None, :]
        vals = tl.load(ptrs, mask=s_mask[None, :], other=0.0)
        vals = vals * mult[:, None]
        vals = (vals - mean_c[:, None]) * inv_std[:, None]
        vals = tl.minimum(tl.maximum(vals, clamp_min), clamp_max)
        vals = vals * mult[:, None]
        out = tl.max(vals, axis=0)
        tl.store(out_ptr + n * S + s_offs, out, mask=s_mask)


@triton.jit
def stats_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    mean_ptr,        # [N, C]
    invstd_ptr,      # [N, C]
    S: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    c_offs = tl.arange(0, C)
    mult = tl.load(mult_ptr + c_offs)

    sum_c = tl.zeros([C], dtype=tl.float32)
    sumsq_c = tl.zeros([C], dtype=tl.float32)

    NUM_BLOCKS: tl.constexpr = (S + BLOCK_S - 1) // BLOCK_S
    base = n * C * S
    for sb in range(NUM_BLOCKS):
        s_idx = tl.arange(0, BLOCK_S) + sb * BLOCK_S
        sm = s_idx < S
        ptrs = x_ptr + base + c_offs[:, None] * S + s_idx[None, :]
        vals = tl.load(ptrs, mask=sm[None, :], other=0.0)
        vals = vals * mult[:, None]
        sum_c += tl.sum(vals, axis=1)
        sumsq_c += tl.sum(vals * vals, axis=1)

    inv_S = 1.0 / S
    mean_c = sum_c * inv_S
    var_c = sumsq_c * inv_S - mean_c * mean_c
    inv_std = 1.0 / tl.sqrt(var_c + eps)

    tl.store(mean_ptr + n * C + c_offs, mean_c)
    tl.store(invstd_ptr + n * C + c_offs, inv_std)


@triton.jit
def apply_max_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    mean_ptr,        # [N, C]
    invstd_ptr,      # [N, C]
    out_ptr,         # [N, S]
    S: tl.constexpr,
    C: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    s_block = tl.program_id(1)

    c_offs = tl.arange(0, C)
    mult = tl.load(mult_ptr + c_offs)
    mean_c = tl.load(mean_ptr + n * C + c_offs)
    inv_std = tl.load(invstd_ptr + n * C + c_offs)

    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    base = n * C * S
    ptrs = x_ptr + base + c_offs[:, None] * S + s_offs[None, :]
    vals = tl.load(ptrs, mask=s_mask[None, :], other=0.0)
    vals = vals * mult[:, None]
    vals = (vals - mean_c[:, None]) * inv_std[:, None]
    vals = tl.minimum(tl.maximum(vals, clamp_min), clamp_max)
    vals = vals * mult[:, None]
    out = tl.max(vals, axis=0)
    tl.store(out_ptr + n * S + s_offs, out, mask=s_mask)


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

        OS = OD * OH * OW
        BLOCK_SP = 128

        grid = (N, (OS + BLOCK_SP - 1) // BLOCK_SP)
        conv3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )

        S = OS
        x_flat = conv_out.view(N, OC, S)
        mult_flat = self.multiplier.view(-1).contiguous()
        out = torch.empty((N, S), device=x.device, dtype=x.dtype)

        # Choose BLOCK_S that evenly divides S=12600 if possible
        # 12600 = 2^3 * 3^2 * 5^2 * 7
        # Use 1024 (16 blocks with masked tail) or use divisor 1400 (9 blocks)
        BLOCK_S = 1024
        NUM_BLOCKS = (S + BLOCK_S - 1) // BLOCK_S

        fused_norm_max_kernel[(N,)](
            x_flat, mult_flat, out,
            S, OC,
            self.clamp_min, self.clamp_max, 1e-5,
            BLOCK_S=BLOCK_S,
            NUM_BLOCKS=NUM_BLOCKS,
            num_warps=8,
            num_stages=2,
        )

        return out.view(N, OD, OH, OW)