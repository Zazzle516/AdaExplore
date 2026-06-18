import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused Conv3d + HardSwish + GroupNorm + spatial mean
# One program per (batch, group). Computes conv directly (no im2col).
# Conv params: IC=3, OC=16, K=4, num_groups=4 => CPG=4
# Input spatial: D=16,H=32,W=32 ; Output spatial: D=13,H=29,W=29 => S=10933

@triton.jit
def fused_conv_gn_kernel(
    x_ptr,           # (B, IC, ID, IH, IW)
    w_ptr,           # (OC, IC, KD, KH, KW)
    b_ptr,           # (OC,)
    gamma_ptr,       # (OC,)
    beta_ptr,        # (OC,)
    out_ptr,         # (B, OC)
    B, IC, ID, IH, IW,
    OD, OH, OW,
    eps: tl.constexpr,
    IC_C: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    CPG: tl.constexpr,
    OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    OH_C: tl.constexpr,
    OW_C: tl.constexpr,
    OD_C: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    S = OD * OH * OW
    HW = OH * OW

    # Output channels handled by this program
    c_offs = g * CPG + tl.arange(0, CPG)  # (CPG,)

    # Load weights for this group's output channels: (CPG, IC*KD*KH*KW)
    K_VOL: tl.constexpr = IC_C * KD * KH * KW
    k_range = tl.arange(0, K_VOL)
    w_ptrs = w_ptr + c_offs[:, None] * K_VOL + k_range[None, :]
    w_vals = tl.load(w_ptrs).to(tl.float32)  # (CPG, K_VOL)

    bias_vals = tl.load(b_ptr + c_offs).to(tl.float32)  # (CPG,)

    # Decode k_range into (ic, kd, kh, kw)
    kw_idx = k_range % KW
    tmp1 = k_range // KW
    kh_idx = tmp1 % KH
    tmp2 = tmp1 // KH
    kd_idx = tmp2 % KD
    ic_idx = tmp2 // KD

    # Accumulators
    ch_sum = tl.zeros((CPG,), dtype=tl.float32)
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    base_x = b * (IC_C * ID * IH * IW)

    s_offs = tl.arange(0, BLOCK_S)
    num_chunks = (S + BLOCK_S - 1) // BLOCK_S
    inv6 = 1.0 / 6.0

    for chunk in range(0, num_chunks):
        s_cur = chunk * BLOCK_S + s_offs  # (BLOCK_S,)
        mask_s = s_cur < S

        # Decode s_cur into (od, oh, ow)
        ow = s_cur % OW
        tmp = s_cur // OW
        oh = tmp % OH
        od = tmp // OH

        # Compute input indices for the K_VOL kernel positions
        # in_d = od + kd, in_h = oh + kh, in_w = ow + kw
        # input offset = base_x + ic*ID*IH*IW + in_d*IH*IW + in_h*IW + in_w
        # Shape: (BLOCK_S, K_VOL)
        in_d = od[:, None] + kd_idx[None, :]  # (BLOCK_S, K_VOL)
        in_h = oh[:, None] + kh_idx[None, :]
        in_w = ow[:, None] + kw_idx[None, :]
        ic_b = ic_idx[None, :]  # (1, K_VOL)

        in_off = (ic_b * (ID * IH * IW)
                  + in_d * (IH * IW)
                  + in_h * IW
                  + in_w)
        ptrs = x_ptr + base_x + in_off
        mask = mask_s[:, None]
        x_vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)  # (BLOCK_S, K_VOL)

        # Compute conv: (BLOCK_S, K_VOL) @ (K_VOL, CPG) -> (BLOCK_S, CPG)
        # Use tl.dot
        w_t = tl.trans(w_vals)  # (K_VOL, CPG)
        conv_out = tl.dot(x_vals, w_t)  # (BLOCK_S, CPG)
        conv_out = conv_out + bias_vals[None, :]

        # HardSwish
        t = conv_out + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = conv_out * t * inv6  # (BLOCK_S, CPG)

        hs = tl.where(mask_s[:, None], hs, 0.0)

        ch_sum += tl.sum(hs, axis=0)  # (CPG,)
        sum_val += tl.sum(hs)
        sumsq_val += tl.sum(hs * hs)

    n = (S * CPG).to(tl.float32)
    mean = sum_val / n
    var = sumsq_val / n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    s_f = S.to(tl.float32)
    gamma = tl.load(gamma_ptr + c_offs).to(tl.float32)
    beta = tl.load(beta_ptr + c_offs).to(tl.float32)
    out_val = (ch_sum / s_f - mean) * rstd * gamma + beta
    tl.store(out_ptr + b * OC + c_offs, out_val)


# Fallback: post-conv kernel (when fused conv isn't suitable)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_post_conv_kernel(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    B, C, S,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CPG: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    c_offs = tl.arange(0, CPG) + g * CPG
    s_offs = tl.arange(0, BLOCK_S)

    ch_sum = tl.zeros((CPG,), dtype=tl.float32)
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    num_chunks = (S + BLOCK_S - 1) // BLOCK_S
    base = b * (C * S)
    for chunk in range(0, num_chunks):
        s_cur = chunk * BLOCK_S + s_offs
        mask_s = s_cur < S

        ptrs = x_ptr + base + c_offs[:, None] * S + s_cur[None, :]
        mask = mask_s[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)

        t = x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = x * t * (1.0 / 6.0)
        hs = tl.where(mask, hs, 0.0)

        ch_sum += tl.sum(hs, axis=1)
        sum_val += tl.sum(hs)
        sumsq_val += tl.sum(hs * hs)

    n = (S * CPG).to(tl.float32)
    mean = sum_val / n
    var = sumsq_val / n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    s_f = S.to(tl.float32)
    gamma = tl.load(gamma_ptr + c_offs).to(tl.float32)
    beta = tl.load(beta_ptr + c_offs).to(tl.float32)
    out_val = (ch_sum / s_f - mean) * rstd * gamma + beta
    tl.store(out_ptr + b * C + c_offs, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5
        self.cpg = out_channels // num_groups

        # Decide whether to use the fused conv path
        # Requirements for fused path:
        #  - CPG is a power of 2 and >= 16 for tl.dot (or we pad)
        #  - Reasonable K_VOL
        self.use_fused = (
            isinstance(kernel_size, int)
            and self.cpg >= 1
            and in_channels * kernel_size * kernel_size * kernel_size <= 512
        )

    def forward(self, x):
        if not self.use_fused:
            return self._fallback(x)

        B, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        OC = self.out_channels
        CPG = self.cpg

        x = x.contiguous()
        w = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous() if self.conv.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)

        out = torch.empty((B, OC), device=x.device, dtype=x.dtype)

        # tl.dot requires inner dim >=16 on most archs. CPG=4 < 16 — tl.dot may not be allowed.
        # Use manual matmul via broadcasting if CPG small. For safety, fall back to non-fused if CPG<16.
        if CPG < 16:
            return self._fallback_with(x, w, bias)

        grid = (B, self.num_groups)
        fused_conv_gn_kernel[grid](
            x, w, bias,
            self.group_norm.weight, self.group_norm.bias,
            out,
            B, IC, ID, IH, IW,
            OD, OH, OW,
            eps=self.eps,
            IC_C=IC,
            KD=KD, KH=KH, KW=KW,
            CPG=CPG,
            OC=OC,
            BLOCK_S=256,
            OH_C=OH, OW_C=OW, OD_C=OD,
            num_warps=4,
            num_stages=2,
        )
        return out

    def _fallback(self, x):
        x = self.conv(x)
        B, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(B, C, S)
        out = torch.empty((B, C), device=x.device, dtype=x.dtype)
        cpg = C // self.num_groups
        grid = (B, self.num_groups)
        fused_post_conv_kernel[grid](
            x_flat, out,
            self.group_norm.weight, self.group_norm.bias,
            B, C, S,
            eps=self.eps,
            CPG=cpg,
        )
        return out

    def _fallback_with(self, x, w, bias):
        # Do conv via torch, then fused post
        y = F.conv3d(x, w, bias)
        B, C, D, H, W = y.shape
        S = D * H * W
        y_flat = y.contiguous().view(B, C, S)
        out = torch.empty((B, C), device=y.device, dtype=y.dtype)
        cpg = C // self.num_groups
        grid = (B, self.num_groups)
        fused_post_conv_kernel[grid](
            y_flat, out,
            self.group_norm.weight, self.group_norm.bias,
            B, C, S,
            eps=self.eps,
            CPG=cpg,
        )
        return out