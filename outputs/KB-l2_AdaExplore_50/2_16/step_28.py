import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    b_ptr,           # [OC]
    out_ptr,         # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    add_value, scale,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program: (n, oc_block, sp_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oh = sp_offs // OW
    ow = sp_offs % OW

    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    # For stride=2, padding=1, kernel=3, output_padding=1:
    # out[oh, ow] = sum over (kh, kw) where (oh + 1 - kh) % 2 == 0 and similar for ow
    # ih = (oh + 1 - kh) / 2, must be in [0, IH)
    # iw = (iw + 1 - kw) / 2, must be in [0, IW)
    # Possible kh values: kh such that (oh + 1 - kh) is even and non-negative and < 2*IH
    # Since kh in {0,1,2}, parity: oh+1-kh even => kh and oh+1 same parity
    # So 2 valid kh values (out of 3), and similarly 2 valid kw.

    # We'll precompute the (kh, ih) pairs for each oh.
    # kh0 = (oh + 1) % 2  (parity-matched, smallest)
    # kh1 = kh0 + 2
    # ih corresponding: ih = (oh + 1 - kh) // 2
    
    acc = tl.zeros([BLOCK_OC, BLOCK_SP], dtype=tl.float32)

    # iterate over the 4 (kh, kw) contributing pairs
    for kh_idx in tl.static_range(0, 2):
        kh0 = (oh + 1) % 2  # [BLOCK_SP]
        kh = kh0 + kh_idx * 2  # [BLOCK_SP]
        ih = (oh + 1 - kh) // 2  # [BLOCK_SP]
        kh_valid = (kh < 3) & (ih >= 0) & (ih < IH)

        for kw_idx in tl.static_range(0, 2):
            kw0 = (ow + 1) % 2
            kw = kw0 + kw_idx * 2
            iw = (ow + 1 - kw) // 2
            kw_valid = (kw < 3) & (iw >= 0) & (iw < IW)

            valid = kh_valid & kw_valid & sp_mask  # [BLOCK_SP]

            # Loop over IC in blocks to do a tiny matmul-style accumulation
            # For each (kh, kw), w has shape [IC, OC] slice; x has [IC] slice per sp.
            # accumulator += w[:, oc_offs, kh, kw]^T @ x[:, ih, iw]  (but ih,iw differ per sp)
            # So accumulate per-sp: for each ic, acc[oc, sp] += w[ic, oc, kh, kw] * x[n, ic, ih[sp], iw[sp]]
            
            BLOCK_IC: tl.constexpr = 16
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                ic_mask = ic_offs < IC

                # Load x[n, ic_offs, ih[sp], iw[sp]] -> [BLOCK_IC, BLOCK_SP]
                # x ptr: n*IC*IH*IW + ic*IH*IW + ih*IW + iw
                x_off = (pid_n * IC * IH * IW
                         + ic_offs[:, None] * (IH * IW)
                         + ih[None, :] * IW
                         + iw[None, :])
                x_mask = ic_mask[:, None] & valid[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_IC, BLOCK_SP]

                # Load w[ic_offs, oc_offs, kh, kw] -> [BLOCK_IC, BLOCK_OC]
                # w shape: [IC, OC, KH, KW]
                w_off = (ic_offs[:, None] * (OC * 3 * 3)
                         + oc_offs[None, :] * (3 * 3)
                         + kh_idx)  # placeholder; need actual kh,kw which vary per sp
                # Problem: kh and kw vary per sp_offs. So we can't broadcast easily.
                # Workaround: kh varies per sp but is determined by oh parity.
                # We need w to depend on per-sp kh,kw, making weight load [BLOCK_IC, BLOCK_OC, BLOCK_SP].
                # That's too large. Instead, for each sp, the kh is either kh_idx mapped from parity.
                # Actually: kh = (oh+1)%2 + kh_idx*2, so kh depends on oh parity.
                # Within a sp block, different sp may have different oh parity => different kh.
                # 
                # Approach: load w for both possible kh values and select per sp.
                
                # We'll load w[ic, oc, 0, ?] and w[ic, oc, 1, ?] etc.
                # For kh_idx=0: kh in {1, 0} depending on (oh+1)%2 (1 if oh even, 0 if oh odd)
                #   oh even => kh=1, oh odd => kh=0
                # For kh_idx=1: kh in {3 invalid, 2} -> kh=2 when oh odd (then kh_valid=false for even since kh=3)
                # Hmm but we masked kh<3.
                # So:
                # kh_idx=0: kh = 1 if oh%2==0 else 0
                # kh_idx=1: kh = 3 (invalid) if oh%2==0 else 2
                # Similarly for kw.

                # Just compute kh directly per sp and use it in offset:
                w_off = (ic_offs[:, None, None] * (OC * 3 * 3)
                         + oc_offs[None, :, None] * (3 * 3)
                         + kh[None, None, :] * 3
                         + kw[None, None, :])
                # Shape [BLOCK_IC, BLOCK_OC, BLOCK_SP] - large but manageable
                w_mask = ic_mask[:, None, None] & oc_mask[None, :, None] & valid[None, None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC, BLOCK_SP]

                # acc[oc, sp] += sum_ic w[ic, oc, sp] * x[ic, sp]
                acc += tl.sum(w_vals * x_vals[:, None, :], axis=0)

    # Add bias
    b = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + b[:, None]

    # Mish: x * tanh(softplus(x))
    sp_val = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(tl.where(acc > 20.0, 0.0, acc))))
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    y = acc * tanh_sp

    # add value
    y = y + add_value
    # hardtanh
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    # scale
    y = y * scale

    # store
    out_off = (pid_n * OC * OH * OW
               + oc_offs[:, None] * (OH * OW)
               + sp_offs[None, :])
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, y, mask=out_mask)


def conv_transpose_fused(x, w, b, add_value, scale, OH, OW):
    N, IC, IH, IW = x.shape
    _, OC, KH, KW = w.shape
    assert KH == 3 and KW == 3

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

    conv_transpose_fused_kernel[grid](
        x, w, b, out,
        N, IC, IH, IW,
        OC, OH, OW,
        float(add_value), float(scale),
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def fused_epilogue_kernel(
    x_ptr, out_ptr, n_elements,
    add_value: tl.constexpr, scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Stable softplus: sp = max(x, 0) + log1p(exp(-|x|))  approximated as log(1+exp(x)) for x<=20, x for x>20
    x_clamped = tl.where(x > 20.0, 0.0, x)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x_clamped)))
    # tanh(sp) using exp
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    mish = x * tanh_sp
    y = mish + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x, add_value, scale):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, out, n,
        add_value=float(add_value), scale=float(scale),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_epilogue(x, self.add_value, self.scale)