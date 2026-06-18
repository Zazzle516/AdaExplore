import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose2d as a direct gather conv on the output grid.
# For each output pixel (n, oc, oh, ow):
#   y = sum_{ic, kh, kw} input[n, ic, ih, iw] * W_flipped[oc, ic, kh, kw]
# where ih = (oh + pad - kh) / stride, valid only if divisible.
# We tile by (N*OH*OW, OC). With KH=KW=4, stride=2, pad=1, each output
# touches exactly 4 (kh,kw) taps. We unroll those.
# Then we fuse bias subtract + tanh.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 32, 'BLOCK_OC': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
    ],
    key=['N', 'IC', 'OC', 'H', 'W', 'OH', 'OW'],
)
@triton.jit
def conv_transpose2d_k4s2p1_kernel(
    x_ptr,           # [N, H, W, IC] channels-last (NHWC)
    w_ptr,           # [OC, KH, KW, IC] pre-flipped weight in (OC, KH, KW, IC)
    bias_ptr,        # [OC] subtractive bias
    out_ptr,         # [N, OH, OW, OC] NHWC
    N, IC, OC, H, W, OH, OW,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_on, stride_oh, stride_ow, stride_oc,
    BLOCK_SP: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    oh = sp_offs // OW
    ow = sp_offs % OW

    # ConvTranspose: ih_num = oh + pad - kh; ih = ih_num / stride; valid if mod == 0
    # pad=1, stride=2, kh,kw in [0,3]
    # ih_num = oh + 1 - kh
    # For oh even: oh+1 is odd. kh must be odd -> kh in {1,3}
    # For oh odd : oh+1 is even. kh must be even -> kh in {0,2}
    # Similarly for kw.
    # Each output has exactly 4 taps: (kh0, kh1) x (kw0, kw1).

    # Determine the two valid kh values per sp element
    oh_parity = (oh + 1) & 1  # 0 if oh+1 even (need kh even), 1 if oh+1 odd (need kh odd)
    # kh values: if oh_parity==0 -> {0,2}; if oh_parity==1 -> {1,3}
    kh_a = oh_parity            # 0 or 1
    kh_b = oh_parity + 2        # 2 or 3
    ih_a = (oh + 1 - kh_a) // 2
    ih_b = (oh + 1 - kh_b) // 2

    ow_parity = (ow + 1) & 1
    kw_a = ow_parity
    kw_b = ow_parity + 2
    iw_a = (ow + 1 - kw_a) // 2
    iw_b = (ow + 1 - kw_b) // 2

    # Validity masks for input indices
    ih_a_valid = (ih_a >= 0) & (ih_a < H)
    ih_b_valid = (ih_b >= 0) & (ih_b < H)
    iw_a_valid = (iw_a >= 0) & (iw_a < W)
    iw_b_valid = (iw_b >= 0) & (iw_b < W)

    # Four taps: (kh_a,kw_a), (kh_a,kw_b), (kh_b,kw_a), (kh_b,kw_b)
    v_aa = ih_a_valid & iw_a_valid  # [BLOCK_SP]
    v_ab = ih_a_valid & iw_b_valid
    v_ba = ih_b_valid & iw_a_valid
    v_bb = ih_b_valid & iw_b_valid

    # Base input offset for this batch n
    n_base_x = pid_n * stride_xn

    # Accumulator
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Loop over IC in chunks
    for ic_start in range(0, IC, BLOCK_IC):
        ic_offs = ic_start + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offs < IC

        # ---- Load 4 input tiles: shape [BLOCK_SP, BLOCK_IC] ----
        # x[n, ih, iw, ic]
        def load_x(ih, iw, v):
            x_off = n_base_x + ih[:, None] * stride_xh + iw[:, None] * stride_xw + ic_offs[None, :] * stride_xc
            m = v[:, None] & ic_mask[None, :]
            return tl.load(x_ptr + x_off, mask=m, other=0.0)

        # Actually, can't define helper inside; inline manually:
        x_off_aa = n_base_x + ih_a[:, None] * stride_xh + iw_a[:, None] * stride_xw + ic_offs[None, :] * stride_xc
        x_off_ab = n_base_x + ih_a[:, None] * stride_xh + iw_b[:, None] * stride_xw + ic_offs[None, :] * stride_xc
        x_off_ba = n_base_x + ih_b[:, None] * stride_xh + iw_a[:, None] * stride_xw + ic_offs[None, :] * stride_xc
        x_off_bb = n_base_x + ih_b[:, None] * stride_xh + iw_b[:, None] * stride_xw + ic_offs[None, :] * stride_xc

        m_aa = v_aa[:, None] & ic_mask[None, :]
        m_ab = v_ab[:, None] & ic_mask[None, :]
        m_ba = v_ba[:, None] & ic_mask[None, :]
        m_bb = v_bb[:, None] & ic_mask[None, :]

        x_aa = tl.load(x_ptr + x_off_aa, mask=m_aa, other=0.0)
        x_ab = tl.load(x_ptr + x_off_ab, mask=m_ab, other=0.0)
        x_ba = tl.load(x_ptr + x_off_ba, mask=m_ba, other=0.0)
        x_bb = tl.load(x_ptr + x_off_bb, mask=m_bb, other=0.0)

        # ---- Load 4 weight tiles: shape [BLOCK_IC, BLOCK_OC] ----
        # w_ptr layout: [OC, KH, KW, IC] -> offset = oc*KH*KW*IC + kh*KW*IC + kw*IC + ic
        # But per-sp element, kh/kw differ! That's bad. We need broadcast over sp.
        # Instead: since kh_a, kh_b, kw_a, kw_b depend on sp parity (only 2 possibilities each),
        # we can't easily batch. However, kh_a/kw_a are vectors of shape [BLOCK_SP].
        # Workaround: kh_a is either 0 or 1 uniformly within a parity group. Within a BLOCK_SP
        # there can be all 4 parity combinations.
        # Solution: load weight for each of 4 (kh,kw) pairs individually. But kh varies per sp.
        # Alternative approach: load weight twice per kh value (0,1,2,3) is too much.
        #
        # Simpler: iterate over kh in {0,1,2,3}, kw in {0,1,2,3} and mask. But that's 16 taps.
        # Let's just do it: 16 weight loads per IC block, 16 input loads. Too much.
        #
        # Better: realize parity is per-sp. Use the actual computed kh_a[:] as a per-row index.
        # We can compute weight offsets that depend on sp by broadcasting kh_a over IC.
        # w shape [BLOCK_SP, BLOCK_IC, BLOCK_OC]? Too big.
        #
        # Re-approach: use einsum-style: split BLOCK_SP by parity. Too complex.
        #
        # Let's restructure: load weight as [4_taps, BLOCK_IC, BLOCK_OC] for kh in {0,1,2,3}? No.
        # Each output sp picks 2 kh out of 4 based on parity. There are 2 parity classes for h,
        # and 2 for w, so 4 combinations of (parity_h, parity_w). Within each combination, kh_a,
        # kh_b, kw_a, kw_b are constants. So we'd load weight 4 (= 4 taps) times per combination,
        # and apply only to the matching sp rows via mask.
        #
        # Just do all 16 (kh,kw) pairs, mask by validity. With BLOCK_IC=32, BLOCK_OC=64,
        # weight load is 32*64*4=8KB per tap, 16 taps = 128KB per IC iter — too much.
        #
        # Better plan: do 4 separate kernel launches by (oh_parity, ow_parity)? Output is contiguous
        # in NHWC though; rows would be strided. Skip.
        #
        # Use per-row weight indexing: compute kh_a (shape [BLOCK_SP]) and gather weight as
        # [BLOCK_SP, BLOCK_OC] for fixed ic (broadcasted). Then we need a matmul-like sum over IC.
        # That's a gather GEMM which Triton can do but is expensive.
        #
        # Final decision: launch the kernel with grid also over (oh_parity, ow_parity) — 4 launches
        # cover the output. Within each launch, kh_a,kh_b,kw_a,kw_b are compile-time constants and
        # we can do a clean tl.dot.
        pass

    # placeholder; real work done in the parity-specialized kernel below.
    _ = acc
    return


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 32, 'BLOCK_OC': 128, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
    ],
    key=['N', 'IC', 'OC', 'H', 'W'],
)
@triton.jit
def conv_transpose2d_parity_kernel(
    x_ptr,           # [N, H, W, IC] NHWC
    w_ptr,           # [KH, KW, IC, OC]  (pre-flipped, reshaped)
    bias_ptr,        # [OC]
    out_ptr,         # [N, OH, OW, OC] NHWC
    N, IC, OC, H, W, OH, OW,
    OH2, OW2,        # OH/2, OW/2 (number of output rows/cols per parity class)
    PH: tl.constexpr,  # output-row parity (0 or 1)
    PW: tl.constexpr,  # output-col parity (0 or 1)
    KH_A: tl.constexpr, KH_B: tl.constexpr,
    KW_A: tl.constexpr, KW_B: tl.constexpr,
    stride_xn, stride_xh, stride_xw,
    stride_on, stride_oh, stride_ow,
    stride_wkh, stride_wkw, stride_wic, stride_woc,
    BLOCK_SP: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    SP_TOTAL = OH2 * OW2  # number of output pixels in this parity class

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_mask = sp_offs < SP_TOTAL
    oc_mask = oc_offs < OC

    # Decode sp into (oh_idx, ow_idx) within parity class, then real (oh, ow)
    oh_idx = sp_offs // OW2
    ow_idx = sp_offs % OW2
    oh = oh_idx * 2 + PH
    ow = ow_idx * 2 + PW

    # Input coords for the two kh/kw choices
    ih_a = (oh + 1 - KH_A) // 2
    ih_b = (oh + 1 - KH_B) // 2
    iw_a = (ow + 1 - KW_A) // 2
    iw_b = (ow + 1 - KW_B) // 2

    ih_a_v = (ih_a >= 0) & (ih_a < H)
    ih_b_v = (ih_b >= 0) & (ih_b < H)
    iw_a_v = (iw_a >= 0) & (iw_a < W)
    iw_b_v = (iw_b >= 0) & (iw_b < W)

    v_aa = ih_a_v & iw_a_v
    v_ab = ih_a_v & iw_b_v
    v_ba = ih_b_v & iw_a_v
    v_bb = ih_b_v & iw_b_v

    n_base_x = pid_n * stride_xn

    # Precompute spatial portion of input offsets (ic dim added inside loop)
    # Input is contiguous in C, so stride_xc = 1; we use plain ic_offs.
    base_aa = n_base_x + ih_a * stride_xh + iw_a * stride_xw  # [BLOCK_SP]
    base_ab = n_base_x + ih_a * stride_xh + iw_b * stride_xw
    base_ba = n_base_x + ih_b * stride_xh + iw_a * stride_xw
    base_bb = n_base_x + ih_b * stride_xh + iw_b * stride_xw

    # Weight base offsets per tap (kh, kw fixed)
    w_base_aa = KH_A * stride_wkh + KW_A * stride_wkw
    w_base_ab = KH_A * stride_wkh + KW_B * stride_wkw
    w_base_ba = KH_B * stride_wkh + KW_A * stride_wkw
    w_base_bb = KH_B * stride_wkh + KW_B * stride_wkw

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    for ic_start in range(0, IC, BLOCK_IC):
        ic_offs = ic_start + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offs < IC

        # x tiles: [BLOCK_SP, BLOCK_IC]
        x_aa_off = base_aa[:, None] + ic_offs[None, :]
        x_ab_off = base_ab[:, None] + ic_offs[None, :]
        x_ba_off = base_ba[:, None] + ic_offs[None, :]
        x_bb_off = base_bb[:, None] + ic_offs[None, :]

        m_aa = v_aa[:, None] & ic_mask[None, :]
        m_ab = v_ab[:, None] & ic_mask[None, :]
        m_ba = v_ba[:, None] & ic_mask[None, :]
        m_bb = v_bb[:, None] & ic_mask[None, :]

        x_aa = tl.load(x_ptr + x_aa_off, mask=m_aa, other=0.0)
        x_ab = tl.load(x_ptr + x_ab_off, mask=m_ab, other=0.0)
        x_ba = tl.load(x_ptr + x_ba_off, mask=m_ba, other=0.0)
        x_bb = tl.load(x_ptr + x_bb_off, mask=m_bb, other=0.0)

        # w tiles: [BLOCK_IC, BLOCK_OC]
        w_aa_off = w_base_aa + ic_offs[:, None] * stride_wic + oc_offs[None, :] * stride_woc
        w_ab_off = w_base_ab + ic_offs[:, None] * stride_wic + oc_offs[None, :] * stride_woc
        w_ba_off = w_base_ba + ic_offs[:, None] * stride_wic + oc_offs[None, :] * stride_woc
        w_bb_off = w_base_bb + ic_offs[:, None] * stride_wic + oc_offs[None, :] * stride_woc

        wm = ic_mask[:, None] & oc_mask[None, :]
        w_aa = tl.load(w_ptr + w_aa_off, mask=wm, other=0.0)
        w_ab = tl.load(w_ptr + w_ab_off, mask=wm, other=0.0)
        w_ba = tl.load(w_ptr + w_ba_off, mask=wm, other=0.0)
        w_bb = tl.load(w_ptr + w_bb_off, mask=wm, other=0.0)

        acc += tl.dot(x_aa, w_aa)
        acc += tl.dot(x_ab, w_ab)
        acc += tl.dot(x_ba, w_ba)
        acc += tl.dot(x_bb, w_bb)

    # Bias subtract + tanh fused
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc - b[None, :]
    acc = tl.extra.cuda.libdevice.tanh(acc)

    # Store to output NHWC: out[n, oh, ow, oc]
    out_off = pid_n * stride_on + oh[:, None] * stride_oh + ow[:, None] * stride_ow + oc_offs[None, :]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose_fused(x_nhwc, w_flipped, conv_bias, sub_bias, N, IC, OC, H, W, KH, KW, OH, OW):
    # x_nhwc: [N, H, W, IC] contiguous
    # w_flipped: [KH, KW, IC, OC] contiguous  -- already includes conv_bias added? No.
    # We need to add conv_bias as well (the conv's own bias) and subtract sub_bias.
    # Combine: effective additive bias = conv_bias - sub_bias.
    eff_bias = sub_bias - conv_bias  # we'll do acc - eff_bias = acc + conv_bias - sub_bias

    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    stride_xn, stride_xh, stride_xw, stride_xc = x_nhwc.stride()
    stride_on, stride_oh, stride_ow, stride_oc = out.stride()
    stride_wkh, stride_wkw, stride_wic, stride_woc = w_flipped.stride()

    OH2 = OH // 2
    OW2 = OW // 2

    # For each parity, compute the two valid kh values:
    # oh_parity_of_oh+1 even -> kh in {0,2}; odd -> {1,3}
    # PH is parity of oh itself. oh = oh_idx*2 + PH, so (oh+1) parity = (PH+1)%2
    # If PH=0: oh+1 odd -> kh in {1,3}; if PH=1: oh+1 even -> kh in {0,2}
    def kh_pair(PH):
        if PH == 0:
            return (1, 3)
        else:
            return (0, 2)

    for PH in (0, 1):
        for PW in (0, 1):
            KH_A, KH_B = kh_pair(PH)
            KW_A, KW_B = kh_pair(PW)
            SP_TOTAL = OH2 * OW2
            grid = lambda meta: (
                triton.cdiv(SP_TOTAL, meta['BLOCK_SP']),
                triton.cdiv(OC, meta['BLOCK_OC']),
                N,
            )
            conv_transpose2d_parity_kernel[grid](
                x_nhwc, w_flipped, eff_bias, out,
                N, IC, OC, H, W, OH, OW,
                OH2, OW2,
                PH, PW,
                KH_A, KH_B, KW_A, KW_B,
                stride_xn, stride_xh, stride_xw,
                stride_on, stride_oh, stride_ow,
                stride_wkh, stride_wkw, stride_wic, stride_woc,
            )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        assert kernel_size == 4 and stride == 2 and padding == 1 and output_padding == 1, \
            "This optimized kernel is specialized for k=4, s=2, p=1, op=1."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Build a regular ConvTranspose2d to inherit standard parameter init,
        # then store its weight/bias as our parameters.
        ct = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, output_padding=output_padding)
        # weight shape: [IC, OC, KH, KW]
        self.weight = nn.Parameter(ct.weight.detach().clone())
        self.conv_bias = nn.Parameter(ct.bias.detach().clone())  # [OC]
        self.bias = nn.Parameter(torch.randn(bias_shape))         # subtractive bias

        self._cached_w = None
        self._cached_w_version = -1

    def _prepare_weight(self):
        # ConvTranspose2d weight layout: [IC, OC, KH, KW]
        # Direct conv formulation needs W_flipped[OC, IC, KH, KW] with kh,kw flipped.
        # Then we reshape to [KH, KW, IC, OC] (NHWC-friendly).
        v = self.weight._version
        if self._cached_w is not None and self._cached_w_version == v:
            return self._cached_w
        w = self.weight  # [IC, OC, KH, KW]
        w = w.flip([-1, -2])  # flip spatial
        w = w.permute(2, 3, 0, 1).contiguous()  # [KH, KW, IC, OC]
        self._cached_w = w
        self._cached_w_version = v
        return w

    def forward(self, x):
        # x: [N, IC, H, W]
        x = x.cuda() if not x.is_cuda else x
        N, IC, H, W = x.shape
        KH = KW = self.kernel_size
        OH = (H - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (W - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, IC]

        w_prepped = self._prepare_weight()  # [KH, KW, IC, OC]

        sub_bias = self.bias.view(-1).contiguous()  # [OC]
        conv_bias = self.conv_bias.contiguous()     # [OC]

        out_nhwc = conv_transpose_fused(
            x_nhwc, w_prepped, conv_bias, sub_bias,
            N, IC, self.out_channels, H, W, KH, KW, OH, OW
        )

        # Back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out