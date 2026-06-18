import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OD_HALF', 'OH_HALF', 'OW_HALF'],
)
@triton.jit
def conv_transpose3d_parity_kernel(
    x_ptr, w_packed_ptr, conv_bias_ptr, add_input_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    OD_HALF, OH_HALF, OW_HALF,
    PD, PH, PW,  # parity: which (od%2, oh%2, ow%2) - constexpr
    KEFF: tl.constexpr,  # number of valid kernel taps for this parity = 2*2*2 = 8
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Parity-split implicit GEMM:
    # For stride=2, kernel=3, padding=1: each output position (od,oh,ow) takes
    # contributions from kernel taps where (od + 1 - kd) % 2 == 0, i.e. kd has
    # same parity as (od + 1). Since od_parity is fixed in this grid, kd is fixed
    # to specific values: if od%2==0, kd in {1}, no wait...
    # Actually KD=3, so kd in {0,1,2}. (od+1-kd) % 2 == 0 means kd parity = (od+1) parity.
    # od%2=0: kd parity = 1, so kd in {1}  -> 1 tap
    # od%2=1: kd parity = 0, so kd in {0,2} -> 2 taps
    # So KEFF varies per parity: 1*1*1=1, 1*1*2=2, 1*2*1=2, 2*1*1=2, 2*2*1=4, 2*1*2=4, 1*2*2=4, 2*2*2=8
    #
    # We pre-pack weights for each parity into a contiguous tensor of shape
    # [parity_idx, IC, OC, KEFF_max] to allow a clean GEMM. Actually we pack per-parity
    # weight as [IC * KEFF, OC] and dispatch one kernel per parity.

    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    SP = OD_HALF * OH_HALF * OW_HALF
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < SP

    # Decode local (odh, ohh, owh) within the half-grid
    owh = sp_offs % OW_HALF
    ohh = (sp_offs // OW_HALF) % OH_HALF
    odh = sp_offs // (OW_HALF * OH_HALF)

    # Actual output coords
    od = odh * 2 + PD
    oh = ohh * 2 + PH
    ow = owh * 2 + PW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Iterate over valid kernel taps for this parity.
    # For each parity, the valid kd values: kd s.t. (od+1-kd) is even & nonneg.
    # For od=2*odh+PD: od+1-kd = 2*odh+PD+1-kd. For this to be even, kd parity = PD+1 parity.
    # KD=3, so:
    #   PD=0: kd in {1}        (1 tap)
    #   PD=1: kd in {0, 2}     (2 taps)
    # Same for kh, kw.

    KD_LIST_LEN: tl.constexpr = 2 if PD == 1 else 1
    KH_LIST_LEN: tl.constexpr = 2 if PH == 1 else 1
    KW_LIST_LEN: tl.constexpr = 2 if PW == 1 else 1

    IH_IW = IH * IW
    ID_IH_IW = ID * IH_IW
    x_batch_base = pid_n * (IC * ID_IH_IW)

    # Pre-packed weight layout: [tap_idx, IC, OC] where tap_idx ranges over KEFF
    # tap_idx = ti_d * (KH_LIST_LEN * KW_LIST_LEN) + ti_h * KW_LIST_LEN + ti_w
    # Total: KEFF taps. Stride between consecutive tap is IC*OC.
    IC_OC = IC * OC
    ic_arange = tl.arange(0, BLOCK_IC)

    for ti_d in tl.static_range(0, KD_LIST_LEN):
        # kd value
        if PD == 0:
            kd = 1
        else:
            kd = ti_d * 2  # 0 or 2
        id_val = (od + 1 - kd) // 2  # since both numerator divisible by 2
        # but od is per-element, so this is a vector
        id_valid = (id_val >= 0) & (id_val < ID)

        for ti_h in tl.static_range(0, KH_LIST_LEN):
            if PH == 0:
                kh = 1
            else:
                kh = ti_h * 2
            ih_val = (oh + 1 - kh) // 2
            ih_valid = (ih_val >= 0) & (ih_val < IH)

            for ti_w in tl.static_range(0, KW_LIST_LEN):
                if PW == 0:
                    kw = 1
                else:
                    kw = ti_w * 2
                iw_val = (ow + 1 - kw) // 2
                iw_valid = (iw_val >= 0) & (iw_val < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask
                in_spatial_idx = id_val * IH_IW + ih_val * IW + iw_val
                in_spatial_idx_safe = tl.where(spatial_valid, in_spatial_idx, 0)

                tap_idx = ti_d * (KH_LIST_LEN * KW_LIST_LEN) + ti_h * KW_LIST_LEN + ti_w
                w_tap_base = tap_idx * IC_OC

                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + ic_arange
                    ic_mask = ic_offs < IC

                    # Load x: [BLOCK_IC, BLOCK_SP]
                    x_offs = x_batch_base + ic_offs[:, None] * ID_IH_IW + in_spatial_idx_safe[None, :]
                    x_load_mask = ic_mask[:, None] & spatial_valid[None, :]
                    x_vals = tl.load(x_ptr + x_offs, mask=x_load_mask, other=0.0)

                    # Load w: [BLOCK_OC, BLOCK_IC] from packed weight [tap, ic, oc]
                    w_offs = w_tap_base + ic_offs[None, :] * OC + oc_offs[:, None]
                    w_load_mask = oc_mask[:, None] & ic_mask[None, :]
                    w_vals = tl.load(w_packed_ptr + w_offs, mask=w_load_mask, other=0.0)

                    acc += tl.dot(w_vals, x_vals, allow_tf32=True)

    # Add conv bias
    cbias = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cbias[:, None]

    # Compute output offset
    DHW = OD * OH * OW
    out_spatial = od[None, :] * (OH * OW) + oh[None, :] * OW + ow[None, :]
    out_offs = pid_n * (OC * DHW) + oc_offs[:, None] * DHW + out_spatial
    out_mask = oc_mask[:, None] & sp_mask[None, :]

    add_vals = tl.load(add_input_ptr + out_offs, mask=out_mask, other=0.0)
    acc += add_vals

    # HardSwish: x * (x * relu6(x+3)/6)
    hs_inner = acc + 3.0
    hs_inner = tl.maximum(hs_inner, 0.0)
    hs_inner = tl.minimum(hs_inner, 6.0)
    hardswish_val = acc * hs_inner * (1.0 / 6.0)
    result = acc * hardswish_val

    tl.store(out_ptr + out_offs, result, mask=out_mask)


def _pack_weight_for_parity(weight, kd_list, kh_list, kw_list):
    # weight: (IC, OC, KD, KH, KW)
    # Output: (KEFF, IC, OC) contiguous
    IC, OC, KD, KH, KW = weight.shape
    taps = []
    for kd in kd_list:
        for kh in kh_list:
            for kw in kw_list:
                taps.append(weight[:, :, kd, kh, kw])  # (IC, OC)
    packed = torch.stack(taps, dim=0).contiguous()  # (KEFF, IC, OC)
    return packed


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

        self._packed_cache = {}

    def _get_packed(self, parity_idx, weight):
        if parity_idx in self._packed_cache and self._packed_cache[parity_idx][0] is weight:
            return self._packed_cache[parity_idx][1]

        PD = (parity_idx >> 2) & 1
        PH = (parity_idx >> 1) & 1
        PW = parity_idx & 1

        kd_list = [1] if PD == 0 else [0, 2]
        kh_list = [1] if PH == 0 else [0, 2]
        kw_list = [1] if PW == 0 else [0, 2]

        packed = _pack_weight_for_parity(weight, kd_list, kh_list, kw_list)
        self._packed_cache[parity_idx] = (weight, packed)
        return packed

    def forward(self, x, add_input):
        x = x.contiguous()
        add_input = add_input.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias.contiguous()

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding

        OD = (ID - 1) * S - 2 * P + KD + OP
        OH = (IH - 1) * S - 2 * P + KH + OP
        OW = (IW - 1) * S - 2 * P + KW + OP

        # Stride must be 2, kernel 3, padding 1 for this parity-split kernel.
        assert S == 2 and KD == 3 and P == 1, "ModelNew parity kernel requires stride=2, kernel=3, padding=1"

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        # Launch 8 kernels, one per parity
        for parity_idx in range(8):
            PD = (parity_idx >> 2) & 1
            PH = (parity_idx >> 1) & 1
            PW = parity_idx & 1

            # Number of output positions for this parity
            OD_HALF = (OD - PD + 1) // 2  # PD=0: ceil(OD/2), PD=1: floor(OD/2)
            OH_HALF = (OH - PH + 1) // 2
            OW_HALF = (OW - PW + 1) // 2

            if OD_HALF <= 0 or OH_HALF <= 0 or OW_HALF <= 0:
                continue

            packed_w = self._get_packed(parity_idx, weight)

            keff = (2 if PD == 1 else 1) * (2 if PH == 1 else 1) * (2 if PW == 1 else 1)

            # Pick BLOCK_IC: must be power of 2 >= some min for tl.dot
            BLOCK_IC = 32 if IC >= 32 else 16

            grid = lambda meta: (
                N,
                triton.cdiv(OC, meta['BLOCK_OC']),
                triton.cdiv(OD_HALF * OH_HALF * OW_HALF, meta['BLOCK_SP']),
            )

            conv_transpose3d_parity_kernel[grid](
                x, packed_w, conv_bias, add_input, out,
                N, IC, OC,
                ID, IH, IW,
                OD, OH, OW,
                OD_HALF, OH_HALF, OW_HALF,
                PD, PH, PW,
                keff,
                BLOCK_IC=BLOCK_IC,
            )

        return out