import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr,          # [N, ID, IH, IW, IC]
    w_ptr,          # [IC, KD, KH, KW, OC]
    b_ptr,          # [OC]
    out_ptr,        # [N, OD, OH, OW, OC]
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    MIN_VAL, INV_DIV,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_n = tl.program_id(0)        # batch
    pid_m = tl.program_id(1)        # OC tile
    pid_s = tl.program_id(2)        # spatial tile

    DHW = OD * OH * OW
    spatial_offs = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)
    spatial_mask = spatial_offs < DHW

    # decode od, oh, ow
    od = spatial_offs // (OH * OW)
    rem = spatial_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    # for each kernel position, find the input voxel that contributes
    # od + PAD - kd must be divisible by STRIDE, and the resulting id must be in [0, ID)
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_div = id_num // STRIDE
        id_ok = (id_num >= 0) & ((id_num - id_div * STRIDE) == 0) & (id_div >= 0) & (id_div < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_div = ih_num // STRIDE
            ih_ok = (ih_num >= 0) & ((ih_num - ih_div * STRIDE) == 0) & (ih_div >= 0) & (ih_div < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_div = iw_num // STRIDE
                iw_ok = (iw_num >= 0) & ((iw_num - iw_div * STRIDE) == 0) & (iw_div >= 0) & (iw_div < IW)

                valid = id_ok & ih_ok & iw_ok & spatial_mask  # [BLOCK_N]

                # compute base input offset for each spatial pos (NDHWC layout)
                in_base = ((pid_n * ID + id_div) * IH + ih_div) * IW + iw_div  # [BLOCK_N]
                in_base = in_base * IC

                # compute base weight offset: weight is [IC, KD, KH, KW, OC]
                # per (kd,kh,kw) and oc: w[ic, kd, kh, kw, oc]
                # base for ic=0: ((kd*KH+kh)*KW+kw)*OC + oc
                w_base_kk = ((kd * KH + kh) * KW + kw) * OC

                # K loop over IC
                for ic_start in range(0, IC, BLOCK_K):
                    ic_offs = ic_start + tl.arange(0, BLOCK_K)
                    ic_mask = ic_offs < IC

                    # load x: [BLOCK_N, BLOCK_K]
                    x_ptrs = x_ptr + in_base[:, None] + ic_offs[None, :]
                    x_load_mask = valid[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                    # load w: [BLOCK_K, BLOCK_M]
                    # w[ic, kd, kh, kw, oc] = w_ptr + ic * (KD*KH*KW*OC) + w_base_kk + oc
                    w_ptrs = w_ptr + ic_offs[:, None] * (KD * KH * KW * OC) + w_base_kk + oc_offs[None, :]
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_vals[None, :]

    # clamp + divide
    acc = tl.where(acc < MIN_VAL, MIN_VAL, acc)
    acc = acc * INV_DIV

    # store NDHWC
    out_ptrs = out_ptr + ((pid_n * DHW) + spatial_offs)[:, None] * OC + oc_offs[None, :]
    store_mask = spatial_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


def conv_transpose3d_fused(x, weight, bias, stride, padding, min_value, divisor):
    """
    x: [N, IC, ID, IH, IW]
    weight: [IC, OC, KD, KH, KW]
    bias: [OC]
    Returns: [N, OC, OD, OH, OW]
    """
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    assert KD == KH == KW

    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    # Convert to NDHWC layout
    x_ndhwc = x.permute(0, 2, 3, 4, 1).contiguous()
    # weight: [IC, OC, KD, KH, KW] -> [IC, KD, KH, KW, OC]
    w_layout = weight.permute(0, 2, 3, 4, 1).contiguous()

    out_ndhwc = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (N, triton.cdiv(OC, BLOCK_M), triton.cdiv(OD * OH * OW, BLOCK_N))

    conv_transpose3d_gather_kernel[grid](
        x_ndhwc, w_layout, bias, out_ndhwc,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        stride, padding,
        float(min_value), 1.0 / float(divisor),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )

    # back to NCDHW
    out = out_ndhwc.permute(0, 4, 1, 2, 3).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous().cuda()
        return conv_transpose3d_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride, self.padding,
            self.min_value, self.divisor,
        )