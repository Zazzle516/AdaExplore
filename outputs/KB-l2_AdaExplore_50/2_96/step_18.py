import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convt_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr, OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    POD: tl.constexpr, POH: tl.constexpr, POW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr, KS: tl.constexpr,
    MP: tl.constexpr,
    SCALE: tl.constexpr,
    INV_POOL: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    bias = tl.load(b_ptr + oc)

    NEG_INF = -1.0e30
    sum_acc = 0.0

    # Iterate over pooled output cells
    for pd in tl.static_range(0, POD):
        for ph in tl.static_range(0, POH):
            for pw in tl.static_range(0, POW):
                max_val = NEG_INF
                for md in tl.static_range(0, MP):
                    od = pd * MP + md
                    for mh in tl.static_range(0, MP):
                        oh = ph * MP + mh
                        for mw in tl.static_range(0, MP):
                            ow = pw * MP + mw
                            val = 0.0
                            for kd in tl.static_range(0, KS):
                                id_num = od + PAD - kd
                                id_q = id_num // STRIDE
                                id_r = id_num - id_q * STRIDE
                                d_valid = (id_r == 0) & (id_q >= 0) & (id_q < ID)
                                for kh in tl.static_range(0, KS):
                                    ih_num = oh + PAD - kh
                                    ih_q = ih_num // STRIDE
                                    ih_r = ih_num - ih_q * STRIDE
                                    h_valid = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
                                    for kw in tl.static_range(0, KS):
                                        iw_num = ow + PAD - kw
                                        iw_q = iw_num // STRIDE
                                        iw_r = iw_num - iw_q * STRIDE
                                        w_valid = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                                        valid = d_valid & h_valid & w_valid
                                        ic_off = tl.arange(0, IC)
                                        x_off = (((n * IC) + ic_off) * ID + id_q) * (IH * IW) + ih_q * IW + iw_q
                                        w_off = (((ic_off * OC) + oc) * KS + kd) * (KS * KS) + kh * KS + kw
                                        xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                        wv = tl.load(w_ptr + w_off, mask=valid, other=0.0)
                                        val += tl.sum(xv * wv, axis=0)
                            val = (val + bias) * SCALE
                            max_val = tl.maximum(max_val, val)
                sum_acc += max_val

    mean = sum_acc * INV_POOL
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * OC + oc, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = float(scale)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KS = self.kernel_size
        S = self.stride
        P = self.padding
        MP = self.maxpool_kernel_size

        # ConvTranspose3d output shape (no output_padding, dilation=1)
        OD = (ID - 1) * S - 2 * P + KS
        OH = (IH - 1) * S - 2 * P + KS
        OW = (IW - 1) * S - 2 * P + KS

        POD = OD // MP
        POH = OH // MP
        POW = OW // MP

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KS, KS, KS)
        bias = self.conv_transpose.bias.contiguous()

        out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)

        pool_count = POD * POH * POW
        inv_pool = 1.0 / pool_count

        grid = (N * OC,)
        fused_convt_pool_kernel[grid](
            x, weight, bias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            POD, POH, POW,
            S, P, KS,
            MP,
            self.scale,
            inv_pool,
            num_warps=4,
        )
        return out