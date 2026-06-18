import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _get_autotune_configs():
    configs = []
    for bp in [32, 64, 128, 256]:
        for nw in [2, 4, 8]:
            for ns in [2, 3]:
                configs.append(triton.Config({'BLOCK_P': bp}, num_warps=nw, num_stages=ns))
    return configs


@triton.autotune(configs=_get_autotune_configs(), key=['PD', 'PH', 'PW', 'IC', 'OC'])
@triton.jit
def convt3d_fused_kernel(
    x_ptr,         # (N, IC, ID, IH, IW)
    w_ptr,         # (IC, OC, KD, KH, KW)
    b_ptr,         # (OC,)
    out_ptr,       # (N, OC)
    N,
    IC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    MK: tl.constexpr,
    scale: tl.constexpr,
    inv_count: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    bias = tl.load(b_ptr + oc)

    total_pooled: tl.constexpr = PD * PH * PW
    acc_sum = 0.0

    for off in range(0, total_pooled, BLOCK_P):
        idx = off + tl.arange(0, BLOCK_P)
        mask = idx < total_pooled

        pd = idx // (PH * PW)
        rem = idx % (PH * PW)
        ph = rem // PW
        pw = rem % PW

        d0 = pd * MK
        h0 = ph * MK
        w0 = pw * MK

        max_v = tl.full((BLOCK_P,), -float('inf'), dtype=tl.float32)

        for mkd in tl.static_range(0, MK):
            for mkh in tl.static_range(0, MK):
                for mkw in tl.static_range(0, MK):
                    od = d0 + mkd
                    oh = h0 + mkh
                    ow = w0 + mkw

                    val = tl.zeros((BLOCK_P,), dtype=tl.float32) + bias

                    for kd in tl.static_range(0, KD):
                        id_num = od + PAD_D - kd
                        id_q = id_num // SD
                        id_r = id_num - id_q * SD
                        d_ok = (id_r == 0) & (id_q >= 0) & (id_q < ID)

                        for kh in tl.static_range(0, KH):
                            ih_num = oh + PAD_H - kh
                            ih_q = ih_num // SH
                            ih_r = ih_num - ih_q * SH
                            h_ok = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)

                            for kw in tl.static_range(0, KW):
                                iw_num = ow + PAD_W - kw
                                iw_q = iw_num // SW
                                iw_r = iw_num - iw_q * SW
                                w_ok = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)

                                cell_ok = d_ok & h_ok & w_ok

                                for ic in tl.static_range(0, IC):
                                    x_off = ((n * IC + ic) * ID + id_q) * IH * IW + ih_q * IW + iw_q
                                    xv = tl.load(x_ptr + x_off, mask=cell_ok, other=0.0)
                                    w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                                    wv = tl.load(w_ptr + w_off)
                                    val += xv * wv

                    val = val * scale
                    max_v = tl.maximum(max_v, val)

        max_v = tl.where(mask, max_v, 0.0)
        acc_sum += tl.sum(max_v, axis=0)

    mean = acc_sum * inv_count
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * OC + oc, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        N = x.shape[0]
        IC = self.in_channels
        OC = self.out_channels
        ID, IH, IW = x.shape[2], x.shape[3], x.shape[4]
        K = self.kernel_size
        S = self.stride
        P = self.padding
        MK = self.maxpool_kernel_size

        OD = (ID - 1) * S - 2 * P + K
        OH = (IH - 1) * S - 2 * P + K
        OW = (IW - 1) * S - 2 * P + K

        PD = OD // MK
        PH = OH // MK
        PW = OW // MK

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else \
               torch.zeros(OC, device=x.device, dtype=x.dtype)

        out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)

        total_pooled = PD * PH * PW
        inv_count = 1.0 / float(total_pooled)

        grid = (N * OC,)
        convt3d_fused_kernel[grid](
            x, weight, bias, out,
            N, IC,
            ID, IH, IW,
            OC,
            OD, OH, OW,
            PD, PH, PW,
            K, K, K,
            S, S, S,
            P, P, P,
            MK,
            float(self.scale),
            inv_count,
        )
        return out