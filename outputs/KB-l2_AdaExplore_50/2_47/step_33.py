import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 128}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_nhwc_mish_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_OW: tl.constexpr,
):
    pid_ow = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_ndh = tl.program_id(2)

    oh = pid_ndh % OH
    tmp = pid_ndh // OH
    od = tmp % OD
    n = tmp // OD

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    mask_oc = offs_oc < OC
    mask_ow = offs_ow < OW

    offs_ic = tl.arange(0, IC)

    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

    x_n_base = n * (ID * IH * IW * IC)
    w_oc_base = offs_oc[:, None] * (KD * KH * KW * IC)

    for kd in tl.static_range(0, KD):
        id_ = od + kd
        x_d_base = x_n_base + id_ * (IH * IW * IC)
        for kh in tl.static_range(0, KH):
            ih = oh + kh
            x_h_base = x_d_base + ih * (IW * IC)
            for kw in tl.static_range(0, KW):
                iw = offs_ow + kw
                x_offs = x_h_base + iw[:, None] * IC + offs_ic[None, :]
                x_mask = mask_ow[:, None]
                x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                w_offs = (w_oc_base
                          + kd * (KH * KW * IC)
                          + kh * (KW * IC)
                          + kw * IC
                          + offs_ic[None, :])
                w_mask = mask_oc[:, None]
                w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                acc += tl.dot(w, tl.trans(x))

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[:, None]

    # mish + tanh fused
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    t1 = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    m = acc * t1
    t2 = 2.0 * tl.sigmoid(2.0 * m) - 1.0

    out_base = (n * (OC * OD * OH * OW)
                + od * (OH * OW)
                + oh * OW)
    out_offs = out_base + offs_oc[:, None] * (OD * OH * OW) + offs_ow[None, :]
    out_mask = mask_oc[:, None] & mask_ow[None, :]
    tl.store(out_ptr + out_offs, t2, mask=out_mask)


def conv3d_mish_tanh_nhwc(x_nhwc, weight_nhwc, bias, N, IC, ID, IH, IW, OC, KD, KH, KW):
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = lambda meta: (
        triton.cdiv(OW, meta['BLOCK_OW']),
        triton.cdiv(OC, meta['BLOCK_OC']),
        N * OD * OH,
    )

    conv3d_nhwc_mish_tanh_kernel[grid](
        x_nhwc, weight_nhwc, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == 1 and padding == 0, "Custom kernel supports stride=1, padding=0"
        if isinstance(kernel_size, int):
            KD = KH = KW = kernel_size
        else:
            KD, KH, KW = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KD = KD
        self.KH = KH
        self.KW = KW

        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        w = conv.weight.data.detach().clone()
        w_nhwc = w.permute(0, 2, 3, 4, 1).contiguous()
        self.weight_nhwc = nn.Parameter(w_nhwc, requires_grad=False)
        self.bias = nn.Parameter(conv.bias.data.detach().clone(), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()

        out = conv3d_mish_tanh_nhwc(
            x_nhwc, self.weight_nhwc, self.bias,
            N, IC, ID, IH, IW, self.out_channels, self.KD, self.KH, self.KW,
        )
        return out