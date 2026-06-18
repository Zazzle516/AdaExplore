import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
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
    # Program decomposition:
    # pid0: oc tile
    # pid1: ow tile
    # pid2: n * OD * OH + od * OH + oh
    pid_oc = tl.program_id(0)
    pid_ow = tl.program_id(1)
    pid_ndh = tl.program_id(2)

    oh = pid_ndh % OH
    tmp = pid_ndh // OH
    od = tmp % OD
    n  = tmp // OD

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    mask_oc = offs_oc < OC
    mask_ow = offs_ow < OW

    offs_ic = tl.arange(0, IC)  # contiguous channel dim

    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

    # Input layout NDHWC: stride = (ID*IH*IW*IC, IH*IW*IC, IW*IC, IC, 1)
    # Weight layout [OC, KD, KH, KW, IC]: stride = (KD*KH*KW*IC, KH*KW*IC, KW*IC, IC, 1)
    in_d_base = od  # stride = 1 in d direction (no stride/padding)
    in_h_base = oh
    # For each (kd, kh, kw), iw = ow + kw
    # We do GEMM over IC: weight[oc, kd, kh, kw, :] dot x[n, id, ih, iw, :]

    x_n_base = n * (ID * IH * IW * IC)

    for kd in tl.static_range(0, KD):
        id_ = in_d_base + kd
        x_d_base = x_n_base + id_ * (IH * IW * IC)
        for kh in tl.static_range(0, KH):
            ih = in_h_base + kh
            x_h_base = x_d_base + ih * (IW * IC)
            for kw in tl.static_range(0, KW):
                iw = offs_ow + kw  # [BLOCK_OW]
                # x: [BLOCK_OW, IC]
                x_offs = x_h_base + iw[:, None] * IC + offs_ic[None, :]
                x_mask = mask_ow[:, None]
                x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                # w: [BLOCK_OC, IC]
                w_offs = (offs_oc[:, None] * (KD * KH * KW * IC)
                          + kd * (KH * KW * IC)
                          + kh * (KW * IC)
                          + kw * IC
                          + offs_ic[None, :])
                w_mask = mask_oc[:, None]
                w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                # acc[BLOCK_OC, BLOCK_OW] += w[BLOCK_OC, IC] @ x[BLOCK_OW, IC].T
                acc += tl.dot(w, tl.trans(x))

    # Bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[:, None]

    # Fused mish + tanh
    # softplus stable: where(x>20, x, log1p(exp(x)))
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    t1 = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    m = acc * t1
    t2 = 2.0 * tl.sigmoid(2.0 * m) - 1.0

    # Output layout NDHWC: out[n, od, oh, ow, oc]
    # stride = (OD*OH*OW*OC, OH*OW*OC, OW*OC, OC, 1)
    out_base = (n * (OD * OH * OW * OC)
                + od * (OH * OW * OC)
                + oh * (OW * OC))
    out_offs = out_base + offs_ow[None, :] * OC + offs_oc[:, None]
    out_mask = mask_oc[:, None] & mask_ow[None, :]
    tl.store(out_ptr + out_offs, t2, mask=out_mask)


def conv3d_mish_tanh_nhwc(x_nhwc, weight_nhwc, bias, N, IC, ID, IH, IW, OC, KD, KH, KW):
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OD, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_OC']),
        triton.cdiv(OW, meta['BLOCK_OW']),
        N * OD * OH,
    )

    conv3d_nhwc_mish_tanh_kernel[grid](
        x_nhwc, weight_nhwc, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
    )
    return out, OD, OH, OW


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

        # Keep an nn.Conv3d so init matches reference distribution
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Weight: [OC, IC, KD, KH, KW] -> [OC, KD, KH, KW, IC]
        w = conv.weight.data.detach().clone()
        w_nhwc = w.permute(0, 2, 3, 4, 1).contiguous()
        self.weight_nhwc = nn.Parameter(w_nhwc, requires_grad=False)
        self.bias = nn.Parameter(conv.bias.data.detach().clone(), requires_grad=False)

    def forward(self, x):
        # x: [N, IC, D, H, W] -> NDHWC
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()

        out_nhwc, OD, OH, OW = conv3d_mish_tanh_nhwc(
            x_nhwc, self.weight_nhwc, self.bias,
            N, IC, ID, IH, IW, self.out_channels, self.KD, self.KH, self.KW,
        )

        # Back to NCDHW
        out = out_nhwc.permute(0, 4, 1, 2, 3).contiguous()
        return out