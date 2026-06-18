import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convt_lse_hswish_sub_clamp_kernel(
    x_ptr,        # input [N, IC, ID, IH, IW]
    w_ptr,        # weight [IC, OC, KD, KH, KW]
    cb_ptr,       # conv bias [OC]
    bias_ptr,     # scalar
    out_ptr,      # output [N, 1, OD, OH, OW]
    N, IC: tl.constexpr, OC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # grid: (N, OD*OH, ceil(OW / BLOCK_W))
    n = tl.program_id(0)
    dh = tl.program_id(1)
    wb = tl.program_id(2)

    od = dh // OH
    oh = dh % OH

    ow_offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)  # [BLOCK_W]
    ow_mask = ow_offs < OW

    # Accumulator for all OC channels: [OC, BLOCK_W]
    acc = tl.zeros((OC, BLOCK_W), dtype=tl.float32)

    # Add conv bias
    cb = tl.load(cb_ptr + tl.arange(0, OC))  # [OC]
    acc += cb[:, None]

    # Compute positions
    # output coord -> need input coords such that:
    # out_pos = in_pos * STRIDE - PAD + k  =>  in_pos = (out_pos + PAD - k) / STRIDE
    # valid when (out_pos + PAD - k) % STRIDE == 0 and 0 <= in_pos < I*
    pd = od + PAD
    ph = oh + PAD
    pw = ow_offs + PAD  # [BLOCK_W]

    for kd in tl.static_range(0, KD):
        td = pd - kd
        if (td % STRIDE) == 0:
            id_ = td // STRIDE
            if (id_ >= 0) & (id_ < ID):
                for kh in tl.static_range(0, KH):
                    th = ph - kh
                    if (th % STRIDE) == 0:
                        ih = th // STRIDE
                        if (ih >= 0) & (ih < IH):
                            for kw in tl.static_range(0, KW):
                                tw = pw - kw  # [BLOCK_W]
                                kw_align = (tw % STRIDE) == 0
                                iw = tw // STRIDE
                                iw_valid = kw_align & (iw >= 0) & (iw < IW) & ow_mask
                                # Load IC input values at this (id_, ih, iw[:]): shape [IC, BLOCK_W]
                                ic_range = tl.arange(0, IC)
                                in_offs = (n * IC + ic_range)[:, None] * (ID * IH * IW) + \
                                          id_ * (IH * IW) + ih * IW + iw[None, :]
                                x_vals = tl.load(x_ptr + in_offs,
                                                 mask=iw_valid[None, :],
                                                 other=0.0)  # [IC, BLOCK_W]
                                # Load weight slice [IC, OC] at (kd, kh, kw)
                                oc_range = tl.arange(0, OC)
                                w_offs = ic_range[:, None] * (OC * KD * KH * KW) + \
                                         oc_range[None, :] * (KD * KH * KW) + \
                                         kd * (KH * KW) + kh * KW + kw
                                w_vals = tl.load(w_ptr + w_offs)  # [IC, OC]
                                # acc[oc, w] += sum_ic w[ic,oc] * x[ic,w]
                                acc += tl.dot(tl.trans(w_vals), x_vals)

    # LogSumExp across OC
    max_val = tl.max(acc, axis=0)  # [BLOCK_W]
    sum_exp = tl.sum(tl.exp(acc - max_val[None, :]), axis=0)  # [BLOCK_W]
    lse = max_val + tl.log(sum_exp)  # [BLOCK_W]

    # HardSwish
    hs = lse * tl.sigmoid(lse + 3.0) / 6.0

    # Subtract bias
    b = tl.load(bias_ptr)
    out = hs - b

    # Clamp
    out = tl.minimum(tl.maximum(out, -1.0), 1.0)

    # Store
    out_idx = n * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow_offs
    tl.store(out_ptr + out_idx, out, mask=ow_mask)


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        OC = self.out_channels
        OD = (ID - 1) * S - 2 * P + K
        OH = (IH - 1) * S - 2 * P + K
        OW = (IW - 1) * S - 2 * P + K

        out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

        w = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        cb = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias_flat = self.bias.contiguous().view(-1)[:1]

        BLOCK_W = 32
        grid = (N, OD * OH, (OW + BLOCK_W - 1) // BLOCK_W)
        fused_convt_lse_hswish_sub_clamp_kernel[grid](
            x, w, cb, bias_flat, out,
            N, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            K, K, K,
            S, P,
            BLOCK_W=BLOCK_W,
            num_warps=4,
            num_stages=2,
        )
        return out