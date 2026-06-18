import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_mish_kernel_nhwc(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # x is NHWC: (N, IH, IW, IC)
    # w is OC, KH, KW, IC (reordered) — contiguous in IC
    # out is NCHW: (N, OC, OH, OW)
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    ic_range = tl.arange(0, BLOCK_IC)
    ic_mask = ic_range < IC

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    x_base = pid_n * (IH * IW * IC)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw

            # x (NHWC) tile [BLOCK_IC, BLOCK_SP]:
            # offset = n*IH*IW*IC + ih*IW*IC + iw*IC + ic
            x_off = (x_base
                     + ih[None, :] * (IW * IC)
                     + iw[None, :] * IC
                     + ic_range[:, None])
            x_m = ic_mask[:, None] & sp_mask[None, :]
            x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

            # w (OC, KH, KW, IC) tile [BLOCK_OC, BLOCK_IC]:
            # offset = oc*(KH*KW*IC) + kh*(KW*IC) + kw*IC + ic
            w_off = (oc_offs[:, None] * (KH * KW * IC)
                     + (kh * KW + kw) * IC
                     + ic_range[None, :])
            w_m = oc_mask[:, None] & ic_mask[None, :]
            w_tile = tl.load(w_ptr + w_off, mask=w_m, other=0.0)

            acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = acc - SUB

    # mish: x * tanh(softplus(x))
    sp_val = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp_val)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    out = acc * tanh_val

    out_off = (pid_n * OC * OH * OW
               + oc_offs[:, None] * (OH * OW)
               + sp_offs[None, :])
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Pre-permute weight to (OC, KH, KW, IC) contiguous
        with torch.no_grad():
            w = self.conv.weight.detach().clone()  # (OC, IC, KH, KW)
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)
        self.register_buffer('w_nhwc', w_nhwc)

    def forward(self, x):
        x = x.contiguous().cuda()
        # Permute input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w = self.w_nhwc
        if w.device != x.device:
            w = w.to(x.device)
            self.w_nhwc = w
        b = self.conv.bias.contiguous().to(x.device)

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        sub = float(self.subtract_value_1 + self.subtract_value_2)

        BLOCK_IC = 1
        while BLOCK_IC < IC:
            BLOCK_IC *= 2
        if BLOCK_IC < 16:
            BLOCK_IC = 16

        grid = lambda meta: (
            triton.cdiv(OH * OW, meta['BLOCK_SP']),
            triton.cdiv(OC, meta['BLOCK_OC']),
            N,
        )

        conv2d_mish_kernel_nhwc[grid](
            x_nhwc, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            sub,
            BLOCK_IC=BLOCK_IC,
        )
        return out