import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_configs():
    configs = []
    for bsp in [64, 128, 256]:
        for boc in [32, 64]:
            for nw in [4, 8]:
                for ns in [2, 3]:
                    configs.append(triton.Config(
                        {'BLOCK_OC': boc, 'BLOCK_SP': bsp},
                        num_warps=nw, num_stages=ns,
                    ))
    return configs


@triton.autotune(configs=_get_configs(), key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'])
@triton.jit
def conv2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    INV_DIV: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    # Hoisted base offsets
    x_n_base = pid_n * (IC * IH * IW)
    in_row_base = oh * IW + ow  # [BLOCK_SP] - base position in input plane (top-left)

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop over input channels and kernel positions
    K = IC * KH * KW
    for ic in range(0, IC):
        x_ic_base = x_n_base + ic * (IH * IW)
        w_ic_base = ic * (KH * KW)
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                x_off = x_ic_base + in_row_base + kh * IW + kw
                x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)
                w_off = oc_offs * K + w_ic_base + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                acc += w_val[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = acc * INV_DIV
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)

    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_SP']))

        conv2d_fused_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            1.0 / float(self.divisor),
            0.01,
        )
        return out