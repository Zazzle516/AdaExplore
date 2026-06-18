import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=2, num_stages=3),
    ],
    key=['B', 'OH', 'OW', 'OC', 'IC'],
)
@triton.jit
def conv_relu_hardswish_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, IH, IW,
    OC, OH, OW,
    IC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    N_OUT,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, BLOCK_IC)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC
    mask_ic = offs_ic < IC

    # decompose n -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    bn = tmp // OH

    IHW = IH * IW
    base_b = bn * (IC * IHW)

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            w_off_base = (kh * KW + kw) * IC * OC
            w_offs = w_off_base + offs_ic[:, None] * OC + offs_oc[None, :]
            w_mask = mask_ic[:, None] & mask_oc[None, :]
            w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

            ih_kh = oh + kh
            iw_kw = ow + kw
            x_offs = (base_b[:, None] +
                      offs_ic[None, :] * IHW +
                      ih_kh[:, None] * IW +
                      iw_kw[:, None])
            x_mask = mask_n[:, None] & mask_ic[None, :]
            x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

            acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    acc = tl.maximum(acc, 0.0)
    hs = (acc + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    out = acc * hs

    OHW = OH * OW
    y_offs = (bn[:, None] * (OC * OHW) +
              offs_oc[None, :] * OHW +
              oh[:, None] * OW +
              ow[:, None])
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(y_ptr + y_offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        with torch.no_grad():
            w = self.conv.weight.detach().cuda().contiguous()
            OC, IC, KH, KW = w.shape
            w_perm = w.permute(2, 3, 1, 0).contiguous()  # (KH, KW, IC, OC)
            self.register_buffer('w_packed', w_perm.view(-1).contiguous())
            self.register_buffer('b_packed', self.conv.bias.detach().cuda().contiguous())

    def forward(self, x):
        x = x.cuda().contiguous()
        B, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        y = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW
        BLOCK_IC = 16
        while BLOCK_IC < IC:
            BLOCK_IC *= 2

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv_relu_hardswish_nchw_kernel[grid](
            x, self.w_packed, self.b_packed, y,
            B, IH, IW,
            OC, OH, OW,
            IC, KH, KW,
            BLOCK_IC,
            N_OUT,
        )
        return y