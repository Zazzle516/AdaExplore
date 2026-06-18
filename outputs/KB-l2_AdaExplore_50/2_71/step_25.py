import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IH, IW,
    OC, OH, OW,
    IC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    inv_div: tl.constexpr, neg_slope: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    oh = hw_offs // OW
    ow = hw_offs % OW

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    # x is NHWC: [N, IH, IW, IC]
    # w is OC-major: [OC, KH, KW, IC]
    n_base = pid_n * (IH * IW * IC)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(KH):
        ih = oh + kh  # [BLOCK_HW]
        ih_off = ih * (IW * IC)
        for kw in tl.static_range(KW):
            iw = ow + kw
            # x_block: [BLOCK_HW, BLOCK_IC]
            x_off = n_base + ih_off[:, None] + iw[:, None] * IC + ic_offs[None, :]
            x_block = tl.load(x_ptr + x_off, mask=hw_mask[:, None] & ic_mask[None, :], other=0.0)

            # w_block: [BLOCK_OC, BLOCK_IC]
            w_off = oc_offs[:, None] * (KH * KW * IC) + (kh * KW + kw) * IC + ic_offs[None, :]
            w_block = tl.load(w_ptr + w_off, mask=oc_mask[:, None] & ic_mask[None, :], other=0.0)

            # acc[BLOCK_OC, BLOCK_HW] += w_block @ x_block^T
            acc += tl.dot(w_block, tl.trans(x_block), allow_tf32=True)

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[:, None]
    acc = acc * inv_div
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # output is NCHW
    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + hw_offs[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.BLOCK_IC = max(16, _next_pow2(in_channels))
        # cached permuted weight (OC, KH, KW, IC)
        self._w_nhwc = None
        self._b_cuda = None

    def _prepare(self):
        if self._w_nhwc is None or not self._w_nhwc.is_cuda:
            self._w_nhwc = self.conv.weight.detach().cuda().permute(0, 2, 3, 1).contiguous()
            self._b_cuda = self.conv.bias.detach().contiguous().cuda()

    def forward(self, x):
        x = x.cuda()
        self._prepare()
        # convert to NHWC contiguous
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w = self._w_nhwc
        b = self._b_cuda

        N, IH, IW, IC = x_nhwc.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_HW']))

        conv2d_div_lrelu_kernel[grid](
            x_nhwc, w, b, out,
            N, IH, IW,
            OC, OH, OW,
            IC, self.BLOCK_IC, KH, KW,
            1.0 / float(self.divisor), 0.01,
        )
        return out