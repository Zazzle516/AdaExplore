import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'OD', 'OH', 'OW'],
)
@triton.jit
def fused_pool_bias_scale_kernel(
    in_ptr, bias_ptr, out_ptr,
    N, C, OD, OH, OW,
    POD, POH, POW,
    scale1,
    scale2,
    BLOCK: tl.constexpr,
):
    # one program processes BLOCK consecutive pooled outputs within a single (n,c) slice
    # grid: (POD*POH*POW / BLOCK, N*C)
    pid_s = tl.program_id(0)
    pid_nc = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    pooled_per_nc = POD * POH * POW
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < pooled_per_nc

    pw = offs % POW
    tmp = offs // POW
    ph = tmp % POH
    pd = tmp // POH

    base = ((n * C + c) * OD) * OH * OW
    d0 = pd * 2
    h0 = ph * 2
    w0 = pw * 2

    OHW = OH * OW

    o000 = base + d0 * OHW + h0 * OW + w0
    o001 = o000 + 1
    o010 = o000 + OW
    o011 = o010 + 1
    o100 = o000 + OHW
    o101 = o100 + 1
    o110 = o100 + OW
    o111 = o110 + 1

    v000 = tl.load(in_ptr + o000, mask=mask, other=0.0)
    v001 = tl.load(in_ptr + o001, mask=mask, other=0.0)
    v010 = tl.load(in_ptr + o010, mask=mask, other=0.0)
    v011 = tl.load(in_ptr + o011, mask=mask, other=0.0)
    v100 = tl.load(in_ptr + o100, mask=mask, other=0.0)
    v101 = tl.load(in_ptr + o101, mask=mask, other=0.0)
    v110 = tl.load(in_ptr + o110, mask=mask, other=0.0)
    v111 = tl.load(in_ptr + o111, mask=mask, other=0.0)

    acc = v000 + v001 + v010 + v011 + v100 + v101 + v110 + v111

    bias_val = tl.load(bias_ptr + c)

    # pooled = acc / 8 ; then *scale1 ; then +bias ; then *scale2
    val = (acc * (scale1 / 8.0) + bias_val) * scale2
    out_off = pid_nc * pooled_per_nc + offs
    tl.store(out_ptr + out_off, val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3

    def forward(self, x):
        x = x.contiguous()
        # Heavy op: full conv_transpose3d materialized via cuDNN
        out = F.conv_transpose3d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.stride,
            padding=self.padding,
        )
        N, OC, OD, OH, OW = out.shape
        POD, POH, POW = OD // 2, OH // 2, OW // 2

        pooled = torch.empty((N, OC, POD, POH, POW), device=x.device, dtype=x.dtype)

        # bias of shape (OC,1,1,1) - flatten to (OC,)
        bias_flat = self.bias.view(-1).contiguous()

        scale1_val = float(self.scale1.item())
        scale2_val = float(self.scale2.item())

        pooled_per_nc = POD * POH * POW
        grid = lambda meta: (triton.cdiv(pooled_per_nc, meta['BLOCK']), N * OC)
        fused_pool_bias_scale_kernel[grid](
            out, bias_flat, pooled,
            N, OC, OD, OH, OW,
            POD, POH, POW,
            scale1_val, scale2_val,
        )

        return pooled