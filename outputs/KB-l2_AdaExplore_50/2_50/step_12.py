import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'POD', 'POH', 'POW', 'IC'],
)
@triton.jit
def fused_convt_pool_bias_scale_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, POD, POH, POW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    scale1, scale2,
    BLOCK_OC: tl.constexpr,
):
    pid_spatial = tl.program_id(0)
    pid_oc = tl.program_id(1)

    # decode pid_spatial -> n, pd, ph, pw
    pw = pid_spatial % POW
    tmp = pid_spatial // POW
    ph = tmp % POH
    tmp = tmp // POH
    pd = tmp % POD
    n = tmp // POD

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # 8 spatial positions in the 2x2x2 pooling window
    for dd in tl.static_range(2):
        od = pd * 2 + dd
        for hh in tl.static_range(2):
            oh = ph * 2 + hh
            for ww in tl.static_range(2):
                ow = pw * 2 + ww
                for kd in tl.static_range(KD):
                    id_num = od + PD - kd
                    id_ = id_num // SD
                    id_valid = (id_num - id_ * SD == 0) & (id_ >= 0) & (id_ < ID)
                    for kh in tl.static_range(KH):
                        ih_num = oh + PH - kh
                        ih = ih_num // SH
                        ih_valid = (ih_num - ih * SH == 0) & (ih >= 0) & (ih < IH)
                        for kw in tl.static_range(KW):
                            iw_num = ow + PW - kw
                            iw = iw_num // SW
                            iw_valid = (iw_num - iw * SW == 0) & (iw >= 0) & (iw < IW)
                            spatial_valid = id_valid & ih_valid & iw_valid
                            for ic in range(IC):
                                x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                                x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)
                                w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc += x_val * w_val

    # avg pool: divide by 8. conv bias added at every position then divided by 8 = cb.
    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    extra_bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    val = (acc * (1.0 / 8.0) + cb) * scale1
    val = (val + extra_bias) * scale2

    out_off = ((n * OC + oc_offs) * POD + pd) * POH * POW + ph * POW + pw
    tl.store(out_ptr + out_off, val, mask=oc_mask)


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
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        PD, PH, PW = self.padding
        OC = self.out_channels

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        POD, POH, POW = OD // 2, OH // 2, OW // 2
        pooled = torch.empty((N, OC, POD, POH, POW), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        conv_bias = self.conv_transpose.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        scale1_val = float(self.scale1.item())
        scale2_val = float(self.scale2.item())

        grid = lambda meta: (N * POD * POH * POW, triton.cdiv(OC, meta['BLOCK_OC']))
        fused_convt_pool_bias_scale_kernel[grid](
            x, weight, conv_bias, bias_flat, pooled,
            N, IC, ID, IH, IW,
            OC, POD, POH, POW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            scale1_val, scale2_val,
        )

        return pooled