import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _get_conv_configs():
    configs = []
    for bm in [32, 64, 128]:
        for bn in [64, 128, 256]:
            for bk in [16, 32, 64]:
                for nw in [4, 8]:
                    for ns in [2, 3, 4]:
                        configs.append(triton.Config(
                            {'BLOCK_OC': bm, 'BLOCK_HW': bn, 'BLOCK_K': bk},
                            num_warps=nw, num_stages=ns,
                        ))
    return configs


@triton.autotune(configs=_get_conv_configs(), key=['OC', 'OH', 'OW', 'IC'])
@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    x_base = pid_n * (IC * IH * IW)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD_H - kh
            iw_num = ow + PAD_W - kw
            ih = ih_num // STRIDE_H
            iw = iw_num // STRIDE_W
            valid = (ih_num >= 0) & (iw_num >= 0) & \
                    ((ih_num % STRIDE_H) == 0) & ((iw_num % STRIDE_W) == 0) & \
                    (ih < IH) & (iw < IW) & hw_mask
            spatial_idx = ih * IW + iw

            for k_start in range(0, IC, BLOCK_K):
                ic_offs = k_start + tl.arange(0, BLOCK_K)
                ic_mask = ic_offs < IC

                x_idx = x_base + ic_offs[:, None] * (IH * IW) + spatial_idx[None, :]
                x_mask = ic_mask[:, None] & valid[None, :]
                x_slab = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

                w_idx = ic_offs[None, :] * (OC * KH * KW) + oc_offs[:, None] * (KH * KW) + kh * KW + kw
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_slab = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

                acc += tl.dot(w_slab, x_slab, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = acc * multiplier

    out_idx = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + hw_offs[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def mean_hw_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    n = pid // C
    c = pid % C

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for start in range(0, HW, BLOCK):
        idx = start + offs
        mask = idx < HW
        x = tl.load(x_ptr + pid * HW + idx, mask=mask, other=0.0)
        acc += x

    s = tl.sum(acc, axis=0)
    mean_val = s / HW
    tl.store(out_ptr + pid, mean_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_HW']))

        conv_transpose_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            SH, SW,
            PH, PW,
            float(self.multiplier),
        )

        # Now mean over H, W
        out_mean = torch.empty((N, OC), device=x.device, dtype=x.dtype)
        HW = OH * OW
        BLOCK = 4096
        mean_hw_kernel[(N * OC,)](
            conv_out, out_mean,
            N, OC, HW,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=3,
        )

        return out_mean.view(N, OC, 1, 1)