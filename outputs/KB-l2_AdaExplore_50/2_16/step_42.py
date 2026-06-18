import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,        # [N, IC, IH, IW] contiguous
    w_ptr,        # [IC, OC, KH, KW] contiguous (ConvTranspose2d layout)
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    add_value: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,  # spatial tile size (OH*OW dim)
):
    # grid: (N, ceil(OC/BLOCK_OC), ceil(OH*OW / BLOCK_SP))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    # For ConvTranspose2d with stride=2, padding=1, kernel=3, output_padding=1:
    # output[n, oc, oh, ow] = sum over (ic, kh, kw) of x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where (oh + padding - kh) = ih * stride  =>  ih = (oh + padding - kh) / stride
    # Only valid when (oh + padding - kh) % stride == 0 and 0 <= ih < IH
    # padding=1, stride=2, kh in {0,1,2}
    # (oh + 1 - kh) must be even and >=0 and < 2*IH
    # For each oh, there are exactly 2 valid kh values (one even, one odd offset).
    # Similarly for ow/kw.

    # Hardcoded for stride=2, padding=1, kernel=3, output_padding=1
    # oh_p = oh + 1
    # kh candidates: kh such that (oh_p - kh) % 2 == 0, i.e. kh has same parity as oh_p
    # If oh_p is even: kh in {0, 2}
    # If oh_p is odd:  kh in {1}
    # ih = (oh_p - kh) // 2

    # We accumulate by looping over all 3x3 kernel positions but masking out invalid ones.
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Pre-compute strides
    x_n_stride = IC * IH * IW
    x_c_stride = IH * IW

    w_ic_stride = OC * 3 * 3
    w_oc_stride = 3 * 3

    for kh in tl.static_range(0, 3):
        oh_p = oh + 1 - kh  # [BLOCK_SP]
        ih = oh_p // 2
        valid_h = ((oh_p % 2) == 0) & (oh_p >= 0) & (ih < IH) & (ih >= 0)

        for kw in tl.static_range(0, 3):
            ow_p = ow + 1 - kw  # [BLOCK_SP]
            iw = ow_p // 2
            valid_w = ((ow_p % 2) == 0) & (ow_p >= 0) & (iw < IW) & (iw >= 0)
            valid = valid_h & valid_w & sp_mask  # [BLOCK_SP]

            # Load x[n, :, ih, iw] for all IC, and w[:, oc_offs, kh, kw]
            # Then accumulate: acc[sp, oc] += sum_ic x[ic, ih[sp], iw[sp]] * w[ic, oc, kh, kw]
            # We do this as a small GEMM: K=IC, M=BLOCK_SP, N=BLOCK_OC
            # But the (ih, iw) varies per sp, so we need gather per sp.

            for ic_start in tl.static_range(0, 64, 16):
                ic_offs = ic_start + tl.arange(0, 16)  # [16]
                # x ptrs: [BLOCK_SP, 16]
                x_offs = (pid_n * x_n_stride
                          + ic_offs[None, :] * x_c_stride
                          + ih[:, None] * IW
                          + iw[:, None])
                x_mask = valid[:, None]
                x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_SP, 16]

                # w ptrs: [16, BLOCK_OC]
                w_offs = (ic_offs[:, None] * w_ic_stride
                          + oc_offs[None, :] * w_oc_stride
                          + kh * 3 + kw)
                w_mask = oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [16, BLOCK_OC]

                acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    sp_val = tl.log(1.0 + tl.exp(-tl.abs(acc))) + tl.maximum(acc, 0.0)
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    y = acc * tanh_sp
    y = y + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale

    # Store: out[n, oc, oh*OW+ow] - layout [N, OC, OH*OW]
    out_offs = (pid_n * OC * OH * OW
                + oc_offs[None, :] * (OH * OW)
                + sp_offs[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=out_mask)


def conv_transpose_fused(x, weight, bias, add_value, scale):
    N, IC, IH, IW = x.shape
    _, OC, KH, KW = weight.shape
    # ConvTranspose2d output shape: (IH - 1)*stride - 2*padding + kernel + output_padding
    # = (IH-1)*2 - 2 + 3 + 1 = 2*IH
    OH = 2 * IH
    OW = 2 * IW

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

    conv_transpose_fused_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        float(add_value), float(scale),
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        # Check if we can use fast path
        self.use_fast = (
            kernel_size == 3 and stride == 2 and padding == 1 and output_padding == 1
            and in_channels == 64 and out_channels == 64
        )

    def forward(self, x):
        if self.use_fast and x.is_cuda:
            return conv_transpose_fused(
                x, self.conv_transpose.weight, self.conv_transpose.bias,
                self.add_value, self.scale,
            )
        # Fallback
        x = self.conv_transpose(x)
        x = F.mish(x)
        x = x + self.add_value
        x = F.hardtanh(x, min_val=-1, max_val=1)
        x = x * self.scale
        return x