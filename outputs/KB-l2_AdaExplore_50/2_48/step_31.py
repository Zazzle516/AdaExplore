import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_OC': 16, 'BLOCK_K': 32}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'ODHW', 'K_TOTAL'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    ODHW, K_TOTAL,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n_block = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_oc = tl.program_id(2)

    OHW = OH * OW

    spatial_offs = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    spatial_mask = spatial_offs < ODHW

    od = spatial_offs // OHW
    rem = spatial_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    IHW = IH * IW
    IDHW = ID * IHW
    KHW = KH * KW
    KDHW = KD * KHW

    # Base input offset per output position (for ic=0, kd=0, kh=0, kw=0)
    x_batch_base = pid_batch * IC * IDHW
    x_base = x_batch_base + od * IHW + oh * IW + ow  # [BLOCK_N]

    acc = tl.zeros([BLOCK_OC, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offs < K_TOTAL

        # decode k
        ic = k_offs // KDHW
        rem_k = k_offs % KDHW
        kd = rem_k // KHW
        rem_k2 = rem_k % KHW
        kh = rem_k2 // KW
        kw = rem_k2 % KW

        # weight tile [BLOCK_OC, BLOCK_K]
        w_offs = oc_offs[:, None] * K_TOTAL + k_offs[None, :]
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        # input tile [BLOCK_K, BLOCK_N]
        # offset = x_base[None,:] + ic*IDHW + kd*IHW + kh*IW + kw  per (k,n)
        k_input_off = ic * IDHW + kd * IHW + kh * IW + kw  # [BLOCK_K]
        x_offs = x_base[None, :] + k_input_off[:, None]    # [BLOCK_K, BLOCK_N]
        x_mask = k_mask[:, None] & spatial_mask[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    # Add conv bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_vals[:, None]

    # Epilogue
    scale_vals = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    bias2_vals = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)

    y = acc * scale_vals[:, None]
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    y = y * bias2_vals[:, None]
    y = tl.sigmoid(y)

    out_offs = pid_batch * OC * ODHW + oc_offs[:, None] * ODHW + spatial_offs[None, :]
    out_mask = oc_mask[:, None] & spatial_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=out_mask)


def conv3d_fused(x, weight, conv_bias, scale, bias):
    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    ODHW = OD * OH * OW
    K_TOTAL = IC * KD * KH * KW

    grid = lambda META: (
        (ODHW + META['BLOCK_N'] - 1) // META['BLOCK_N'],
        N,
        (OC + META['BLOCK_OC'] - 1) // META['BLOCK_OC'],
    )

    conv3d_fused_kernel[grid](
        x, weight, conv_bias,
        scale.contiguous().view(-1), bias.contiguous().view(-1),
        out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        ODHW, K_TOTAL,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return conv3d_fused(
            x, self.conv.weight, self.conv.bias,
            self.scaling_factor, self.bias,
        )