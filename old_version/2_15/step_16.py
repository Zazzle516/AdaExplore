import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'SPATIAL', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    SPATIAL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_start = pid_oc * BLOCK_N
    offs_oc = oc_start + tl.arange(0, BLOCK_N)
    mask_oc = offs_oc < OC

    sp_start = pid_sp * BLOCK_M
    offs_sp = sp_start + tl.arange(0, BLOCK_M)
    mask_sp = offs_sp < SPATIAL

    od = offs_sp // (OH * OW)
    rem = offs_sp % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    bias_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n = pid_n
    ic_stride = ID * IH * IW
    w_ic_stride = OC * KD * KH * KW

    offs_k = tl.arange(0, BLOCK_K)

    for kd in range(0, KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_val < ID) & (id_val >= 0)
        for kh in range(0, KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val < IH) & (ih_val >= 0)
            for kw in range(0, KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val < IW) & (iw_val >= 0)

                spatial_valid = id_valid & ih_valid & iw_valid & mask_sp

                x_base = (n * IC) * ic_stride + id_val * (IH * IW) + ih_val * IW + iw_val
                w_base_oc = (offs_oc * KD + kd) * (KH * KW) + kh * KW + kw

                for ic0 in range(0, IC, BLOCK_K):
                    ic_idx = ic0 + offs_k
                    ic_mask = ic_idx < IC

                    # x tile [BLOCK_M, BLOCK_K]
                    x_off = x_base[:, None] + ic_idx[None, :] * ic_stride
                    x_mask = spatial_valid[:, None] & ic_mask[None, :]
                    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                    # w tile [BLOCK_K, BLOCK_N]
                    w_off = w_base_oc[None, :] + ic_idx[:, None] * w_ic_stride
                    w_mask = ic_mask[:, None] & mask_oc[None, :]
                    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                    acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    acc += bias_vals[None, :]
    out_off = (n * OC + offs_oc[None, :]) * SPATIAL + offs_sp[:, None]
    full_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, acc, mask=full_mask)


@triton.jit
def bn_meansub_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, SPATIAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    base = (n * C + c) * SPATIAL

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift
        acc += tl.where(mask, y, 0.0)

    total = tl.sum(acc, axis=0)
    mean = total / SPATIAL

    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift - mean
        tl.store(out_ptr + base + idx, y, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW
    SPATIAL = OD * OH * OW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_N']), triton.cdiv(SPATIAL, meta['BLOCK_M']))

    conv_transpose3d_gemm_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        SPATIAL,
    )
    return out


def bn_meansub_triton(x, scale, shift):
    N, C, D, H, W = x.shape
    SPATIAL = D * H * W
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (N * C,)
    bn_meansub_kernel[grid](
        x, out, scale, shift,
        N, C, SPATIAL,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        if self.conv_transpose.bias is not None:
            conv_bias = self.conv_transpose.bias.contiguous()
        else:
            conv_bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        y = conv_transpose3d_triton(x, weight, conv_bias, self.stride, self.padding)

        # PyTorch BN (training mode) - matches reference
        y = self.batch_norm(y)

        # Subtract spatial mean per (n,c) using fused two-pass kernel
        OC = self.out_channels
        scale = torch.ones(OC, device=y.device, dtype=y.dtype)
        shift = torch.zeros(OC, device=y.device, dtype=y.dtype)
        y = bn_meansub_triton(y.contiguous(), scale, shift)
        return y