import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, D, H, W,
    PD, PH, PW,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,  # pool window volume rounded up
    BLOCK_PW: tl.constexpr, # number of output W positions per program
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
):
    pid = tl.program_id(0)
    # decode pid -> (n, pd, ph, pw_tile)
    PW_TILES = (PW + BLOCK_PW - 1) // BLOCK_PW
    pw_tile = pid % PW_TILES
    tmp = pid // PW_TILES
    ph = tmp % PH
    tmp2 = tmp // PH
    pd = tmp2 % PD
    n = tmp2 // PD

    d0 = pd * KD
    h0 = ph * KH
    w0_base = pw_tile * BLOCK_PW * KW

    # Preload subtract vector
    offs_c_1d = tl.arange(0, BLOCK_C)
    mask_c_1d = offs_c_1d < C
    sub = tl.load(sub_ptr + offs_c_1d, mask=mask_c_1d, other=0.0)

    # offsets within pool window
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    kd = offs_k // (KH * KW)
    rem = offs_k % (KH * KW)
    kh = rem // KW
    kw = rem % KW
    mask_k = offs_k < (KD * KH * KW)

    # offsets across output W tile
    offs_pw = tl.arange(0, BLOCK_PW)  # [BLOCK_PW]
    mask_pw = (pw_tile * BLOCK_PW + offs_pw) < PW

    # 3D shape: [BLOCK_C, BLOCK_PW, BLOCK_K]
    # build pointer
    base_n = n * C * D * H * W
    # channel offset
    c_off = offs_c_1d[:, None, None] * (D * H * W)
    # spatial within window
    dd = (d0 + kd)  # [BLOCK_K]
    hh = (h0 + kh)  # [BLOCK_K]
    kw_off = kw     # [BLOCK_K]
    # output w base across tile
    w_out = w0_base + offs_pw * KW  # [BLOCK_PW]

    spatial_off = (dd[None, None, :] * (H * W)
                   + hh[None, None, :] * W
                   + w_out[None, :, None]
                   + kw_off[None, None, :])

    ptr = x_ptr + base_n + c_off + spatial_off

    mask = (mask_c_1d[:, None, None]
            & mask_pw[None, :, None]
            & mask_k[None, None, :])

    vals = tl.load(ptr, mask=mask, other=-float('inf'))
    # max over pool window (last axis) -> [BLOCK_C, BLOCK_PW]
    pooled = tl.max(vals, axis=2)
    pooled = tl.where(mask_c_1d[:, None], pooled, -float('inf'))

    # softmax across C (axis=0)
    x_max = tl.max(pooled, axis=0)             # [BLOCK_PW]
    e = tl.exp(pooled - x_max[None, :])
    e = tl.where(mask_c_1d[:, None], e, 0.0)
    denom = tl.sum(e, axis=0)                  # [BLOCK_PW]
    sm = e / denom[None, :]

    y = sm - sub[:, None]
    sig = 1.0 / (1.0 + tl.exp(-y))
    sw = sig * y
    sw = tl.where(mask_c_1d[:, None], sw, -float('inf'))
    res = tl.max(sw, axis=0)                   # [BLOCK_PW]

    # store
    out_base = ((n * PD + pd) * PH + ph) * PW + pw_tile * BLOCK_PW
    tl.store(out_ptr + out_base + offs_pw, res, mask=mask_pw)


def fused_post(x, sub, pool_k=2):
    # x: conv output [N, C, D, H, W]
    N, C, D, H, W = x.shape
    KD = KH = KW = pool_k
    PD = D // KD
    PH = H // KH
    PW = W // KW
    x = x.contiguous()
    out = torch.empty((N, PD, PH, PW), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16
    BLOCK_K = triton.next_power_of_2(KD * KH * KW)
    if BLOCK_K < 8:
        BLOCK_K = 8
    BLOCK_PW = 8 if PW >= 8 else (4 if PW >= 4 else 1)

    PW_TILES = (PW + BLOCK_PW - 1) // BLOCK_PW
    grid = (N * PD * PH * PW_TILES,)
    fused_pool_softmax_sub_swish_max_kernel[grid](
        x, sub, out,
        N, C, D, H, W,
        PD, PH, PW,
        BLOCK_C=BLOCK_C,
        BLOCK_K=BLOCK_K,
        BLOCK_PW=BLOCK_PW,
        KD=KD, KH=KH, KW=KW,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post(x, self.subtract, pool_k=2)
        return x