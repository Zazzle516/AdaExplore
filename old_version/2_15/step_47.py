import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose3d implemented as a GEMM with im2col-style gather.
# Per program: compute a tile of (BLOCK_M output spatial positions) x (BLOCK_N output channels)
# Reduction axis K = IC * KD * KH * KW
# For each k in the reduction:
#   ic, kd, kh, kw = unravel(k)
#   id = (od + PD - kd) / SD  if divisible & in range
#   ih = (oh + PH - kh) / SH
#   iw = (ow + PW - kw) / SW
#   x_tile[m, k] = x[n, ic, id, ih, iw]  (0 if invalid)
#   w_tile[k, n] = weight[ic, oc, kd, kh, kw]
# acc += dot(x_tile, w_tile)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
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
    KHW, KDHW, K_TOTAL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)  # spatial tile
    pid_oc = tl.program_id(2)  # oc tile

    DHW = OD * OH * OW
    HW = OH * OW

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < DHW
    n_mask = n_offs < OC

    od = m_offs // HW
    rem = m_offs % HW
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_batch_off = pid_n * IC * ID * IH * IW

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_TOTAL

        # unravel k -> ic, kd, kh, kw
        ic = k_offs // KDHW
        krem = k_offs % KDHW
        kd = krem // KHW
        krem2 = krem % KHW
        kh = krem2 // KW
        kw = krem2 % KW

        # input spatial index per (m, k)
        # id_num[m,k] = od[m] + PD - kd[k]
        id_num = od[:, None] + PD - kd[None, :]
        ih_num = oh[:, None] + PH - kh[None, :]
        iw_num = ow[:, None] + PW - kw[None, :]

        id_ = id_num // SD
        ih_ = ih_num // SH
        iw_ = iw_num // SW

        valid = ((id_num >= 0) & (id_num % SD == 0) & (id_ >= 0) & (id_ < ID) &
                 (ih_num >= 0) & (ih_num % SH == 0) & (ih_ >= 0) & (ih_ < IH) &
                 (iw_num >= 0) & (iw_num % SW == 0) & (iw_ >= 0) & (iw_ < IW))

        x_off = (x_batch_off
                 + ic[None, :] * (ID * IH * IW)
                 + id_ * (IH * IW)
                 + ih_ * IW
                 + iw_)

        x_load_mask = valid & m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # weight: w[ic, oc, kd, kh, kw]; shape (IC, OC, KD, KH, KW)
        # offset = ic*OC*KDHW + oc*KDHW + kd*KHW + kh*KW + kw
        w_off = (ic[:, None] * (OC * KDHW)
                 + n_offs[None, :] * KDHW
                 + kd[:, None] * KHW
                 + kh[:, None] * KW
                 + kw[:, None])

        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
    acc += bias[None, :]

    out_off = (pid_n * OC * DHW
               + n_offs[None, :] * DHW
               + m_offs[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose3d_triton(x, w, b, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC2, OC, KD, KH, KW = w.shape
    assert IC == IC2
    SD, SH, SW = stride, stride, stride
    PD, PH, PW = padding, padding, padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    KHW = KH * KW
    KDHW = KD * KH * KW
    K_TOTAL = IC * KDHW

    grid = lambda META: (
        N,
        triton.cdiv(OD * OH * OW, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    conv_transpose3d_gemm_kernel[grid](
        x, w, b, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        KHW, KDHW, K_TOTAL,
    )
    return out


# Fused BN + spatial-mean subtraction.
# BN(train): y_bn = (y - mu_c)/sqrt(var_c+eps)*gamma + beta
# Subtract spatial mean per (N,C): final = y_bn - mean_spatial(y_bn)
#   = gamma/sqrt(var_c+eps) * (y - mean_spatial(y))
# We compute scale = gamma/sqrt(var_c+eps) using torch (cheap), then
# fuse mean(y) computation and write of scale*(y - mean(y)) in one kernel.

@triton.jit
def fused_submean_kernel(
    x_ptr, scale_ptr, out_ptr,
    SP,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n,c) row
    c = tl.program_id(1)
    C = tl.num_programs(1)

    row_id = pid * C + c  # not needed; we use pid as n and c separately
    # Actually we'll launch as (N*C,) flattened; redo:
    pass


@triton.jit
def fused_submean_kernel2(
    x_ptr, scale_ptr, out_ptr,
    N, C, SP,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = n * C * SP + c * SP

    # First pass: compute mean
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, SP, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SP
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        acc += v
    mean = tl.sum(acc) / SP

    scale = tl.load(scale_ptr + c)

    # Second pass: write
    for off in range(0, SP, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SP
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        out = scale * (v - mean)
        tl.store(out_ptr + base + idx, out, mask=mask)


def fused_submean(x, scale):
    N, C, D, H, W = x.shape
    SP = D * H * W
    out = torch.empty_like(x)
    BLOCK = 2048
    grid = (N * C,)
    fused_submean_kernel2[grid](
        x, scale, out,
        N, C, SP,
        BLOCK=BLOCK, num_warps=8, num_stages=2,
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
        w = self.conv_transpose.weight.contiguous()
        if self.conv_transpose.bias is not None:
            b = self.conv_transpose.bias.contiguous()
        else:
            b = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        y = conv_transpose3d_triton(x, w, b, self.stride, self.padding)

        # Replicate training-mode BN3d normalization, then subtract spatial mean.
        # Algebraically simplifies to scale * (y - spatial_mean(y))
        # where scale = gamma / sqrt(var_c + eps), var_c computed across (N,D,H,W).
        eps = self.batch_norm.eps
        dims = (0, 2, 3, 4)
        var = y.var(dim=dims, unbiased=False)
        gamma = self.batch_norm.weight
        scale = (gamma / torch.sqrt(var + eps)).contiguous()

        out = fused_submean(y, scale)
        return out