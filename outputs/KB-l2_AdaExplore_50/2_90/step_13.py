import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'IC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr,         # (N, ID, IH, IW, IC) channels-last
    w_ptr,         # (OC, KD*KH*KW*IC)
    b_ptr,         # (OC,)
    sum_ptr,       # (OC,)
    out_ptr,       # (N, OC, OD, OH, OW)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n_blk = tl.program_id(0)   # output spatial tile (within sample)
    pid_oc_blk = tl.program_id(1)  # OC tile
    pid_batch = tl.program_id(2)   # batch index

    OUT_SPATIAL = OD * OH * OW

    offs_n = pid_n_blk * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial positions
    offs_oc = pid_oc_blk * BLOCK_OC + tl.arange(0, BLOCK_OC)  # output channels

    n_mask = offs_n < OUT_SPATIAL
    oc_mask = offs_oc < OC

    # Decompose offs_n into (od, oh, ow)
    od = offs_n // (OH * OW)
    rem = offs_n % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    K_TOTAL = KD * KH * KW * IC_C  # full K dimension

    # weight: (OC, K_TOTAL) — row-major
    # x: (N, ID, IH, IW, IC) — channels last

    acc = tl.zeros((BLOCK_OC, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # base pointer for this batch
    x_batch_ptr = x_ptr + pid_batch * ID * IH * IW * IC_C

    # Loop over K in BLOCK_K chunks
    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_valid = k_idx < K_TOTAL

        # Decompose k_idx -> (kd, kh, kw, ic)
        ic = k_idx % IC_C
        tmp = k_idx // IC_C
        kw = tmp % KW
        tmp2 = tmp // KW
        kh = tmp2 % KH
        kd = tmp2 // KH

        # Compute input spatial coords
        # id_pos = od + kd, ih_pos = oh + kh, iw_pos = ow + kw (stride=1, pad=0)
        id_pos = od[:, None] + kd[None, :]   # [BLOCK_N, BLOCK_K]
        ih_pos = oh[:, None] + kh[None, :]
        iw_pos = ow[:, None] + kw[None, :]
        ic_pos = ic[None, :]                  # [1, BLOCK_K]

        # Compute address into x: id*IH*IW*IC + ih*IW*IC + iw*IC + ic
        x_offs = (id_pos * (IH * IW * IC_C)
                  + ih_pos * (IW * IC_C)
                  + iw_pos * IC_C
                  + ic_pos)  # [BLOCK_N, BLOCK_K]

        x_mask = n_mask[:, None] & k_valid[None, :]
        x_tile = tl.load(x_batch_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

        # weight: w_ptr[oc, k] -> shape [BLOCK_OC, BLOCK_K]
        w_offs = offs_oc[:, None] * K_TOTAL + k_idx[None, :]
        w_mask = oc_mask[:, None] & k_valid[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_K]

        # acc += w_tile @ x_tile.T  => [BLOCK_OC, BLOCK_N]
        acc += tl.dot(w_tile, tl.trans(x_tile))

    # Bias + sum_tensor (per OC)
    bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    sum_v = tl.load(sum_ptr + offs_oc, mask=oc_mask, other=0.0)

    acc = acc + bias[:, None]
    # LeakyReLU
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)
    # Add sum_tensor
    acc = acc + sum_v[:, None]
    # Clamp [-1, 1]
    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)
    # GELU exact
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store: out[n, oc, od, oh, ow]  shape (N, OC, OD*OH*OW)
    out_batch_ptr = out_ptr + pid_batch * OC * OUT_SPATIAL
    out_offs = offs_oc[:, None] * OUT_SPATIAL + offs_n[None, :]
    out_mask = oc_mask[:, None] & n_mask[None, :]
    tl.store(out_batch_ptr + out_offs, acc, mask=out_mask)


def conv3d_fused(x, weight, bias, sum_tensor, neg_slope=0.2):
    """
    x: (N, IC, ID, IH, IW)
    weight: (OC, IC, KD, KH, KW)
    bias: (OC,)
    sum_tensor: (OC, 1, 1, 1)
    Returns: (N, OC, OD, OH, OW)
    """
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    # Permute x to channels-last: (N, ID, IH, IW, IC)
    x_cl = x.permute(0, 2, 3, 4, 1).contiguous()

    # Reshape weight to (OC, KD*KH*KW*IC) where K-dim ordering matches x decomp:
    # k -> (kd, kh, kw, ic), so weight needs to be (OC, KD, KH, KW, IC)
    w_cl = weight.permute(0, 2, 3, 4, 1).contiguous().view(OC, KD * KH * KW * IC)

    sum_flat = sum_tensor.contiguous().view(-1)

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    OUT_SPATIAL = OD * OH * OW

    grid = lambda meta: (
        triton.cdiv(OUT_SPATIAL, meta['BLOCK_N']),
        triton.cdiv(OC, meta['BLOCK_OC']),
        N,
    )

    conv3d_fused_kernel[grid](
        x_cl, w_cl, bias, sum_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        IC_C=IC,
        NEG_SLOPE=neg_slope,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        return conv3d_fused(
            x,
            self.conv.weight,
            self.conv.bias,
            self.sum_tensor,
            neg_slope=0.2,
        )