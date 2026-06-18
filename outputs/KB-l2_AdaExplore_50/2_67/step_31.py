import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_gelu_avgpool_persistent_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H, W,
    OH, OW,
    inv_npix,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
    IC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # Preload entire weight slice for this OC tile: (BLOCK_OC, IC*KH*KW)
    K = IC * KH * KW
    offs_k = tl.arange(0, IC * KH * KW)
    w_ptrs = offs_oc[:, None] * K + offs_k[None, :]
    w_tile = tl.load(w_ptrs + w_ptr, mask=mask_oc[:, None], other=0.0)  # (BLOCK_OC, K)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)

    npix = OH * OW
    n_blocks = tl.cdiv(npix, BLOCK_PIX)

    acc_sum = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    x_base = pid_n * stride_n

    for blk in range(0, n_blocks):
        offs_p = blk * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
        mask_p = offs_p < npix
        oh = offs_p // OW
        ow = offs_p % OW

        # Build im2col tile (K, BLOCK_PIX)
        # offs_k decomposition
        ic = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        ih = oh[None, :] + kh[:, None]  # (K, BLOCK_PIX)
        iw = ow[None, :] + kw[:, None]
        ic_b = ic[:, None]              # (K, 1)

        x_off = x_base + ic_b * stride_c + ih * stride_h + iw * stride_w
        x_tile = tl.load(x_ptr + x_off, mask=mask_p[None, :], other=0.0)  # (K, BLOCK_PIX)

        # matmul: (BLOCK_OC, K) x (K, BLOCK_PIX) -> (BLOCK_OC, BLOCK_PIX)
        acc = tl.dot(w_tile, x_tile)
        acc = acc + bias[:, None]

        # GELU
        inv_sqrt2 = 0.70710678118654752440
        gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

        gelu = tl.where(mask_p[None, :], gelu, 0.0)
        acc_sum += tl.sum(gelu, axis=1)

    out_val = acc_sum * inv_npix
    out_off = pid_n * OC + offs_oc
    tl.store(out_ptr + out_off, out_val, mask=mask_oc)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        npix = OH * OW
        inv_npix = 1.0 / npix

        out = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_PIX = 128

        grid = (N, triton.cdiv(OC, BLOCK_OC))

        conv_gelu_avgpool_persistent_kernel[grid](
            x, w, b, out,
            N, H, W,
            OH, OW,
            inv_npix,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            BLOCK_OC=BLOCK_OC,
            BLOCK_PIX=BLOCK_PIX,
            IC=IC,
            KH=KH,
            KW=KW,
            OC=OC,
            num_warps=4,
            num_stages=2,
        )

        return out