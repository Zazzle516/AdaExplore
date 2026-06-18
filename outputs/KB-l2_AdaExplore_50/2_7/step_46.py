import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'M', 'K'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,
    M, K,  # M = OD*OH*OW, K = IC*KD*KH*KW
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n_batch = tl.program_id(0)  # combined (batch, M-tile)
    pid_oc = tl.program_id(1)       # OC tile

    num_m_tiles = tl.cdiv(M, BLOCK_M)
    batch_idx = pid_n_batch // num_m_tiles
    m_tile_idx = pid_n_batch % num_m_tiles

    offs_m = m_tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # decode offs_m into (od, oh, ow)
    od = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    m_mask = offs_m < M
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW
    KDHW = KD * KHW
    ICKDHW = IC * KDHW

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # decode k -> (ic, kd, kh, kw)
        ic = k_idx // KDHW
        krem = k_idx % KDHW
        kd = krem // KHW
        krem2 = krem % KHW
        kh = krem2 // KW
        kw = krem2 % KW

        # input coords
        in_d = od[:, None] + kd[None, :]   # [BLOCK_M, BLOCK_K]
        in_h = oh[:, None] + kh[None, :]
        in_w = ow[:, None] + kw[None, :]

        x_offset = (batch_idx * stride_xn
                    + ic[None, :] * stride_xc
                    + in_d * stride_xd
                    + in_h * stride_xh
                    + in_w * stride_xw)

        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_block = tl.load(x_ptr + x_offset, mask=x_load_mask, other=0.0)

        # weight: [OC, IC, KD, KH, KW] -> [OC, K]
        # offset = oc * K + k
        w_offset = offs_n[None, :] * K + k_idx[:, None]  # [BLOCK_K, BLOCK_N]
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_block = tl.load(w_ptr + w_offset, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_block, w_block)

    # Add conv bias (per-OC)
    cb = tl.load(conv_bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + cb[None, :]

    # Activations: ReLU -> LeakyReLU(0.01) -> GELU -> Sigmoid -> + extra_bias
    y = tl.maximum(acc, 0.0)
    # leaky relu on y (y >= 0 already, no-op but keep for correctness)
    y = tl.where(y >= 0.0, y, y * 0.01)
    # GELU (exact)
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    # sigmoid
    y = tl.sigmoid(y)
    # extra bias per OC
    eb = tl.load(extra_bias_ptr + offs_n, mask=n_mask, other=0.0)
    y = y + eb[None, :]

    # store: out[N, OC, OD, OH, OW]
    out_offset = (batch_idx * stride_on
                  + offs_n[None, :] * stride_oc
                  + od[:, None] * stride_od
                  + oh[:, None] * stride_oh
                  + ow[:, None] * stride_ow)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offset, y, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size

        # weight: [OC, IC, KD, KH, KW] -> flatten to [OC, K]
        w = conv.weight.data.contiguous().view(out_channels, -1).contiguous()
        self.weight = nn.Parameter(w)
        self.conv_bias = nn.Parameter(conv.bias.data.contiguous())
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kd, self.kh, self.kw
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        M = OD * OH * OW
        K = IC * KD * KH * KW

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        bias_flat = self.bias.contiguous().view(-1)

        sxn, sxc, sxd, sxh, sxw = x.stride()
        son, soc, sod, soh, sow = out.stride()

        def grid(meta):
            return (
                N * triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(OC, meta['BLOCK_N']),
            )

        conv3d_fused_kernel[grid](
            x, self.weight, self.conv_bias, bias_flat, out,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            OD, OH, OW,
            M, K,
            sxn, sxc, sxd, sxh, sxw,
            son, soc, sod, soh, sow,
        )
        return out