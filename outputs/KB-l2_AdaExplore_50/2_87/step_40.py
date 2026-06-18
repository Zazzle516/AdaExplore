import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 72}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 72}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 72}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 72}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 32,  'BLOCK_K': 72}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'KH', 'KW', 'OH', 'OW', 'OC'],
)
@triton.jit
def conv2d_implicit_gemm_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M dimension: N * OH * OW (output spatial flattened)
    # N dimension: OC
    # K dimension: IC * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    M = OH * OW
    KHKW: tl.constexpr = KH * KW
    K = IC * KHKW
    IHIW = IH * IW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial idx within batch
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC idx
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < OC

    # Decompose offs_m into (oh, ow); precompute base input offset (without channel)
    oh = offs_m // OW
    ow = offs_m % OW
    spatial_base = oh * IW + ow  # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_batch_base = pid_b * IC * IHIW

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # decompose k_idx into (ic, kh, kw)
        ic = k_idx // KHKW
        rem = k_idx % KHKW
        kh = rem // KW
        kw = rem % KW

        # x address: [BLOCK_M, BLOCK_K]
        kernel_offset = kh * IW + kw  # [BLOCK_K]
        x_addr = x_batch_base + ic[None, :] * IHIW + spatial_base[:, None] + kernel_offset[None, :]
        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_addr, mask=x_load_mask, other=0.0)

        # w address: weight is [OC, IC, KH, KW] contiguous => oc * (IC*KH*KW) + k_idx
        # shape [BLOCK_K, BLOCK_N]
        w_addr = k_idx[:, None] + offs_n[None, :] * K
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_addr, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # bias + sub
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b_vals[None, :] - SUB

    # mish: x * tanh(softplus(x))
    # softplus stable: max(x,0) + log1p(exp(-|x|))
    abs_acc = tl.abs(acc)
    sp = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-abs_acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    # Transpose to [BLOCK_N, BLOCK_M] so stores are contiguous along M for each OC
    out_t = tl.trans(out)
    # output: [N, OC, OH, OW] => flat addr = b*OC*M + oc*M + m
    out_addr = pid_b * OC * M + offs_n[:, None] * M + offs_m[None, :]
    out_mask = n_mask[:, None] & m_mask[None, :]
    tl.store(out_ptr + out_addr, out_t, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        sub = float(self.subtract_value_1 + self.subtract_value_2)

        M = OH * OW
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
            N,
        )

        conv2d_implicit_gemm_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            sub,
        )
        return out