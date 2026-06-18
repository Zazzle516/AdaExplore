import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'KH', 'KW', 'OH', 'OW', 'OC'],
)
@triton.jit
def conv2d_implicit_gemm_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr, IC_C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    K_CONST: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    M = OH * OW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K_CONST)

    m_mask = offs_m < M
    n_mask = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    KHKW = KH * KW
    K = IC_C * KH * KW

    # decompose k indices
    ic = offs_k // KHKW
    rem = offs_k % KHKW
    kh = rem // KW
    kw = rem % KW
    k_mask = offs_k < K

    x_batch_base = pid_b * IC_C * IH * IW

    # Load x tile [BLOCK_M, K_CONST]
    ih = oh[:, None] + kh[None, :]
    iw = ow[:, None] + kw[None, :]
    x_addr = x_batch_base + ic[None, :] * (IH * IW) + ih * IW + iw
    x_load_mask = m_mask[:, None] & k_mask[None, :]
    x_vals = tl.load(x_ptr + x_addr, mask=x_load_mask, other=0.0)

    # Load w tile [K_CONST, BLOCK_N] - weight pre-transposed to [K, OC]
    w_addr = offs_k[:, None] * OC + offs_n[None, :]
    w_load_mask = k_mask[:, None] & n_mask[None, :]
    w_vals = tl.load(w_ptr + w_addr, mask=w_load_mask, other=0.0)

    acc = tl.dot(x_vals, w_vals)

    # bias (already pre-fused with -SUB)
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b_vals[None, :]

    # mish
    abs_acc = tl.abs(acc)
    sp = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-abs_acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    out_addr = pid_b * OC * M + offs_n[None, :] * M + offs_m[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_addr, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        sub = float(subtract_value_1 + subtract_value_2)
        # Pre-transpose weight from [OC, IC, KH, KW] to [K, OC] where K = IC*KH*KW
        K = in_channels * kernel_size * kernel_size
        with torch.no_grad():
            w = self.conv.weight.detach().reshape(out_channels, K).t().contiguous()
            b = self.conv.bias.detach() - sub
        self.register_buffer('w_t', w.cuda())
        self.register_buffer('b_fused', b.cuda())

    def forward(self, x):
        x = x.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        M = OH * OW
        K = IC * KH * KW
        K_CONST = triton.next_power_of_2(K)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
            N,
        )

        conv2d_implicit_gemm_mish_kernel[grid](
            x, self.w_t, self.b_fused, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, IC,
            K_CONST=K_CONST,
        )
        return out