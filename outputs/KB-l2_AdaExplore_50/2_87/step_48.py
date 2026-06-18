import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'KH', 'KW', 'OH', 'OW', 'OC'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # M dimension: spatial positions OH*OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC tile
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    oh = offs_m // OW  # [BLOCK_M]
    ow = offs_m % OW

    # X is NHWC: shape (N, IH, IW, IC). Base for this batch:
    x_batch_base = pid_b * IH * IW * IC

    # W packed: (OC, KH*KW*IC), so for fixed OC, K stride is 1 along (kh,kw,ic)
    # W offset = oc * K_TOTAL + k
    w_row_base = offs_n * K_TOTAL  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K_TOTAL

        # decompose k_idx into (kh, kw, ic) where layout is kh*KW*IC + kw*IC + ic
        ic_idx = k_idx % IC
        khw = k_idx // IC
        kw_idx = khw % KW
        kh_idx = khw // KW

        # X load: x[b, oh+kh, ow+kw, ic]
        # offset = x_batch_base + (oh+kh)*IW*IC + (ow+kw)*IC + ic
        ih_idx = oh[:, None] + kh_idx[None, :]
        iw_idx = ow[:, None] + kw_idx[None, :]
        x_off = x_batch_base + ih_idx * (IW * IC) + iw_idx * IC + ic_idx[None, :]
        x_m = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)  # [BLOCK_M, BLOCK_K]

        # W load: w[oc, k]
        w_off = w_row_base[:, None] + k_idx[None, :]
        w_m = n_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_m, other=0.0)  # [BLOCK_N, BLOCK_K]

        acc += tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)

    # bias + sub
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b_vals[None, :] - SUB

    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    # Store as NCHW: out[b, oc, oh, ow]
    out_offset = (pid_b * OC * OH * OW
                  + offs_n[None, :] * (OH * OW)
                  + offs_m[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offset, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-pack weight: (OC, IC, KH, KW) -> (OC, KH, KW, IC) -> (OC, KH*KW*IC)
        with torch.no_grad():
            w = self.conv.weight.detach()
            w_packed = w.permute(0, 2, 3, 1).contiguous().view(out_channels, -1)
        self.register_buffer('w_packed', w_packed.cuda())
        self.register_buffer('bias_buf', self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.cuda()
        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        N, IH, IW, IC = x_nhwc.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        K_TOTAL = IC * KH * KW

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        sub = float(self.subtract_value_1 + self.subtract_value_2)

        grid = lambda meta: (
            triton.cdiv(OH * OW, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
            N,
        )

        conv2d_implicit_gemm_kernel[grid](
            x_nhwc, self.w_packed, self.bias_buf, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            sub,
            K_TOTAL,
        )
        return out