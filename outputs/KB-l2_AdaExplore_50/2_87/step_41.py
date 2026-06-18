import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'KH', 'KW', 'OH', 'OW', 'OC'],
)
@triton.jit
def conv2d_nchw_sub_mish_kernel(
    x_ptr,        # [N, IC, IH, IW] NCHW input
    w_ptr,        # [OC, KH*KW*IC] (each row is the flattened patch in (kh,kw,ic) order)
    b_ptr,        # [OC]
    out_ptr,      # [N, OH, OW, OC] NHWC output
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,        # IC*KH*KW
    SUB: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over N*OH*OW
    BLOCK_N: tl.constexpr,  # tile over OC
    BLOCK_K: tl.constexpr,  # padded K (>= K)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M] over flattened (N,OH,OW)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N] over OC
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K] reduction

    M = N * OH * OW
    m_mask = offs_m < M
    n_mask = offs_n < OC
    k_mask = offs_k < K

    # Decompose flattened m into (n_idx, oh, ow)
    ow_idx = offs_m % OW
    tmp = offs_m // OW
    oh_idx = tmp % OH
    n_idx = tmp // OH

    # Decompose k into (kh, kw, ic)
    ic_k = offs_k % IC
    tmp_k = offs_k // IC
    kw_k = tmp_k % KW
    kh_k = tmp_k // KW

    ih = oh_idx[:, None] + kh_k[None, :]   # [BLOCK_M, BLOCK_K]
    iw = ow_idx[:, None] + kw_k[None, :]   # [BLOCK_M, BLOCK_K]

    # x is NCHW: addr = ((n*IC + ic)*IH + ih)*IW + iw
    x_addr = ((n_idx[:, None] * IC + ic_k[None, :]) * IH + ih) * IW + iw
    x_load_mask = m_mask[:, None] & k_mask[None, :]
    x_tile = tl.load(x_ptr + x_addr, mask=x_load_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

    # w is [OC, K] row-major
    w_addr = offs_n[None, :] * K + offs_k[:, None]   # [BLOCK_K, BLOCK_N]
    w_load_mask = n_mask[None, :] & k_mask[:, None]
    w_tile = tl.load(w_ptr + w_addr, mask=w_load_mask, other=0.0)

    acc = tl.dot(x_tile, w_tile, out_dtype=tl.float32)   # [BLOCK_M, BLOCK_N]

    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b_vals[None, :] - SUB

    # Mish: x * tanh(softplus(x)); use stable softplus & tanh via 2*sigmoid(2x)-1
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    out = acc * th

    # Store NHWC: out_ptr[n, oh, ow, oc] = out_ptr[m, oc]
    # Linear addr = offs_m * OC + offs_n
    out_addr = offs_m[:, None] * OC + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_addr, out, mask=out_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-pack weight in shape [OC, KH*KW*IC] with (kh, kw, ic) ordering
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            # Permute to [OC, KH, KW, IC] then flatten last 3 dims
            w_packed = w.permute(0, 2, 3, 1).contiguous().view(out_channels, -1).contiguous()
        self.register_buffer('w_packed', w_packed.cuda())
        self.register_buffer('bias_buf', self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.cuda(non_blocking=True).contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Output in NHWC layout (will permute at end)
        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        K = IC * KH * KW
        BLOCK_K = _next_pow2(K)
        if BLOCK_K < 16:
            BLOCK_K = 16

        sub = float(self.subtract_value_1 + self.subtract_value_2)

        grid = lambda meta: (
            triton.cdiv(N * OH * OW, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv2d_nchw_sub_mish_kernel[grid](
            x, self.w_packed, self.bias_buf, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            K,
            sub,
            BLOCK_K=BLOCK_K,
        )
        # Permute NHWC -> NCHW
        return out_nhwc.permute(0, 3, 1, 2).contiguous()