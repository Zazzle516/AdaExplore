import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW', 'N'],
)
@triton.jit
def conv2d_relu_bias_nhwc_kernel(
    x_ptr,         # x in NHWC: [N, IH, IW, IC]
    w_ptr,         # w in (KH, KW, IC, OC)
    conv_b_ptr,    # [OC]
    bias_ptr,      # [OC]
    out_ptr,       # NHWC out: [N, OH, OW, OC]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)   # OC tile (N-axis of GEMM)
    pid_m = tl.program_id(1)   # spatial+batch tile (M-axis of GEMM)

    HW = OH * OW
    NHW = N * HW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # output row index in [0, NHW)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # OC index

    nb = offs_m // HW
    hw = offs_m % HW
    oh = hw // OW
    ow = hw % OW

    m_mask = offs_m < NHW
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # Loop over kh, kw, then ic blocks. For 3x3, KH/KW=3 so they unroll.
    for kh in tl.static_range(0, KH):
        ih = oh + kh  # [BLOCK_M]
        for kw in tl.static_range(0, KW):
            iw = ow + kw  # [BLOCK_M]
            # base pointer in x for this (kh, kw) at all M positions
            x_base = nb * (IH * IW * IC) + ih * (IW * IC) + iw * IC  # [BLOCK_M]
            # base pointer in w for this (kh, kw): w[kh, kw, :, :]
            w_base = kh * (KW * IC * OC) + kw * (IC * OC)            # scalar

            for ic_start in range(0, IC, BLOCK_K):
                ic = ic_start + offs_k  # [BLOCK_K]
                ic_mask = ic < IC

                # Load x: [BLOCK_M, BLOCK_K]
                x_ptrs = x_ptr + x_base[:, None] + ic[None, :]
                x_load_mask = m_mask[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                # Load w: [BLOCK_K, BLOCK_N], w[kh, kw, ic, oc]
                w_ptrs = w_ptr + w_base + ic[:, None] * OC + offs_n[None, :]
                w_load_mask = ic_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # add conv bias (per-OC)
    cb = tl.load(conv_b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += cb[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # add extra bias (per-OC)
    eb = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc += eb[None, :]

    # Store NHWC: out[nb, oh, ow, oc]
    out_ptrs = out_ptr + offs_m[:, None] * OC + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self._w_nhwc_cached = None
        self._w_version = None

    def _get_w_nhwc(self):
        # weight is (OC, IC, KH, KW) -> we want (KH, KW, IC, OC) contiguous
        w = self.conv.weight
        ver = w._version
        if (self._w_nhwc_cached is None) or (self._w_version != ver):
            w_nhwc = w.detach().permute(2, 3, 1, 0).contiguous()
            self._w_nhwc_cached = w_nhwc
            self._w_version = ver
        return self._w_nhwc_cached

    def forward(self, x):
        # Transpose to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_nhwc = self._get_w_nhwc()

        cb = self.conv.bias.contiguous()
        eb = self.bias.reshape(-1).contiguous()

        N, IH, IW, IC = x_nhwc.shape
        KH, KW, _, OC = w_nhwc.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        NHW = N * OH * OW
        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_N']),
            triton.cdiv(NHW, meta['BLOCK_M']),
        )

        conv2d_relu_bias_nhwc_kernel[grid](
            x_nhwc, w_nhwc, cb, eb, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
        )

        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out