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
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'NHW', 'IC_KH_KW'],
)
@triton.jit
def conv2d_nhwc_fused_kernel(
    x_ptr,         # NHWC input: [N, H, W, IC]
    w_ptr,         # weight: [OC, KH*KW*IC] (rearranged from [OC,IC,KH,KW] -> [OC,KH,KW,IC])
    b_ptr,         # conv bias: [OC]
    bias_ptr,      # extra bias: [OC]
    out_ptr,       # NHWC output: [N, OH, OW, OC]
    N, H, W,
    OC, KH, KW,
    OH, OW,
    constant_value, scaling_factor,
    NHW, IC, IC_KH_KW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M axis: NHW (= N * OH * OW)
    # N axis: OC
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial-batch
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    m_mask = offs_m < NHW
    n_mask = offs_n < OC

    offs_m_safe = tl.where(m_mask, offs_m, 0)
    offs_n_safe = tl.where(n_mask, offs_n, 0)

    # decode m into (n_idx, oh, ow)
    n_idx = offs_m_safe // (OH * OW)
    rem   = offs_m_safe % (OH * OW)
    oh    = rem // OW
    ow    = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K = IC_KH_KW
    KW_IC = KW * IC

    # K dim is laid out as (kh, kw, ic) contiguous
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        offs_k_safe = tl.where(k_mask, offs_k, 0)

        kh = offs_k_safe // KW_IC
        kwic = offs_k_safe % KW_IC
        kw = kwic // IC
        ic = kwic % IC

        # input: x[n_idx, oh+kh, ow+kw, ic]
        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        x_offset = (n_idx[:, None] * (H * W * IC)
                    + ih * (W * IC)
                    + iw * IC
                    + ic[None, :])
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        # weight: w[oc, kh, kw, ic] flattened -> [OC, K]
        w_offset = offs_n_safe[None, :] * K + offs_k_safe[:, None]
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # conv bias [OC]
    b_vals = tl.load(b_ptr + offs_n_safe, mask=n_mask, other=0.0)
    acc += b_vals[None, :]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # extra bias [OC]
    eb = tl.load(bias_ptr + offs_n_safe, mask=n_mask, other=0.0)
    acc += eb[None, :]

    # scale
    acc = acc * scaling_factor

    # store NHWC: out[n_idx, oh, ow, oc]
    out_offset = (n_idx[:, None] * (OH * OW * OC)
                  + oh[:, None] * (OW * OC)
                  + ow[:, None] * OC
                  + offs_n[None, :])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-rearrange weight to [OC, KH, KW, IC] -> flatten last 3
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]
            self.register_buffer('w_packed', w_nhwc.view(w_nhwc.shape[0], -1).contiguous(), persistent=False)

    def _refresh_weight(self):
        w = self.conv.weight.detach()
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()
        self.w_packed = w_nhwc.view(w_nhwc.shape[0], -1).contiguous()

    def forward(self, x):
        x = x.cuda()
        # NHWC input
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        N, H, W, IC = x_nhwc.shape

        w_packed = self.w_packed
        if w_packed.device != x.device:
            w_packed = w_packed.to(x.device)
            self.w_packed = w_packed

        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        b = self.conv.bias.contiguous()
        bias_extra = self.bias.contiguous().view(-1)

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        NHW = N * OH * OW
        IC_KH_KW = IC * KH * KW

        grid = lambda meta: (
            (NHW + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
            (OC + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
        )

        conv2d_nhwc_fused_kernel[grid](
            x_nhwc, w_packed, b, bias_extra, out_nhwc,
            N, H, W,
            OC, KH, KW,
            OH, OW,
            float(self.constant_value), float(self.scaling_factor),
            NHW, IC, IC_KH_KW,
        )

        # convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out