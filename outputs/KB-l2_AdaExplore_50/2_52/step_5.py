import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'M_TOTAL', 'IC_KHKW'],
)
@triton.jit
def conv2d_nhwc_mish_bn_kernel(
    x_ptr,           # NHWC: [N, IH, IW, IC]
    w_ptr,           # KHKWIC_OC: [KH*KW*IC, OC]  (reduction axis contiguous in IC)
    b_ptr,           # [OC]
    scale_ptr,       # [OC]
    shift_ptr,       # [OC]
    out_ptr,         # NHWC: [N, OH, OW, OC]
    N, IH, IW, IC,
    OH, OW, OC,
    KH, KW,
    M_TOTAL, IC_KHKW,
    APPLY_BN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial-batch positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels

    # Decompose offs_m into (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    m_mask = offs_m < M_TOTAL
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K dim is iterated as (kh, kw, ic). Layout: w is [KH*KW*IC, OC], so K-stride = OC.
    # x_ptr offset:  n*IH*IW*IC + ih*IW*IC + iw*IC + ic
    IH_IW_IC = IH * IW * IC
    IW_IC = IW * IC

    for k_start in range(0, IC_KHKW, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < IC_KHKW

        # decompose K into (kh, kw, ic)
        kh = offs_k // (KW * IC)
        rem_k = offs_k % (KW * IC)
        kw = rem_k // IC
        ic = rem_k % IC

        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]

        x_offsets = (n_idx[:, None] * IH_IW_IC
                     + ih * IW_IC
                     + iw * IC
                     + ic[None, :])
        x_mask2 = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask2, other=0.0)

        # weight: [K, OC], stride along K is OC, along OC is 1
        w_offsets = offs_k[:, None] * OC + offs_n[None, :]
        w_mask2 = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_offsets, mask=w_mask2, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # bias
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b[None, :]

    # mish: x * tanh(softplus(x))
    x = acc
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    e2 = tl.exp(2.0 * sp)
    t = (e2 - 1.0) / (e2 + 1.0)
    y = x * t

    if APPLY_BN:
        scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=1.0)
        shift = tl.load(shift_ptr + offs_n, mask=n_mask, other=0.0)
        y = y * scale[None, :] + shift[None, :]

    # store NHWC
    out_offsets = (n_idx[:, None] * (OH * OW * OC)
                   + oh[:, None] * (OW * OC)
                   + ow[:, None] * OC
                   + offs_n[None, :])
    out_mask2 = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offsets, y, mask=out_mask2)


def _run_fused_conv_mish_bn(x_nhwc, w_k_oc, bias, scale, shift, N, IH, IW, IC, OH, OW, OC, KH, KW, apply_bn):
    M_TOTAL = N * OH * OW
    IC_KHKW = IC * KH * KW

    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    if scale is None:
        # Triton requires real pointers; pass dummy.
        scale = torch.empty(OC, device=x_nhwc.device, dtype=x_nhwc.dtype)
        shift = torch.empty(OC, device=x_nhwc.device, dtype=x_nhwc.dtype)

    conv2d_nhwc_mish_bn_kernel[grid](
        x_nhwc, w_k_oc, bias, scale, shift, out,
        N, IH, IW, IC,
        OH, OW, OC,
        KH, KW,
        M_TOTAL, IC_KHKW,
        APPLY_BN=apply_bn,
    )
    return out  # NHWC


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self._cached_w_k_oc = None
        self._cached_w_version = None

    def _get_weight_k_oc(self):
        # weight: [OC, IC, KH, KW] -> [KH, KW, IC, OC] -> reshape to [KH*KW*IC, OC]
        w = self.conv.weight
        if (self._cached_w_k_oc is not None
                and self._cached_w_version == w._version
                and self._cached_w_k_oc.device == w.device):
            return self._cached_w_k_oc
        w_perm = w.permute(2, 3, 1, 0).contiguous()  # [KH, KW, IC, OC]
        w_k_oc = w_perm.view(-1, w.shape[0]).contiguous()
        self._cached_w_k_oc = w_k_oc
        self._cached_w_version = w._version
        return w_k_oc

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        OH = IH - KH + 1
        OW = IW - KW + 1
        OC = self.out_channels

        # NHWC input
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_k_oc = self._get_weight_k_oc()
        bias = self.conv.bias.contiguous() if self.conv.bias is not None else torch.zeros(
            OC, device=x.device, dtype=x.dtype)

        if self.training:
            # Run fused conv+mish, then run BN in NCHW so it updates running stats.
            out_nhwc = _run_fused_conv_mish_bn(
                x_nhwc, w_k_oc, bias, None, None,
                N, IH, IW, IC, OH, OW, OC, KH, KW,
                apply_bn=False,
            )
            # to NCHW
            out_nchw = out_nhwc.permute(0, 3, 1, 2).contiguous()
            return self.bn(out_nchw)
        else:
            rm = self.bn.running_mean
            rv = self.bn.running_var
            bw = self.bn.weight
            bb = self.bn.bias
            invstd = torch.rsqrt(rv + self.eps)
            scale = (bw * invstd).contiguous()
            shift = (bb - rm * scale).contiguous()

            out_nhwc = _run_fused_conv_mish_bn(
                x_nhwc, w_k_oc, bias, scale, shift,
                N, IH, IW, IC, OH, OW, OC, KH, KW,
                apply_bn=True,
            )
            # back to NCHW
            return out_nhwc.permute(0, 3, 1, 2).contiguous()