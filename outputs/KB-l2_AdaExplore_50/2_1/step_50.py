import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'N_OUT_HW', 'IC_KH_KW'],
)
@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias_add_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    OUT_HW, N_OUT_HW,
    IC_KH_KW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # spatial tile id (rows of GEMM)
    pid_n = tl.program_id(1)   # OC tile id (cols of GEMM)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # spatial positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # OC indices

    mask_m = offs_m < N_OUT_HW
    mask_n = offs_n < OC

    n_idx = offs_m // OUT_HW
    hw    = offs_m % OUT_HW
    oh    = hw // OW
    ow    = hw % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    H_W_IC = H * W * IC
    W_IC   = W * IC

    # x base offset per row (no kh, kw, ic yet)
    x_base = n_idx * H_W_IC + oh * W_IC + ow * IC      # [BLOCK_M]
    # weight stride: w is (OC, KH, KW, IC), inner-most is ic
    # w_offset = oc * IC_KH_KW + (kh*KW + kw)*IC + ic
    w_oc_base = offs_n * IC_KH_KW                       # [BLOCK_N]

    # decompose k into (kh, kw, ic) where k = (kh*KW + kw)*IC + ic
    for k_start in range(0, IC_KH_KW, BLOCK_K):
        k = k_start + offs_k
        mask_k = k < IC_KH_KW

        khkw = k // IC
        ic   = k % IC
        kh   = khkw // KW
        kw   = khkw % KW

        # x tile [BLOCK_M, BLOCK_K]
        x_off = x_base[:, None] + kh[None, :] * W_IC + kw[None, :] * IC + ic[None, :]
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # w tile [BLOCK_K, BLOCK_N]
        w_off = w_oc_base[None, :] + k[:, None]
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    # conv bias [OC]
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # learned bias [OC]
    bias_add = tl.load(bias_add_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_add[None, :]

    # store directly to NCHW output: (N, OC, OH, OW)
    OC_OH_OW = OC * OUT_HW
    out_off = (n_idx[:, None] * OC_OH_OW
               + offs_n[None, :] * OUT_HW
               + oh[:, None] * OW
               + ow[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._cached_w_nhwc = None
        self._cached_w_version = None

    def _get_w_nhwc(self):
        w = self.conv.weight
        if (self._cached_w_nhwc is None
                or self._cached_w_version != w._version
                or self._cached_w_nhwc.device != w.device):
            self._cached_w_nhwc = w.detach().permute(0, 2, 3, 1).contiguous()
            self._cached_w_version = w._version
        return self._cached_w_nhwc

    def forward(self, x):
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = self.conv.kernel_size[0]
        KW = self.conv.kernel_size[1]
        OH = H - KH + 1
        OW = W - KW + 1

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        if self.training:
            w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()
        else:
            w_nhwc = self._get_w_nhwc()

        cb = self.conv.bias.contiguous()
        bias_add = self.bias.contiguous().view(-1)

        # output directly in NCHW
        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        OUT_HW = OH * OW
        N_OUT_HW = N * OUT_HW
        IC_KH_KW = IC * KH * KW

        grid = lambda meta: (
            triton.cdiv(N_OUT_HW, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv_relu_bias_kernel[grid](
            x_nhwc, w_nhwc, cb, bias_add, out,
            N, IC, H, W,
            OC, OH, OW,
            OUT_HW, N_OUT_HW,
            IC_KH_KW,
            KH, KW,
        )

        return out