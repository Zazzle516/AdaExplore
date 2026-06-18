import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KH_KW'],
)
@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias_add_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    OUT_HW,         # OH*OW
    N_OUT_HW,       # N*OH*OW
    IC_KH_KW,       # IC*KH*KW
    KH_KW,          # KH*KW
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # x is NHWC contiguous: stride = (H*W*IC, W*IC, IC, 1)
    # w is (OC, KH, KW, IC) contiguous: stride = (KH*KW*IC, KW*IC, IC, 1)
    # out is NHWC contiguous: stride = (OH*OW*OC, OW*OC, OC, 1)
    pid_m = tl.program_id(0)             # OC tile
    pid_n = tl.program_id(1)             # N*OUT_HW tile

    offs_oc = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_nhw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    n_idx = offs_nhw // OUT_HW
    hw_idx = offs_nhw % OUT_HW
    oh = hw_idx // OW
    ow = hw_idx % OW

    mask_oc = offs_oc < OC
    mask_nhw = offs_nhw < N_OUT_HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, IC_KH_KW, BLOCK_K):
        k = k_start + offs_k                    # [BLOCK_K]
        mask_k = k < IC_KH_KW

        # k indexes (kh, kw, ic) for NHWC weight layout
        khkw = k // IC
        ic = k % IC
        kh = khkw // KW
        kw_ = khkw % KW

        # weight tile [BLOCK_M, BLOCK_K]
        # w stride: oc * (KH*KW*IC) + kh*(KW*IC) + kw*IC + ic
        w_offsets = (offs_oc[:, None] * IC_KH_KW
                     + kh[None, :] * (KW * IC)
                     + kw_[None, :] * IC
                     + ic[None, :])
        w_mask = mask_oc[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # input tile [BLOCK_K, BLOCK_N]
        in_h = oh[None, :] + kh[:, None]
        in_w = ow[None, :] + kw_[:, None]

        # x stride NHWC: n*(H*W*IC) + h*(W*IC) + w*IC + ic
        x_offsets = (n_idx[None, :] * (H * W * IC)
                     + in_h * (W * IC)
                     + in_w * IC
                     + ic[:, None])
        x_mask = mask_k[:, None] & mask_nhw[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile)

    # bias from conv
    b = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # add learned bias (shape [OC])
    bias_add = tl.load(bias_add_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias_add[:, None]

    # store NHWC: n*(OH*OW*OC) + h*(OW*OC) + w*OC + oc
    out_offsets = (n_idx[None, :] * (OUT_HW * OC)
                   + oh[None, :] * (OW * OC)
                   + ow[None, :] * OC
                   + offs_oc[:, None])
    out_mask = mask_oc[:, None] & mask_nhw[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


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
            # (OC, IC, KH, KW) -> (OC, KH, KW, IC)
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

        # convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        if self.training:
            w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()
        else:
            w_nhwc = self._get_w_nhwc()

        cb = self.conv.bias.contiguous()
        bias_add = self.bias.contiguous().view(-1)

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        OUT_HW = OH * OW
        N_OUT_HW = N * OUT_HW
        IC_KH_KW = IC * KH * KW
        KH_KW = KH * KW

        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(N_OUT_HW, meta['BLOCK_N']),
        )

        conv_relu_bias_kernel[grid](
            x_nhwc, w_nhwc, cb, bias_add, out_nhwc,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            OUT_HW, N_OUT_HW, IC_KH_KW, KH_KW,
        )

        # convert back to NCHW
        return out_nhwc.permute(0, 3, 1, 2).contiguous()