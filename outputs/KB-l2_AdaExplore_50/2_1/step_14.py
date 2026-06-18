import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N_OHOW", "OC", "IC", "KH", "KW"])
@triton.jit
def conv2d_relu_bias_nhwc_kernel(
    x_ptr,        # NHWC: [N, IH, IW, IC]
    w_ptr,        # [OC, KH, KW, IC]
    b_ptr,        # [OC]
    bias2_ptr,    # [OC]
    out_ptr,      # NHWC: [N, OH, OW, OC]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    N_OHOW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ohow = OH * OW
    n_idx = offs_m // ohow
    rem_m = offs_m % ohow
    oh = rem_m // OW
    ow = rem_m % OW

    m_mask = offs_m < N_OHOW
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            x_row_base = (n_idx * IH * IW + ih * IW + iw) * IC
            w_col_base = (offs_n * KH * KW + kh * KW + kw) * IC

            x_offs = x_row_base[:, None] + offs_k[None, :]
            x_vals = tl.load(x_ptr + x_offs, mask=m_mask[:, None], other=0.0)

            w_offs = w_col_base[None, :] + offs_k[:, None]
            w_vals = tl.load(w_ptr + w_offs, mask=n_mask[None, :], other=0.0)

            acc += tl.dot(x_vals, w_vals)

    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    bias2_vals = tl.load(bias2_ptr + offs_n, mask=n_mask, other=0.0)

    acc = acc + b_vals[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    acc = acc + bias2_vals[None, :]

    # Store NHWC: out[n, oh, ow, oc] -> offset = (n*OH*OW + oh*OW + ow) * OC + oc
    out_row = n_idx * OH * OW + oh * OW + ow
    out_offs = out_row[:, None] * OC + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv2d_relu_bias_nhwc(x_nhwc, w_perm, b, bias2, N, IC, IH, IW, OC, OH, OW, KH, KW):
    out_nhwc = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)
    N_OHOW = N * OH * OW

    grid = lambda meta: (
        triton.cdiv(N_OHOW, meta["BLOCK_M"]),
        triton.cdiv(OC, meta["BLOCK_N"]),
    )

    conv2d_relu_bias_nhwc_kernel[grid](
        x_nhwc, w_perm, b, bias2, out_nhwc,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        N_OHOW,
        BLOCK_K=IC,
    )
    return out_nhwc


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._cached_w = None
        self._cached_w_ver = None

    def _get_perm_weight(self):
        w = self.conv.weight
        ver = w._version
        if (self._cached_w is None
                or self._cached_w_ver != ver
                or self._cached_w.device != w.device):
            self._cached_w = w.permute(0, 2, 3, 1).contiguous()
            self._cached_w_ver = ver
        return self._cached_w

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.conv.out_channels
        KH, KW = self.conv.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_perm = self._get_perm_weight()
        b = self.conv.bias.contiguous()
        bias2 = self.bias.view(-1).contiguous()

        out_nhwc = conv2d_relu_bias_nhwc(
            x_nhwc, w_perm, b, bias2,
            N, IC, IH, IW, OC, OH, OW, KH, KW,
        )
        # Permute back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out