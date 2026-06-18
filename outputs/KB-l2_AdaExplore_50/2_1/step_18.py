import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N_OHOW", "OC", "IC", "KH", "KW"])
@triton.jit
def conv2d_relu_bias_nhwc_kernel(
    x_ptr,        # NHWC: [N, IH, IW, IC]
    w_ptr,        # [KH, KW, IC, OC]  permuted
    b_ptr,        # [OC]
    bias2_ptr,    # [OC]
    out_ptr,      # NCHW: [N, OC, OH, OW]
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

    # weight is laid out as [KH, KW, IC, OC], stride along IC = OC, stride along OC = 1
    # x is NHWC [N, IH, IW, IC], stride along IC = 1
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            x_row_base = (n_idx * IH * IW + ih * IW + iw) * IC  # [BLOCK_M]
            w_base = (kh * KW + kw) * IC * OC  # scalar

            for ic_start in range(0, IC, BLOCK_K):
                k_idx = ic_start + offs_k
                k_mask = k_idx < IC

                # x: [BLOCK_M, BLOCK_K]
                x_offs = x_row_base[:, None] + k_idx[None, :]
                x_vals = tl.load(
                    x_ptr + x_offs,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )

                # w: [BLOCK_K, BLOCK_N]
                w_offs = w_base + k_idx[:, None] * OC + offs_n[None, :]
                w_vals = tl.load(
                    w_ptr + w_offs,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

                acc += tl.dot(x_vals, w_vals)

    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    bias2_vals = tl.load(bias2_ptr + offs_n, mask=n_mask, other=0.0)

    acc = acc + b_vals[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    acc = acc + bias2_vals[None, :]

    spatial_off = oh * OW + ow
    n_off = n_idx * (OC * OH * OW)
    oc_off = offs_n * (OH * OW)
    out_offs = (n_off + spatial_off)[:, None] + oc_off[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv2d_relu_bias_nhwc(x_nhwc, w_perm, b, bias2, N, IC, IH, IW, OC, OH, OW, KH, KW):
    out_nchw = torch.empty((N, OC, OH, OW), device=x_nhwc.device, dtype=x_nhwc.dtype)
    N_OHOW = N * OH * OW

    grid = lambda meta: (
        triton.cdiv(N_OHOW, meta["BLOCK_M"]),
        triton.cdiv(OC, meta["BLOCK_N"]),
    )

    conv2d_relu_bias_nhwc_kernel[grid](
        x_nhwc, w_perm, b, bias2, out_nchw,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        N_OHOW,
    )
    return out_nchw


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._cached_w = None
        self._cached_w_ptr = None

    def _get_perm_weight(self):
        w = self.conv.weight
        if (self._cached_w is None
                or self._cached_w_ptr != w.data_ptr()
                or self._cached_w.device != w.device
                or self._cached_w.dtype != w.dtype):
            # [OC, IC, KH, KW] -> [KH, KW, IC, OC]
            self._cached_w = w.permute(2, 3, 1, 0).contiguous()
            self._cached_w_ptr = w.data_ptr()
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

        out = conv2d_relu_bias_nhwc(
            x_nhwc, w_perm, b, bias2,
            N, IC, IH, IW, OC, OH, OW, KH, KW,
        )
        return out