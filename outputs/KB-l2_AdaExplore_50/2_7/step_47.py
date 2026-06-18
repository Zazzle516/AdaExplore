import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv3d_configs():
    configs = []
    for bm in [32, 64]:
        for bn in [32, 64]:
            for bk in [16, 32]:
                for nw in [4, 8]:
                    for ns in [2, 3]:
                        configs.append(triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=nw, num_stages=ns,
                        ))
    return configs


@triton.autotune(configs=_conv3d_configs(), key=["OC", "OUT_SPATIAL", "K_TOTAL"])
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    D_out, H_out, W_out,
    OUT_SPATIAL,  # D_out * H_out * W_out
    K_TOTAL,      # IC * KD * KH * KW
    # strides for x: NCDHW
    x_sN, x_sC, x_sD, x_sH, x_sW,
    # strides for w: OC, IC, KD, KH, KW
    w_sO, w_sI, w_sD, w_sH, w_sW,
    # strides for out: NCDHW
    o_sN, o_sC, o_sD, o_sH, o_sW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # over batch N
    pid_m = tl.program_id(1)  # over output spatial tiles (BLOCK_M)
    pid_oc = tl.program_id(2)  # over OC tiles (BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial output positions
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    mask_m = offs_m < OUT_SPATIAL
    mask_n = offs_n < OC

    # decompose output spatial pos -> (od, oh, ow)
    HW_out = H_out * W_out
    od = offs_m // HW_out
    rem = offs_m % HW_out
    oh = rem // W_out
    ow = rem % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW
    KDHW = KD * KHW

    for k0 in range(0, K_TOTAL, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOTAL

        ic = offs_k // KDHW
        krem = offs_k % KDHW
        kd = krem // KHW
        krem2 = krem % KHW
        kh = krem2 // KW
        kw = krem2 % KW

        # input spatial coords
        # id = od + kd, ih = oh + kh, iw = ow + kw  (no padding, stride=1)
        id_ = od[:, None] + kd[None, :]
        ih_ = oh[:, None] + kh[None, :]
        iw_ = ow[:, None] + kw[None, :]

        x_offsets = (
            pid_n * x_sN
            + ic[None, :] * x_sC
            + id_ * x_sD
            + ih_ * x_sH
            + iw_ * x_sW
        )
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        w_offsets = (
            offs_n[None, :] * w_sO
            + ic[:, None] * w_sI
            + kd[:, None] * w_sD
            + kh[:, None] * w_sH
            + kw[:, None] * w_sW
        )
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # add conv bias
    cb = tl.load(cb_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + cb[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # LeakyReLU(0.01) - acc>=0 here so just identity
    # GELU tanh approx
    k0c = 0.7978845608028654
    k1c = 0.044715
    inner = k0c * (acc + k1c * acc * acc * acc)
    e_pos = tl.exp(inner)
    e_neg = tl.exp(-inner)
    tanh_v = (e_pos - e_neg) / (e_pos + e_neg)
    gelu = 0.5 * acc * (1.0 + tanh_v)
    # Sigmoid
    sig = 1.0 / (1.0 + tl.exp(-gelu))
    # Add per-channel bias
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    out = sig + b[None, :]

    # Store to output. Output is NCDHW.
    # out indexed by (pid_n, offs_n[:N tile], offs_m[:M tile])
    out_offs = (
        pid_n * o_sN
        + offs_n[None, :] * o_sC
        + od[:, None] * o_sD
        + oh[:, None] * o_sH
        + ow[:, None] * o_sW
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, out, mask=out_mask)


def fused_conv3d_act_bias(x, weight, conv_bias, extra_bias):
    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    extra_bias_flat = extra_bias.contiguous().view(-1)

    N, IC, D, H, W = x.shape
    OC, IC_w, KD, KH, KW = weight.shape
    assert IC == IC_w

    D_out = D - KD + 1
    H_out = H - KH + 1
    W_out = W - KW + 1

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    OUT_SPATIAL = D_out * H_out * W_out
    K_TOTAL = IC * KD * KH * KW

    grid = lambda meta: (
        N,
        triton.cdiv(OUT_SPATIAL, meta["BLOCK_M"]),
        triton.cdiv(OC, meta["BLOCK_N"]),
    )

    conv3d_fused_kernel[grid](
        x, weight, conv_bias, extra_bias_flat, out,
        N, IC, D, H, W,
        OC, KD, KH, KW,
        D_out, H_out, W_out,
        OUT_SPATIAL, K_TOTAL,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        return fused_conv3d_act_bias(x, self.conv.weight, self.conv.bias, self.bias)