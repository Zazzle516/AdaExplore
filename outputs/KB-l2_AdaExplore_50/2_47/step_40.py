import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    M, K, NN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N * OD * OH * OW (output spatial * batch), reduced by tiling
    # N (gemm-N) = OC
    # K = IC * KD * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output positions (flattened n,od,oh,ow)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels

    # decompose offs_m -> (n_idx, od, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n_idx = tmp // OD

    m_mask = offs_m < M
    n_mask = offs_n < NN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # hoist strides
    stride_xn = IC * ID * IH * IW
    stride_xc = ID * IH * IW
    stride_xd = IH * IW
    stride_on = OC * OD * OH * OW
    stride_oc = OD * OH * OW
    stride_oh = OH * OW

    # iterate over K = IC * KD * KH * KW
    KDHW = KD * KH * KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # decompose k: ic * KDHW + kd*KH*KW + kh*KW + kw
        ic = offs_k // KDHW
        krem = offs_k % KDHW
        kd = krem // (KH * KW)
        krem2 = krem % (KH * KW)
        kh = krem2 // KW
        kw = krem2 % KW

        # input indices: id = od + kd, ih = oh + kh, iw = ow + kw (stride=1, padding=0)
        id_ = od[:, None] + kd[None, :]
        ih_ = oh[:, None] + kh[None, :]
        iw_ = ow[:, None] + kw[None, :]

        # input offset for NCDHW: n_idx*IC*ID*IH*IW + ic*ID*IH*IW + id*IH*IW + ih*IW + iw
        x_offset = (n_idx[:, None] * stride_xn
                    + ic[None, :] * stride_xc
                    + id_ * stride_xd
                    + ih_ * IW
                    + iw_)

        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # weight offset for OC x IC x KD x KH x KW: oc*K + k
        w_offset = offs_n[None, :] * K + offs_k[:, None]
        w_mask = n_mask[None, :] & k_mask[:, None]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x_vals, w_vals)

    # bias
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    mish = acc * th
    # tanh
    e2m = tl.exp(2.0 * mish)
    out = (e2m - 1.0) / (e2m + 1.0)

    # store output to NCDHW
    # output flat index: n_idx*OC*OD*OH*OW + oc*OD*OH*OW + od*OH*OW + oh*OW + ow
    out_offset = (n_idx[:, None] * stride_on
                  + offs_n[None, :] * stride_oc
                  + od[:, None] * stride_oh
                  + oh[:, None] * OW
                  + ow[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offset, out, mask=out_mask)


def conv3d_fused(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OD * OH * OW
    K = IC * KD * KH * KW
    NN = OC

    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(NN, meta['BLOCK_N']))

    conv3d_implicit_gemm_kernel[grid](
        x_c, w_c, b_c, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        M, K, NN,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.stride == 1 and self.padding == 0:
            return conv3d_fused(x, self.conv.weight, self.conv.bias)
        else:
            x = self.conv(x)
            x = F.mish(x)
            x = torch.tanh(x)
            return x