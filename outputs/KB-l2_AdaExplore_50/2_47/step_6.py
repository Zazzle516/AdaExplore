import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    M, K_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decompose offs_m -> (n, od, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    od = tmp2 % OD
    n = tmp2 // OD

    m_mask = offs_m < M
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K is laid out as (kd, kh, kw, ic) with stride: ic fastest
    # Total K = KD*KH*KW*IC
    # We iterate over k in BLOCK_K chunks
    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K_total, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K_total

        ic = k_idx % IC
        t = k_idx // IC
        kw = t % KW
        t2 = t // KW
        kh = t2 % KH
        kd = t2 // KH

        # input spatial coords
        id_ = od[:, None] + kd[None, :]  # [BLOCK_M, BLOCK_K]
        ih_ = oh[:, None] + kh[None, :]
        iw_ = ow[:, None] + kw[None, :]
        # No padding (padding=0 default)
        in_bounds = (id_ < D) & (ih_ < H) & (iw_ < W)

        # x is NCDHW: offset = ((n*IC + ic)*D + id)*H*W + ih*W + iw
        x_off = (((n[:, None] * IC + ic[None, :]) * D + id_) * H + ih_) * W + iw_
        x_load_mask = m_mask[:, None] & k_mask[None, :] & in_bounds
        x_vals = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)

        # weight: (OC, IC, KD, KH, KW). We want W[oc, ic, kd, kh, kw]
        # offset = ((((oc*IC + ic)*KD + kd)*KH + kh)*KW + kw)
        # shape [BLOCK_K, BLOCK_N]
        w_off = (((offs_n[None, :] * IC + ic[:, None]) * KD + kd[:, None]) * KH + kh[:, None]) * KW + kw[:, None]
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # Mish: x * tanh(softplus(x))
    # softplus = log(1+exp(x)); tanh(sp) = (e^{2sp}-1)/(e^{2sp}+1)
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    mish_val = acc * th
    # tanh
    e2b = tl.exp(2.0 * mish_val)
    out_val = (e2b - 1.0) / (e2b + 1.0)

    # Store: out is NCDHW with shape (N, OC, OD, OH, OW)
    # offset = ((n*OC + oc)*OD + od)*OH*OW + oh*OW + ow
    out_off = (((n[:, None] * OC + offs_n[None, :]) * OD + od[:, None]) * OH + oh[:, None]) * OW + ow[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, out_val, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kD = self.kH = self.kW = kernel_size
        else:
            self.kD, self.kH, self.kW = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Fall back to torch if stride!=1 or padding!=0
        if self.stride != 1 or self.padding != 0:
            y = self.conv(x)
            sp = torch.nn.functional.softplus(y)
            y = y * torch.tanh(sp)
            return torch.tanh(y)

        x = x.contiguous()
        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kD, self.kH, self.kW
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        M = N * OD * OH * OW
        K_total = IC * KD * KH * KW

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(OC, BLOCK_N))
        conv3d_implicit_gemm_kernel[grid](
            x, weight, bias, out,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            KD, KH, KW,
            M, K_total,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return out