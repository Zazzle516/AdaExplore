import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OS', 'K_TOT'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, bias2_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OS, K_TOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    od = offs_m // (OW * OH)

    mask_m = offs_m < OS

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K_TOT, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOT

        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        kd = (offs_k // (KW * KH)) % KD
        ic = offs_k // (KW * KH * KD)

        id_ = od[:, None] + kd[None, :]
        ih_ = oh[:, None] + kh[None, :]
        iw_ = ow[:, None] + kw[None, :]
        ic_ = ic[None, :]

        x_offset = (pid_b * IC * ID * IH * IW
                    + ic_ * (ID * IH * IW)
                    + id_ * (IH * IW)
                    + ih_ * IW
                    + iw_)
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        w_offset = (offs_n[None, :] * K_TOT + offs_k[:, None])
        w_mask = mask_k[:, None] & (offs_n[None, :] < OC)
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc = acc + bias[None, :]

    acc = tl.maximum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    sig = 1.0 / (1.0 + tl.exp(-gelu))
    bias2 = tl.load(bias2_ptr + offs_n, mask=offs_n < OC, other=0.0)
    out_val = sig + bias2[None, :]

    out_offset = (pid_b * OC * OS
                  + offs_n[None, :] * OS
                  + offs_m[:, None])
    out_mask = mask_m[:, None] & (offs_n[None, :] < OC)
    tl.store(out_ptr + out_offset, out_val, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        with torch.no_grad():
            w = self.conv.weight.detach().contiguous()
            OC = w.shape[0]
            w_flat = w.view(OC, -1).contiguous().cuda()
            self._weight_flat = nn.Parameter(w_flat, requires_grad=False)
            cb = self.conv.bias.detach().contiguous().cuda()
            self._conv_bias = nn.Parameter(cb, requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        weight = self._weight_flat
        conv_bias = self._conv_bias
        bias_flat = self.bias.view(-1).contiguous()
        if not bias_flat.is_cuda:
            bias_flat = bias_flat.cuda()

        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        K_TOT = self.in_channels * KD * KH * KW

        N, IC, ID, IH, IW = x.shape
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        OS = OD * OH * OW
        BLOCK_N = 32  # OC == 32

        grid = lambda meta: (N, triton.cdiv(OS, meta['BLOCK_M']))
        conv3d_fused_kernel[grid](
            x, weight, conv_bias, bias_flat, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            OS, K_TOT,
            BLOCK_N=BLOCK_N,
        )
        return out