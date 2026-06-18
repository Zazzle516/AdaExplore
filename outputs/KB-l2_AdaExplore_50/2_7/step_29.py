import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
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
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    od = offs_m // (OW * OH)

    mask_m = offs_m < OS
    mask_n = offs_n < OC

    IHW = IH * IW
    IDHW = ID * IHW
    x_base = pid_b * IC * IDHW
    # Precompute spatial base for each output row
    out_spatial_base = od * IHW + oh * IW + ow  # shape [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K_TOT, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOT

        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        kd = (offs_k // (KW * KH)) % KD
        ic = offs_k // (KW * KH * KD)

        # kernel spatial offset
        k_spatial = kd * IHW + kh * IW + kw  # [BLOCK_K]
        ic_off = ic * IDHW  # [BLOCK_K]

        x_offset = x_base + out_spatial_base[:, None] + k_spatial[None, :] + ic_off[None, :]
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        w_offset = (offs_n[None, :] * K_TOT + offs_k[:, None])
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    acc = tl.maximum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    sig = 1.0 / (1.0 + tl.exp(-gelu))
    bias2 = tl.load(bias2_ptr + offs_n, mask=mask_n, other=0.0)
    out_val = sig + bias2[None, :]

    # NDHWC layout: [N, OS, OC]
    out_base = pid_b * OS * OC
    out_offset = out_base + offs_m[:, None] * OC + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, out_val, mask=out_mask)


def conv3d_triton_fused(x, weight, bias, bias2):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    OS = OD * OH * OW
    K_TOT = IC * KD * KH * KW

    out_ndhwc = torch.empty((N, OS, OC), device=x.device, dtype=x.dtype)

    BLOCK_N = max(32, triton.next_power_of_2(OC))

    grid = lambda meta: (triton.cdiv(OS, meta['BLOCK_M']), N)

    conv3d_fused_kernel[grid](
        x, weight, bias, bias2, out_ndhwc,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        OS, K_TOT,
        BLOCK_N=BLOCK_N,
    )
    return out_ndhwc.view(N, OD, OH, OW, OC).permute(0, 4, 1, 2, 3).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-flatten weight for im2col-GEMM layout: [OC, IC*KD*KH*KW]
        with torch.no_grad():
            w = self.conv.weight.detach().contiguous()
            OC = w.shape[0]
            self._weight_flat = nn.Parameter(w.view(OC, -1).contiguous(), requires_grad=False)

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self._weight_flat.contiguous().cuda()
        conv_bias = self.conv.bias.contiguous().cuda()
        bias_flat = self.bias.contiguous().cuda().view(-1)

        # Reshape weight back to 5D shape information by passing flat 2D
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        K_TOT = self.in_channels * KD * KH * KW

        N, IC, ID, IH, IW = x.shape
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        OS = OD * OH * OW
        BLOCK_N = max(32, triton.next_power_of_2(OC))

        # Allocate output in NDHWC layout: [N, OD, OH, OW, OC]
        out_ndhwc = torch.empty((N, OS, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(OS, meta['BLOCK_M']), N)
        conv3d_fused_kernel[grid](
            x, weight, conv_bias, bias_flat, out_ndhwc,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            OS, K_TOT,
            BLOCK_N=BLOCK_N,
        )
        # Permute back to NCDHW
        out = out_ndhwc.view(N, OD, OH, OW, OC).permute(0, 4, 1, 2, 3).contiguous()
        return out