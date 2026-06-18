import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    x_ptr,           # [N, IC, S_in]
    wsum_ptr,        # [IC, OC]  precomputed sum_k w[ic, oc, k]
    eff_w_ptr,       # [OC]
    eff_b_ptr,       # [OC]
    out_ptr,         # [N, OC]
    N, IC, OC, S_in,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    offs_s = tl.arange(0, BLOCK_S)

    for ic_start in range(0, IC, BLOCK_IC):
        offs_ic = ic_start + tl.arange(0, BLOCK_IC)
        mask_ic = offs_ic < IC

        # Load wsum[ic, oc] -> [BLOCK_IC, BLOCK_OC]
        w_off = offs_ic[:, None] * OC + offs_oc[None, :]
        w_mask = mask_ic[:, None] & mask_oc[None, :]
        w_tile = tl.load(wsum_ptr + w_off, mask=w_mask, other=0.0)

        # x sum over spatial per ic: x_sum[ic]
        x_sum = tl.zeros([BLOCK_IC], dtype=tl.float32)
        for s_start in range(0, S_in, BLOCK_S):
            offs_p = s_start + offs_s
            mask_p = offs_p < S_in
            x_off = pid_n * (IC * S_in) + offs_ic[:, None] * S_in + offs_p[None, :]
            x_mask = mask_ic[:, None] & mask_p[None, :]
            x_block = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
            x_sum += tl.sum(x_block, axis=1)

        # acc[oc] += sum_ic x_sum[ic] * w_tile[ic, oc]
        acc += tl.sum(x_sum[:, None] * w_tile, axis=0)

    ew = tl.load(eff_w_ptr + offs_oc, mask=mask_oc, other=0.0)
    eb = tl.load(eff_b_ptr + offs_oc, mask=mask_oc, other=0.0)
    out_val = acc * ew + eb
    out_off = pid_n * OC + offs_oc
    tl.store(out_ptr + out_off, out_val, mask=mask_oc)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        if self.training:
            x = self.conv_transpose(x)
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x

        x = x.contiguous()
        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = D + KD - 1
        OH = H + KH - 1
        OW = W + KW - 1

        S_in = D * H * W
        S_out = OD * OH * OW

        s = self.scale_factor
        rstd = torch.rsqrt(self.batch_norm.running_var + self.eps)
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        mean = self.batch_norm.running_mean
        conv_bias = self.conv_transpose.bias

        eff_w = (s * gamma * rstd / S_out).contiguous()
        eff_b = (beta - mean * gamma * rstd + s * gamma * rstd * conv_bias).contiguous()

        # Precompute wsum[ic, oc] = sum_k weight[ic, oc, k]
        weight = self.conv_transpose.weight  # [IC, OC, KD, KH, KW]
        wsum = weight.sum(dim=(2, 3, 4)).contiguous()  # [IC, OC]

        out = torch.empty((N, OC), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_IC = 16
        BLOCK_S = 512

        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC)
        _fused_kernel[grid](
            x, wsum, eff_w, eff_b, out,
            N, IC, OC, S_in,
            BLOCK_OC=BLOCK_OC,
            BLOCK_IC=BLOCK_IC,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )
        return out.view(N, OC, 1, 1, 1)