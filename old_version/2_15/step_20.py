import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    SPATIAL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program over (n, oc_block, spatial_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_start = pid_oc * BLOCK_N
    offs_oc = oc_start + tl.arange(0, BLOCK_N)
    mask_oc = offs_oc < OC

    sp_start = pid_sp * BLOCK_M
    offs_sp = sp_start + tl.arange(0, BLOCK_M)
    mask_sp = offs_sp < SPATIAL

    od = offs_sp // (OH * OW)
    rem = offs_sp % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # load bias for these oc
    bias_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n = pid_n

    ic_stride = ID * IH * IW
    w_ic_stride = OC * KD * KH * KW
    offs_k = tl.arange(0, BLOCK_K)

    for kd in range(0, KD):
        id_num = od + PD - kd  # [BLOCK_M]
        id_val = id_num // SD
        id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_val < ID) & (id_val >= 0)
        for kh in range(0, KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val < IH) & (ih_val >= 0)
            for kw in range(0, KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val < IW) & (iw_val >= 0)

                spatial_valid = id_valid & ih_valid & iw_valid & mask_sp  # [BLOCK_M]

                x_base = (n * IC) * ic_stride + id_val * (IH * IW) + ih_val * IW + iw_val  # [BLOCK_M]
                w_base_oc = (offs_oc * KD + kd) * (KH * KW) + kh * KW + kw  # [BLOCK_N]

                for k0 in range(0, IC, BLOCK_K):
                    k_curr = k0 + offs_k  # [BLOCK_K]
                    k_mask = k_curr < IC
                    # x tile [BLOCK_M, BLOCK_K]
                    x_off = x_base[:, None] + k_curr[None, :] * ic_stride
                    x_mask = spatial_valid[:, None] & k_mask[None, :]
                    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
                    # w tile [BLOCK_K, BLOCK_N]
                    w_off = w_base_oc[None, :] + k_curr[:, None] * w_ic_stride
                    w_mask = k_mask[:, None] & mask_oc[None, :]
                    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)
                    acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    y = acc + bias_vals[None, :]

    # store output: out[n, oc, sp] -- layout (N, OC, SPATIAL)
    out_off = (n * OC + offs_oc[None, :]) * SPATIAL + offs_sp[:, None]
    full_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, y, mask=full_mask)


@triton.jit
def subtract_mean_kernel(
    x_ptr, sum_ptr,
    N, C, SPATIAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    pid_sp = tl.program_id(1)

    n = pid // C
    c = pid % C

    s = tl.load(sum_ptr + n * C + c)
    mean = s / SPATIAL

    base = (n * C + c) * SPATIAL
    offs = pid_sp * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SPATIAL
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    tl.store(x_ptr + base + offs, x - mean, mask=mask)


def fused_convtr_bn_meansub(x, weight, bias, scale, shift, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW
    SPATIAL = OD * OH * OW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)
    sums = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 32

    grid = (N, (OC + BLOCK_N - 1) // BLOCK_N, (SPATIAL + BLOCK_M - 1) // BLOCK_M)

    conv_transpose3d_gemm_kernel[grid](
        x, weight, bias, out, sums,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        SPATIAL,
        scale, shift,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )

    BLOCK = 1024
    grid2 = (N * OC, (SPATIAL + BLOCK - 1) // BLOCK)
    subtract_mean_kernel[grid2](
        out, sums, N, OC, SPATIAL, BLOCK=BLOCK, num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        if self.conv_transpose.bias is not None:
            conv_bias = self.conv_transpose.bias.contiguous()
        else:
            conv_bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        # Update BN running stats to mimic training-mode behavior:
        # We need to compute the conv output to get batch stats. But for the purposes of the
        # forward pass, BN in training subtracts batch mean and divides by batch std.
        # After BN(y) and then subtracting spatial mean per (n,c), the result is invariant
        # to the BN affine and to the batch shift; but the BN scale by 1/std(batch) DOES
        # affect per-sample spatial mean subtraction.
        # We must compute conv first, then BN, then mean-sub.
        # To stay fast, compute conv via custom kernel without folding BN, then call BN, 
        # then subtract spatial mean.
        # But we still want to fuse BN-affine + mean subtraction with the conv.
        # In training, BN computes mean/var over (N,D,H,W) per C. We compute the conv output
        # first (write to memory), reduce to get per-channel mean/var, then compute scale/shift,
        # then re-launch fused kernel? That's expensive.
        # Simpler: compute conv with our kernel (write y), then PyTorch BN, then triton mean-sub.

        # Use the fused kernel only with identity BN affine, then do real BN, then mean-sub.
        OC = self.out_channels
        device = x.device

        # Compute conv-transpose only with identity scale=1, shift=0 (we'll do BN after)
        # then PyTorch BN, then mean subtraction via separate kernel.
        # But that requires a sum-reduction kernel post-BN.

        # We'll do: custom conv (identity affine in fused kernel, sums collected post-BN doesn't help)
        # Instead, simpler path:
        #   y = custom conv (no BN fold)
        #   y = self.batch_norm(y) (PyTorch)
        #   subtract spatial mean per (n,c) via custom kernel

        N, IC, ID, IH, IW = x.shape
        _, _, KD, KH, KW = weight.shape
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW
        SPATIAL = OD * OH * OW

        out = torch.empty((N, OC, OD, OH, OW), device=device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 16
        grid = (N, (OC + BLOCK_N - 1) // BLOCK_N, (SPATIAL + BLOCK_M - 1) // BLOCK_M)

        conv_transpose3d_gemm_kernel[grid](
            x, weight, conv_bias, out,
            N, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            SPATIAL,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # PyTorch BN (training mode) - matches the reference
        y = self.batch_norm(out)

        # Now subtract spatial mean per (n,c)
        sums2 = torch.zeros((N, OC), device=device, dtype=torch.float32)
        # compute sums via reduction
        sums2 = y.sum(dim=(2, 3, 4))  # [N, OC]
        BLOCK = 1024
        grid2 = (N * OC, (SPATIAL + BLOCK - 1) // BLOCK)
        subtract_mean_kernel[grid2](
            y, sums2.contiguous(), N, OC, SPATIAL, BLOCK=BLOCK, num_warps=4,
        )
        return y