import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


def _conv_configs():
    configs = []
    base = [
        (32, 64, 16, 2), (64, 64, 16, 2), (64, 64, 32, 2),
        (32, 128, 16, 2), (64, 128, 16, 2), (64, 128, 32, 2),
        (128, 64, 16, 2), (128, 64, 32, 2),
        (32, 256, 16, 2), (16, 256, 16, 2),
        (32, 128, 32, 2), (32, 256, 32, 2),
        (64, 256, 16, 2), (64, 256, 16, 3),
        (128, 128, 16, 2), (128, 128, 16, 3),
        (32, 128, 32, 3), (64, 128, 32, 3),
        (32, 256, 32, 3),
    ]
    for bm, bn, bk, ns in base:
        for nw in [4, 8]:
            configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SP": 8}, num_warps=nw, num_stages=ns))
    return configs


@triton.autotune(configs=_conv_configs(), key=["OC", "OUT_SP", "K_TOTAL"])
@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr,        # (N, IC, ID, IH, IW)
    w_ptr,        # (OC, IC*KD*KH*KW)  - packed
    conv_bias_ptr,  # (OC,)
    add_bias_ptr,   # (OC,)
    out_ptr,      # (N, OC, OD, OH, OW)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    OUT_SP,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SP: tl.constexpr,
):
    pid_sp_raw = tl.program_id(0)    # over output spatial tiles
    pid_n_oc = tl.program_id(1)      # over (N * OC_tiles)

    num_sp_tiles = tl.cdiv(OUT_SP, BLOCK_N)
    num_oc_tiles = tl.cdiv(OC, BLOCK_M)

    # L2-friendly swizzle: group consecutive sp tiles to share weight loads
    group_size = GROUP_SP
    num_groups = tl.cdiv(num_sp_tiles, group_size)
    group_id = pid_sp_raw // group_size
    in_group = pid_sp_raw % group_size
    pid_sp = group_id * group_size + in_group  # identity but keeps explicit

    n_idx = pid_n_oc // num_oc_tiles
    oc_tile = pid_n_oc % num_oc_tiles

    oc_offs = oc_tile * BLOCK_M + tl.arange(0, BLOCK_M)  # OC dim
    sp_offs = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)   # output spatial dim

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OUT_SP

    # decode spatial -> (od, oh, ow) and precompute base input offset
    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    IHIW = IH * IW
    IDIHIW = ID * IHIW
    # base offset within an (IC=ic) slice, for each output spatial position
    base_sp = od * IHIW + oh * IW + ow  # (BLOCK_N,)
    x_n_base = n_idx * IC * IDIHIW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHKW = KH * KW
    KDKHKW = KD * KH * KW

    # Loop over K = IC * KD * KH * KW
    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_TOTAL

        # decode k -> (ic, kd, kh, kw)
        ic = k_offs // KDKHKW
        krem = k_offs % KDKHKW
        kd = krem // KHKW
        krem2 = krem % KHKW
        kh = krem2 // KW
        kw = krem2 % KW

        # kernel offset within an (ic) slice
        k_spatial = kd * IHIW + kh * IW + kw  # (BLOCK_K,)
        ic_off = ic * IDIHIW  # (BLOCK_K,)

        # x ptr offsets
        x_off = (x_n_base
                 + (ic_off + k_spatial)[:, None]
                 + base_sp[None, :])

        x_load_mask = (k_mask[:, None] & sp_mask[None, :])
        x_tile = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)  # (BK, BN)

        # weight: (OC, K_TOTAL) -> load (BM, BK)
        w_off = oc_offs[:, None] * K_TOTAL + k_offs[None, :]
        w_load_mask = oc_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)  # (BM, BK)

        acc += tl.dot(w_tile, x_tile)

    # add conv bias
    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + cb[:, None]

    # epilogue: ReLU -> LeakyReLU(0.01) -> GELU -> Sigmoid -> add bias
    # ReLU
    y = tl.maximum(acc, 0.0)
    # LeakyReLU with negative_slope=0.01: y>=0 so no-op, but cheap
    # y = tl.where(y >= 0.0, y, y * 0.01)  # skip - mathematically identical since y>=0
    # GELU exact: 0.5 * y * (1 + erf(y / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    # Sigmoid
    y = tl.sigmoid(y)
    # add bias (per-OC)
    ab = tl.load(add_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    y = y + ab[:, None]

    # write out: out shape (N, OC, OD, OH, OW) row-major
    out_off = (n_idx * OC * OUT_SP
               + oc_offs[:, None] * OUT_SP
               + sp_offs[None, :])
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, y, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size

        # Create reference Conv3d to get matching init
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # weight shape: (OC, IC, KD, KH, KW); pack to (OC, IC*KD*KH*KW)
        w = conv.weight.detach().contiguous()
        self.weight_packed = nn.Parameter(w.view(out_channels, -1).contiguous())
        self.conv_bias = nn.Parameter(conv.bias.detach().contiguous())

        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kd, self.kh, self.kw
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        OUT_SP = OD * OH * OW
        K_TOTAL = IC * KD * KH * KW

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        add_bias_flat = self.bias.contiguous().view(-1)

        grid = lambda META: (
            triton.cdiv(OUT_SP, META["BLOCK_N"]),
            N * triton.cdiv(OC, META["BLOCK_M"]),
        )

        conv3d_implicit_gemm_kernel[grid](
            x, self.weight_packed, self.conv_bias, add_bias_flat, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            K_TOTAL, OUT_SP,
        )
        return out