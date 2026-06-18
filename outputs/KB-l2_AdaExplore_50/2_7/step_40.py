import torch
import torch.nn as nn
import triton
import triton.language as tl


def _conv_configs():
    configs = []
    for bm, bn, bk in [
        (32, 64, 32), (32, 128, 32), (32, 128, 64), (32, 256, 32),
        (32, 64, 64), (16, 128, 64), (16, 256, 32), (16, 256, 64),
        (32, 256, 64), (32, 64, 16), (64, 64, 32), (64, 128, 32),
        (32, 128, 16), (16, 128, 32),
    ]:
        for nw in [4, 8]:
            for ns in [2, 3]:
                configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns))
    return configs


@triton.autotune(configs=_conv_configs(), key=["OC", "OUT_SP", "K_TOTAL"])
@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr,
    w_ptr,
    conv_bias_ptr,
    add_bias_ptr,
    out_ptr,
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
    pid_sp_raw = tl.program_id(0)
    pid_n_oc = tl.program_id(1)

    num_oc_tiles = tl.cdiv(OC, BLOCK_M)

    group_id = pid_sp_raw // GROUP_SP
    in_group = pid_sp_raw % GROUP_SP
    pid_sp = group_id * GROUP_SP + in_group

    n_idx = pid_n_oc // num_oc_tiles
    oc_tile = pid_n_oc % num_oc_tiles

    oc_offs = oc_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    sp_offs = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OUT_SP

    OHOW = OH * OW
    IHIW = IH * IW
    od = sp_offs // OHOW
    rem = sp_offs % OHOW
    oh = rem // OW
    ow = rem % OW
    base_in = od * IHIW + oh * IW + ow
    n_off = n_idx * IC * ID * IHIW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHKW = KH * KW
    KDKHKW = KD * KH * KW
    IC_STRIDE = ID * IHIW

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_TOTAL

        ic = k_offs // KDKHKW
        krem = k_offs % KDKHKW
        kd = krem // KHKW
        krem2 = krem % KHKW
        kh = krem2 // KW
        kw = krem2 % KW

        k_off_in = ic * IC_STRIDE + kd * IHIW + kh * IW + kw

        x_off = n_off + k_off_in[:, None] + base_in[None, :]
        x_load_mask = k_mask[:, None] & sp_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)

        w_off = oc_offs[:, None] * K_TOTAL + k_offs[None, :]
        w_load_mask = oc_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile)

    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + cb[:, None]

    y = tl.maximum(acc, 0.0)
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    y = tl.sigmoid(y)
    ab = tl.load(add_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    y = y + ab[:, None]

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

        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
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
            GROUP_SP=8,
        )
        return out