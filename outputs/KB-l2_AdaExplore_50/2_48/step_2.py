import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr,        # (N, D, H, W, IC) channels-last
    w_ptr,        # (OC, kT, kH, kW, IC)
    cb_ptr,       # (OC,) conv bias
    scale_ptr,    # (OC,) scaling_factor
    bias_ptr,     # (OC,) bias param
    out_ptr,      # (N, OC, Do, Ho, Wo) standard layout
    N, IC,
    D, H, W,
    Do, Ho, Wo,
    KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M tile = output spatial points (flattened over N, Do, Ho, Wo)
    pid_m = tl.program_id(0)

    M = N * Do * Ho * Wo
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # decompose offs_m -> (n, do, ho, wo)
    wo_idx = offs_m % Wo
    tmp = offs_m // Wo
    ho_idx = tmp % Ho
    tmp = tmp // Ho
    do_idx = tmp % Do
    n_idx = tmp // Do

    # Output channel range
    offs_n = tl.arange(0, OC)  # full OC (=16)

    acc = tl.zeros((BLOCK_M, OC), dtype=tl.float32)

    K = IC * KT * KH * KW

    # input strides (channels-last): N, D, H, W, IC
    # offset for x = ((n*D + d_in)*H + h_in)*W + w_in)*IC + ic
    # weight (OC, KT, KH, KW, IC): offset = ((((oc*KT + kt)*KH + kh)*KW + kw)*IC + ic
    # iterate over kt, kh, kw, then chunks of IC
    for kt in tl.static_range(0, KT):
        d_in = do_idx + kt
        for kh in tl.static_range(0, KH):
            h_in = ho_idx + kh
            for kw in tl.static_range(0, KW):
                w_in = wo_idx + kw
                base_x = (((n_idx * D + d_in) * H + h_in) * W + w_in) * IC
                base_w = (((tl.arange(0, OC) * KT + kt) * KH + kh) * KW + kw) * IC

                # iterate IC in blocks
                for ic_start in range(0, IC, BLOCK_K):
                    ic_offs = ic_start + tl.arange(0, BLOCK_K)
                    ic_mask = ic_offs < IC

                    x_offs = base_x[:, None] + ic_offs[None, :]
                    x_mask = mask_m[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                    w_offs = base_w[:, None] + ic_offs[None, :]
                    w_mask = ic_mask[None, :]
                    w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)
                    # w_vals: (OC, BLOCK_K). x_vals: (BLOCK_M, BLOCK_K)
                    acc += tl.dot(x_vals, tl.trans(w_vals))

    # epilogue
    cb = tl.load(cb_ptr + offs_n)         # (OC,)
    sc = tl.load(scale_ptr + offs_n)
    bi = tl.load(bias_ptr + offs_n)

    v = (acc + cb[None, :]) * sc[None, :]
    e = tl.exp(2.0 * v)
    t = (e - 1.0) / (e + 1.0)
    y = t * bi[None, :]
    out = 1.0 / (1.0 + tl.exp(-y))

    # store in (N, OC, Do, Ho, Wo) layout
    # offset = ((n*OC + oc)*Do + do)*Ho + ho)*Wo + wo
    spatial = ((n_idx * OC)[:, None] + offs_n[None, :]) * (Do * Ho * Wo) \
              + (do_idx * Ho * Wo + ho_idx * Wo + wo_idx)[:, None]
    store_mask = mask_m[:, None]
    tl.store(out_ptr + spatial, out, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KT = KH = KW = self.kernel_size
        Do = D - KT + 1
        Ho = H - KH + 1
        Wo = W - KW + 1

        # Channels-last input: (N, D, H, W, IC)
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()

        # Weight: (OC, IC, KT, KH, KW) -> (OC, KT, KH, KW, IC)
        w = self.conv.weight.permute(0, 2, 3, 4, 1).contiguous()

        cb = self.conv.bias.contiguous()
        sc = self.scaling_factor.view(-1).contiguous()
        bi = self.bias.view(-1).contiguous()

        out = torch.empty((N, OC, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        M = N * Do * Ho * Wo
        BLOCK_M = 64
        BLOCK_K = 16  # IC=3, so this covers it; needs >=16 for tl.dot
        grid = ((M + BLOCK_M - 1) // BLOCK_M,)

        conv3d_fused_kernel[grid](
            x_cl, w, cb, sc, bi, out,
            N, IC, D, H, W, Do, Ho, Wo,
            KT=KT, KH=KH, KW=KW, OC=OC,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return out