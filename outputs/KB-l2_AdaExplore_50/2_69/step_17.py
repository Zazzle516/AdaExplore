import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_im2col_gemm_kernel(
    x_ptr,       # NHWC input: [B, IH, IW, IC]
    w_ptr,       # weight: [OC, K] where K = IC*KH*KW (row-major)
    b_ptr,       # bias: [OC]
    out_ptr,     # NHWC output: [B, OH, OW, OC]
    B, IH, IW, IC,
    OH, OW, OC,
    M, N, K,    # M = B*OH*OW, N = OC, K = IC*KH*KW
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row idx in M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # col idx (oc)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # decode m -> (b, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    b_idx = tmp // OH

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = KH*KW*IC. Iterate over (kh, kw), then dot over IC block.
    # Weight row layout: w[oc, ic, kh, kw] -> we'll use w[oc, kh, kw, ic] for contiguous IC
    # So K is laid out as (kh, kw, ic) contiguous in ic.
    ic_range = tl.arange(0, IC_C)  # [IC_C]

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # [BLOCK_M]
            iw = ow + kw  # [BLOCK_M]
            # x NHWC offsets: ((b*IH + ih)*IW + iw)*IC + ic
            x_base = ((b_idx * IH + ih) * IW + iw) * IC  # [BLOCK_M]
            x_off = x_base[:, None] + ic_range[None, :]  # [BLOCK_M, IC_C]
            x_mask = mask_m[:, None] & (ic_range[None, :] < IC)
            x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_M, IC_C]

            # weight layout: w[oc, kh, kw, ic] -> contiguous
            # offset = oc * K + (kh*KW + kw)*IC + ic
            w_base = offs_n * K + (kh * KW + kw) * IC  # [BLOCK_N]
            w_off = w_base[:, None] + ic_range[None, :]  # [BLOCK_N, IC_C]
            w_mask = mask_n[:, None] & (ic_range[None, :] < IC)
            w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_N, IC_C]

            # acc += x_vals @ w_vals.T => [BLOCK_M, BLOCK_N]
            acc += tl.dot(x_vals, tl.trans(w_vals))

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # relu(hardswish(x))
    relu6 = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    hs = acc * relu6 * (1.0 / 6.0)
    out_val = tl.maximum(hs, 0.0)

    # output NHWC: ((b*OH + oh)*OW + ow)*OC + oc
    out_row = ((b_idx * OH + oh) * OW + ow) * OC  # [BLOCK_M]
    out_off = out_row[:, None] + offs_n[None, :]  # [BLOCK_M, BLOCK_N]
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to [OC, KH, KW, IC] layout (contiguous in IC)
        with torch.no_grad():
            w = self.conv.weight.detach().cuda().contiguous()  # [OC, IC, KH, KW]
            # permute -> [OC, KH, KW, IC]
            w_perm = w.permute(0, 2, 3, 1).contiguous()
            OC = w_perm.shape[0]
            self._w_packed = w_perm.view(OC, -1).contiguous()
            self._b_packed = self.conv.bias.detach().cuda().contiguous()

        # round up IC to nearest power-of-two for block sizing
        ic_pow2 = 1
        while ic_pow2 < in_channels:
            ic_pow2 *= 2
        self._ic_c = max(ic_pow2, 8)  # min 8 for tl.dot

    def forward(self, x):
        x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()

        B, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [B, IH, IW, IC]

        # Output NHWC
        out_nhwc = torch.empty((B, OH, OW, OC), device=x.device, dtype=x.dtype)

        M = B * OH * OW
        N = OC
        K = IC * KH * KW

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        conv_im2col_gemm_kernel[grid](
            x_nhwc, self._w_packed, self._b_packed, out_nhwc,
            B, IH, IW, IC,
            OH, OW, OC,
            M, N, K,
            KH, KW,
            self._ic_c,
        )

        # convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out