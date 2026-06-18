import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_OC': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'IC', 'H', 'W', 'KH', 'KW'],
)
@triton.jit
def conv_hardswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    # x is NHWC: (N, H, W, IC) contiguous
    # w is (OC, KH, KW, IC) contiguous
    # out is NHWC: (N, OH, OW, OC) contiguous
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, BLOCK_IC)

    NHW = N * OH * OW
    mask_n = offs_n < NHW
    mask_oc = offs_oc < OC
    mask_ic = offs_ic < IC

    # decode n, oh, ow from offs_n
    n_idx = offs_n // (OH * OW)
    rem = offs_n % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    # x base: n*H*W*IC + oh*W*IC + ow*IC  (input pixel ih=oh+kh, iw=ow+kw)
    x_base = n_idx * (H * W * IC) + oh_idx * (W * IC) + ow_idx * IC  # [BLOCK_N]
    # w base: oc*KH*KW*IC
    w_base = offs_oc * (KH * KW * IC)  # [BLOCK_OC]

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # x address: [BLOCK_N, BLOCK_IC]
            x_off = x_base[:, None] + (kh * W * IC + kw * IC) + offs_ic[None, :]
            x_mask = mask_n[:, None] & mask_ic[None, :]
            x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
            # w address: [BLOCK_OC, BLOCK_IC]
            w_off = w_base[:, None] + (kh * KW * IC + kw * IC) + offs_ic[None, :]
            w_mask = mask_oc[:, None] & mask_ic[None, :]
            w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)
            # GEMM: [BLOCK_N, BLOCK_IC] x [BLOCK_IC, BLOCK_OC]
            acc += tl.dot(x_vals, tl.trans(w_vals), allow_tf32=True)

    # bias
    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[None, :]

    # fused hardswish + relu: when x>0, hardswish>=0 so outer relu is redundant
    # output = max(min(x+3, 6), 0) * x / 6
    out = tl.maximum(tl.minimum(acc + 3.0, 6.0), 0.0) * acc * (1.0 / 6.0)

    # store NHWC: out[n, oh, ow, oc]
    out_off = (n_idx * (OH * OW * OC))[:, None] + (oh_idx * (OW * OC))[:, None] + (ow_idx * OC)[:, None] + offs_oc[None, :]
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Pre-transpose weight to (OC, KH, KW, IC)
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()
        self.register_buffer('weight_nhwc', w_nhwc)
        self.block_ic = max(16, _next_pow2(in_channels))

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.conv.weight
        bias = self.conv.bias

        N, IC, H, W = x.shape
        OC, _, KH, KW = weight.shape
        OH = H - KH + 1
        OW = W - KW + 1

        # NHWC layout
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        # Ensure weight_nhwc on correct device
        if self.weight_nhwc.device != x.device:
            self.weight_nhwc = self.weight_nhwc.to(x.device)
        # If weight was updated (training), re-permute
        w_nhwc = weight.permute(0, 2, 3, 1).contiguous() if self.training else self.weight_nhwc

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(N * OH * OW, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv_hardswish_relu_kernel[grid](
            x_nhwc, w_nhwc, bias.contiguous(), out_nhwc,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
            self.block_ic,
        )
        # back to NCHW
        return out_nhwc.permute(0, 3, 1, 2).contiguous()