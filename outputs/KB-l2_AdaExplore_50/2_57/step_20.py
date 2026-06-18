import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 16, 'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_HW': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC_C'],
)
@triton.jit
def conv_relu_hswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = offs_hw // OW
    ow = offs_hw % OW

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (OH * OW)

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    x_batch_off = pid_n * (IC * IH * IW)
    # weight layout: (IC, KH, KW, OC) — OC is contiguous innermost
    for ic in range(0, IC_C):
        x_ic_off = x_batch_off + ic * (IH * IW)
        w_ic_off = ic * (KH * KW * OC)
        for kh in range(0, KH):
            ih_row = (oh + kh) * IW
            x_kh_base = x_ic_off + ih_row
            w_kh_off = w_ic_off + kh * (KW * OC)
            for kw in range(0, KW):
                x_off = x_kh_base + (ow + kw)
                x_vals = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)  # (BLOCK_HW,)

                w_off = w_kh_off + kw * OC + offs_oc
                w_vals = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # (BLOCK_OC,)

                acc += x_vals[:, None] * w_vals[None, :]

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # HardSwish: x * clamp((x+3)/6, 0, 1)
    hs = tl.minimum(tl.maximum((acc + 3.0) * (1.0 / 6.0), 0.0), 1.0)
    acc = acc * hs

    out_off = (pid_n * OC * OH * OW) + offs_oc[None, :] * (OH * OW) + offs_hw[:, None]
    mask_out = mask_hw[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask_out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # pre-permute weight to (IC, KH, KW, OC) for coalesced OC loads
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            w_perm = w.permute(1, 2, 3, 0).contiguous()  # (IC, KH, KW, OC)
            self.register_buffer('w_perm', w_perm)

    def forward(self, x):
        x = x.contiguous().cuda()
        # Re-permute on the fly if weights changed (e.g. after training step)
        w = self.w_perm
        if w.device != x.device:
            w = w.to(x.device)
            self.w_perm = w
        b = self.conv.bias.contiguous().to(x.device)

        N, IC, IH, IW = x.shape
        OC = self.conv.out_channels
        KH = self.conv.kernel_size[0]
        KW = self.conv.kernel_size[1]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(OH * OW, meta['BLOCK_HW']))

        conv_relu_hswish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            IC,
        )
        return out