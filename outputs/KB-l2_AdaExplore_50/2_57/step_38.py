import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'KTOTAL'],
)
@triton.jit
def conv_relu_hswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    KTOTAL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    # K dimension is over (kh, kw, ic). We pack weight as (KH*KW*IC, OC) contiguous.
    # We'll iterate over K in chunks of BLOCK_K.
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    x_batch_off = pid_n * (IC * IH * IW)

    for k_start in range(0, KTOTAL, BLOCK_K):
        k_idx = k_start + offs_k  # (BLOCK_K,)
        mask_k = k_idx < KTOTAL

        # Decompose k into (kh, kw, ic) where layout is kh-major: k = ((kh*KW)+kw)*IC+ic
        ic_k = k_idx % IC
        khw = k_idx // IC
        kw_k = khw % KW
        kh_k = khw // KW

        # ih = oh + kh, iw = ow + kw, for each (hw, k)
        # x_off shape: (BLOCK_HW, BLOCK_K)
        ih = oh[:, None] + kh_k[None, :]
        iw = ow[:, None] + kw_k[None, :]
        x_off = x_batch_off + ic_k[None, :] * (IH * IW) + ih * IW + iw
        x_mask = mask_hw[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # (BLOCK_HW, BLOCK_K)

        # weight (K, OC): w_ptr offset = k_idx[:, None] * OC + offs_oc[None, :]
        w_off = k_idx[:, None] * OC + offs_oc[None, :]
        w_mask = mask_k[:, None] & mask_oc[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # (BLOCK_K, BLOCK_OC)

        acc += tl.dot(x_tile, w_tile)

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)  # (BLOCK_OC,)
    acc = acc + b_vals[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # HardSwish on relu(x): x * clamp((x+3)/6, 0, 1)
    hs = tl.minimum(tl.maximum((acc + 3.0) * (1.0 / 6.0), 0.0), 1.0)
    acc = acc * hs

    # out is (N, OC, OH, OW)
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
        # Pre-pack weight as (KH*KW*IC, OC) once
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            OC, IC, KH, KW = w.shape
            # Want layout [kh, kw, ic, oc] so K = ((kh*KW)+kw)*IC+ic
            w_packed = w.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)
            self.register_buffer('w_packed', w_packed.cuda())
            self.register_buffer('b', self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        KTOTAL = KH * KW * IC

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        # Pick BLOCK_K: prefer 32 (KTOTAL=72 -> 3 chunks of 32 with masking)
        BLOCK_K = 32

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(OH * OW, meta['BLOCK_HW']),
        )

        conv_relu_hswish_kernel[grid](
            x, self.w_packed, self.b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            KTOTAL,
            BLOCK_K=BLOCK_K,
        )
        return out