import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    PH, PW,
    stride_xn, stride_xh, stride_xw,  # NHWC strides for input (C is contiguous)
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
    IC_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < (PH * PW)
    ph = p_offs // PW
    pw = p_offs % PW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    inv_pool2 = 1.0 / (POOL * POOL).to(tl.float32)

    pooled_acc = tl.zeros([BLOCK_OC, BLOCK_P], dtype=tl.float32)

    ic_range = tl.arange(0, IC_C)  # [IC]

    for dy in tl.static_range(0, POOL):
        for dx in tl.static_range(0, POOL):
            oh = ph * POOL + dy   # [BLOCK_P]
            ow = pw * POOL + dx   # [BLOCK_P]

            conv_val = tl.zeros([BLOCK_OC, BLOCK_P], dtype=tl.float32) + bias[:, None]

            for kh in tl.static_range(0, KH_C):
                for kw in tl.static_range(0, KW_C):
                    ih = oh + kh
                    iw = ow + kw
                    # x is NHWC: x[n, ih, iw, ic]
                    # offset: n*stride_xn + ih*stride_xh + iw*stride_xw + ic
                    x_base = pid_n * stride_xn + ih * stride_xh + iw * stride_xw  # [BLOCK_P]
                    # gather over IC: shape [BLOCK_P, IC]
                    x_ptrs = x_base[:, None] + ic_range[None, :]
                    x_m = (p_mask & (ih < H) & (iw < W))[:, None]
                    xv = tl.load(x_ptr + x_ptrs, mask=x_m, other=0.0)  # [BLOCK_P, IC]

                    # weight layout: (KH, KW, IC, OC) contiguous -> w[kh, kw, ic, oc]
                    # offset: ((kh*KW + kw)*IC + ic)*OC + oc
                    w_base = (kh * KW + kw) * IC * OC
                    w_ptrs = w_base + ic_range[:, None] * OC + oc_offs[None, :]  # [IC, OC]
                    w_m = oc_mask[None, :]
                    wv = tl.load(w_ptr + w_ptrs, mask=w_m, other=0.0)  # [IC, OC]

                    # conv_val[OC, BLOCK_P] += wv.T @ xv.T  ==  (xv @ wv).T
                    # xv: [BLOCK_P, IC], wv: [IC, OC] -> [BLOCK_P, OC]
                    partial = tl.dot(xv, wv)  # [BLOCK_P, OC]
                    conv_val += tl.trans(partial)

            pooled_acc += conv_val * inv_pool2

    sig = tl.sigmoid(pooled_acc)
    full_mask = oc_mask[:, None] & p_mask[None, :]
    sig = tl.where(full_mask, sig, 0.0)
    s = tl.sum(tl.sum(sig, axis=0), axis=0)

    tl.atomic_add(out_ptr + pid_n, s)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        # Pre-transpose weight to (KH, KW, IC, OC) layout
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            w_kkio = w.permute(2, 3, 1, 0).contiguous()  # (KH, KW, IC, OC)
            self.register_buffer('w_kkio', w_kkio.cuda())
            self.register_buffer('b_cuda', self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        POOL = self.pool_kernel_size

        OH = H - KH + 1
        OW = W - KW + 1
        PH = OH // POOL
        PW = OW // POOL

        if PH * POOL != OH or PW * POOL != OW:
            y = self.conv(x)
            y = self.avg_pool(y)
            y = torch.sigmoid(y)
            return torch.sum(y, dim=[1, 2, 3])

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, IC)

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_P = 64
        BLOCK_OC = 64
        n_p_tiles = (PH * PW + BLOCK_P - 1) // BLOCK_P
        n_oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC

        # strides for NHWC (in elements)
        stride_xn = H * W * IC
        stride_xh = W * IC
        stride_xw = IC

        grid = (N, n_oc_tiles, n_p_tiles)
        conv_pool_sigmoid_sum_kernel[grid](
            x_nhwc, self.w_kkio, self.b_cuda, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW, PH, PW,
            stride_xn, stride_xh, stride_xw,
            POOL=POOL,
            BLOCK_P=BLOCK_P,
            BLOCK_OC=BLOCK_OC,
            KH_C=KH,
            KW_C=KW,
            IC_C=IC,
            num_warps=4,
            num_stages=2,
        )
        return out