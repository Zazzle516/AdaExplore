import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,        # conv output spatial
    POH, POW,      # pooled output spatial
    sub1, sub2,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    PK: tl.constexpr,  # pool kernel (constexpr)
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC

    poh = sp_offs // POW
    pow_ = sp_offs % POW
    sp_mask = sp_offs < (POH * POW)

    # pool window upper-left in conv-output coords
    oh_base = poh * PK
    ow_base = pow_ * PK

    # accumulator for pooled output: [BLOCK_OC, BLOCK_SP]
    pool_acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Iterate over pool window (PK x PK)
    for pi in tl.static_range(0, PK):
        for pj in tl.static_range(0, PK):
            oh = oh_base + pi  # [BLOCK_SP]
            ow = ow_base + pj  # [BLOCK_SP]

            # conv accumulator for this output position
            conv_acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            # Loop over IC, KH, KW
            for ic in range(0, IC):
                for kh in tl.static_range(0, 3):
                    for kw in tl.static_range(0, 3):
                        ih = oh + kh  # [BLOCK_SP]
                        iw = ow + kw  # [BLOCK_SP]
                        # input ptr
                        in_off = (pid_n * IC * H * W
                                  + ic * H * W
                                  + ih[None, :] * W
                                  + iw[None, :])
                        in_mask = sp_mask[None, :] & (ih[None, :] < H) & (iw[None, :] < W)
                        x_val = tl.load(x_ptr + in_off, mask=in_mask, other=0.0)  # [1, BLOCK_SP]

                        # weight ptr: [OC, IC, KH, KW]
                        w_off = (oc_offs[:, None] * IC * 3 * 3
                                 + ic * 3 * 3
                                 + kh * 3
                                 + kw)
                        w_mask = oc_mask[:, None]
                        w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, 1]

                        conv_acc += w_val * x_val

            # add bias
            b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
            conv_acc += b_val[:, None]

            # subtract1, tanh, subtract2
            v = conv_acc - sub1
            # tanh via exp
            e2 = tl.exp(2.0 * v)
            t = (e2 - 1.0) / (e2 + 1.0)
            v = t - sub2

            pool_acc += v

    pool_acc = pool_acc / (PK * PK)

    # store
    out_off = (pid_n * OC * POH * POW
               + oc_offs[:, None] * POH * POW
               + sp_offs[None, :])
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, pool_acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.kernel_size = kernel_size
        self.kernel_size_pool = kernel_size_pool
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        PK = self.kernel_size_pool
        POH = OH // PK
        POW = OW // PK

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64

        grid = (
            N,
            (OC + BLOCK_OC - 1) // BLOCK_OC,
            (POH * POW + BLOCK_SP - 1) // BLOCK_SP,
        )

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            POH, POW,
            float(self.subtract1_value), float(self.subtract2_value),
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            PK=PK,
            num_warps=4,
            num_stages=2,
        )
        return out