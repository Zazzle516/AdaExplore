import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # x is NHWC: (N, IH, IW, IC)
    # w is (OC, KH, KW, IC)
    # out is (N, POH, POW, OC) -- we'll write NHWC pooled and permute outside
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            oh = poh * POOL + ph  # [BLOCK_SP]
            ow = pow_ * POOL + pw  # [BLOCK_SP]

            conv_val = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh  # [BLOCK_SP]
                    iw = ow + kw  # [BLOCK_SP]
                    # base offsets for x [BLOCK_SP] (without ic)
                    x_base = (pid_n * IH * IW * IC
                              + ih * (IW * IC)
                              + iw * IC)  # [BLOCK_SP]
                    # base offsets for w [BLOCK_OC] (without ic)
                    w_base = (oc_offs * (KH * KW * IC)
                              + kh * (KW * IC)
                              + kw * IC)  # [BLOCK_OC]

                    for ic_base in range(0, IC, BLOCK_IC):
                        ic_offs = ic_base + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                        ic_mask = ic_offs < IC

                        # x[pid_n, ih, iw, ic_offs]: shape [BLOCK_SP, BLOCK_IC]
                        x_off = x_base[:, None] + ic_offs[None, :]
                        x_mask = sp_mask[:, None] & ic_mask[None, :]
                        x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                        # w[oc_offs, kh, kw, ic_offs]: shape [BLOCK_OC, BLOCK_IC]
                        w_off = w_base[:, None] + ic_offs[None, :]
                        w_mask = oc_mask[:, None] & ic_mask[None, :]
                        w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                        # [BLOCK_SP, BLOCK_OC] += [BLOCK_SP, BLOCK_IC] @ [BLOCK_IC, BLOCK_OC]
                        conv_val += tl.dot(x_val, tl.trans(w_val))

            # bias
            b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
            conv_val += b_val[None, :]

            v = conv_val - SUB1
            e2 = tl.exp(2.0 * v)
            t = (e2 - 1.0) / (e2 + 1.0)
            v = t - SUB2
            acc += v

    inv = 1.0 / (POOL * POOL)
    acc = acc * inv

    # out NHWC: (N, POH*POW, OC)
    out_off = (pid_n * (POH * POW * OC)
               + sp_offs[:, None] * OC
               + oc_offs[None, :])
    mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # Pre-permute weight to (OC, KH, KW, IC) layout
        w = self.conv.weight.detach().contiguous()  # (OC, IC, KH, KW)
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)
        self.register_buffer("w_nhwc", w_nhwc.cuda())
        self.register_buffer("b_buf", self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, IH, IW, IC)

        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.kernel_size_pool
        POH = OH // POOL
        POW = OW // POOL

        out_nhwc = torch.empty((N, POH * POW, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_SP = 64
        BLOCK_IC = 32

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(POH * POW, BLOCK_SP))

        fused_conv_tanh_pool_kernel[grid](
            x_nhwc, self.w_nhwc, self.b_buf, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.subtract1_value, self.subtract2_value,
            BLOCK_OC, BLOCK_SP, BLOCK_IC,
            num_warps=4, num_stages=3,
        )

        # reshape to (N, POH, POW, OC) then permute to (N, OC, POH, POW)
        out = out_nhwc.view(N, POH, POW, OC).permute(0, 3, 1, 2).contiguous()
        return out