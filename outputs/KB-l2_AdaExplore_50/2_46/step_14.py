import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_pool_kernel_nhwc(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    """
    NHWC layout for x (N, IH, IW, IC), weights (OC, KH, KW, IC).
    Output: (N, OC, POH, POW)
    One program computes a tile of [BLOCK_OC, BLOCK_PH, BLOCK_PW] pooled outputs.
    Within the tile we compute 2x2 pool window outputs by tiling output spatial.
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    n_ph_tiles = (POH + BLOCK_PH - 1) // BLOCK_PH
    n_pw_tiles = (POW + BLOCK_PW - 1) // BLOCK_PW

    pid_ph = pid_sp // n_pw_tiles
    pid_pw = pid_sp % n_pw_tiles

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    ph_offs = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)  # [BLOCK_PH]
    pw_offs = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)  # [BLOCK_PW]

    oc_mask = oc_offs < OC
    ph_mask = ph_offs < POH
    pw_mask = pw_offs < POW

    BLOCK_SP: tl.constexpr = BLOCK_PH * BLOCK_PW
    # Flatten pooled spatial grid for output: [BLOCK_PH, BLOCK_PW] -> [BLOCK_SP]
    sp_ph = (tl.arange(0, BLOCK_SP) // BLOCK_PW)
    sp_pw = (tl.arange(0, BLOCK_SP) % BLOCK_PW)
    poh = pid_ph * BLOCK_PH + sp_ph  # [BLOCK_SP]
    pow_ = pid_pw * BLOCK_PW + sp_pw  # [BLOCK_SP]
    sp_mask = (poh < POH) & (pow_ < POW)

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    inv = 1.0 / (POOL * POOL)

    # Loop over POOL window positions
    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            oh = poh * POOL + ph  # [BLOCK_SP]
            ow = pow_ * POOL + pw  # [BLOCK_SP]

            conv_val = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh
                    iw = ow + kw
                    x_base = (pid_n * IH * IW * IC
                              + ih[None, :] * (IW * IC)
                              + iw[None, :] * IC)  # [1, BLOCK_SP]
                    w_base = (oc_offs[:, None] * (KH * KW * IC)
                              + kh * (KW * IC)
                              + kw * IC)  # [BLOCK_OC, 1]

                    for ic_base in range(0, IC, BLOCK_IC):
                        ic_offs = ic_base + tl.arange(0, BLOCK_IC)
                        ic_mask = ic_offs < IC

                        x_off = ic_offs[:, None] + x_base  # [BLOCK_IC, BLOCK_SP]
                        x_mask = ic_mask[:, None] & sp_mask[None, :]
                        x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                        w_off = w_base + ic_offs[None, :]
                        w_mask = oc_mask[:, None] & ic_mask[None, :]
                        w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                        conv_val += tl.dot(w_val, x_val)

            b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
            conv_val += b_val[:, None]

            v = conv_val - SUB1
            e2 = tl.exp(2.0 * v)
            t = (e2 - 1.0) / (e2 + 1.0)
            v = t - SUB2
            acc += v

    acc = acc * inv

    sp_offs_lin = poh * POW + pow_  # [BLOCK_SP]
    out_off = (pid_n * (OC * POH * POW)
               + oc_offs[:, None] * (POH * POW)
               + sp_offs_lin[None, :])
    mask = oc_mask[:, None] & sp_mask[None, :]
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

        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)
        self.register_buffer("w_nhwc", w_nhwc.cuda(), persistent=False)
        self.register_buffer("b_buf", self.conv.bias.detach().contiguous().cuda(), persistent=False)

    def forward(self, x):
        x = x.cuda()
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        N, IH, IW, IC = x_nhwc.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.kernel_size_pool
        POH = OH // POOL
        POW = OW // POOL

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_PH = 8
        BLOCK_PW = 8
        BLOCK_IC = 32

        n_ph_tiles = (POH + BLOCK_PH - 1) // BLOCK_PH
        n_pw_tiles = (POW + BLOCK_PW - 1) // BLOCK_PW

        grid = (N, triton.cdiv(OC, BLOCK_OC), n_ph_tiles * n_pw_tiles)

        fused_conv_tanh_pool_kernel_nhwc[grid](
            x_nhwc, self.w_nhwc, self.b_buf, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.subtract1_value, self.subtract2_value,
            BLOCK_OC, BLOCK_PH, BLOCK_PW, BLOCK_IC,
            num_warps=4, num_stages=2,
        )
        return out