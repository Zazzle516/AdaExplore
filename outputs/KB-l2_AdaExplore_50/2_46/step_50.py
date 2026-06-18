import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr,           # NHWC input: [N, H, W, IC]
    w_ptr,           # [OC, KH*KW*IC]
    b_ptr,           # [OC]
    out_ptr,         # [N, OC, POH, POW]
    N, IC, H, W,
    OC,
    OH, OW,
    POH, POW,
    sub1, sub2,
    stride_xn, stride_xh, stride_xw, stride_xc,
    BLOCK_M: tl.constexpr,    # pooled-spatial tile (POH_TILE * POW_TILE)
    BLOCK_N: tl.constexpr,    # OC tile
    BLOCK_K: tl.constexpr,    # IC tile
    POH_TILE: tl.constexpr,
    POW_TILE: tl.constexpr,
    PK: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    n_pool_tiles_w = (POW + POW_TILE - 1) // POW_TILE
    pth = pid_sp // n_pool_tiles_w
    ptw = pid_sp % n_pool_tiles_w

    # pooled coords for this tile
    poh_start = pth * POH_TILE
    pow_start = ptw * POW_TILE

    # conv-output tile size
    CONV_H: tl.constexpr = POH_TILE * PK
    CONV_W: tl.constexpr = POW_TILE * PK
    M_CONV: tl.constexpr = CONV_H * CONV_W

    oh_start = poh_start * PK
    ow_start = pow_start * PK

    # offsets within conv tile [M_CONV]
    m_idx = tl.arange(0, M_CONV)
    m_h = m_idx // CONV_W
    m_w = m_idx % CONV_W

    oh = oh_start + m_h  # [M_CONV]
    ow = ow_start + m_w  # [M_CONV]

    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    oc_mask = oc_offs < OC

    # accumulator for conv tile
    acc = tl.zeros((M_CONV, BLOCK_N), dtype=tl.float32)

    # weight layout: [OC, KH, KW, IC] flattened as [OC, KH*KW*IC]
    ic_offs = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    ic_mask = ic_offs < IC

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # [M_CONV]
            iw = ow + kw  # [M_CONV]
            sp_valid = (ih < H) & (iw < W)

            # base offset into input for this (kh, kw)
            x_base = pid_n * stride_xn + ih * stride_xh + iw * stride_xw  # [M_CONV]
            # weight base: oc * (KH*KW*IC) + (kh*KW + kw)*IC
            w_base = oc_offs * (KH * KW * IC) + (kh * KW + kw) * IC  # [BLOCK_N]

            # load x: [M_CONV, BLOCK_K]
            x_ptrs = x_ptr + x_base[:, None] + ic_offs[None, :] * stride_xc
            x_mask = sp_valid[:, None] & ic_mask[None, :]
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [M_CONV, BLOCK_K]

            # load weights: [BLOCK_K, BLOCK_N]
            w_ptrs = w_ptr + w_base[None, :] + ic_offs[:, None]
            w_mask = oc_mask[None, :] & ic_mask[:, None]
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

            acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # bias
    b = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_N]
    acc = acc + b[None, :]

    # epilogue: -sub1, tanh, -sub2
    v = acc - sub1
    e2 = tl.exp(2.0 * v)
    t = (e2 - 1.0) / (e2 + 1.0)
    v = t - sub2  # [M_CONV, BLOCK_N]

    # pool reduce: average over PK x PK windows via reshape
    POOL_M: tl.constexpr = POH_TILE * POW_TILE
    # v: [M_CONV, BLOCK_N] = [POH_TILE*PK*POW_TILE*PK, BLOCK_N]
    # Step 1: reshape to [POH_TILE, PK, POW_TILE*PK, BLOCK_N], sum axis=1
    v1 = tl.reshape(v, (POH_TILE, PK, POW_TILE * PK, BLOCK_N))
    v1 = tl.sum(v1, axis=1)  # [POH_TILE, POW_TILE*PK, BLOCK_N]
    # Step 2: reshape to [POH_TILE, POW_TILE, PK, BLOCK_N], sum axis=2
    v2 = tl.reshape(v1, (POH_TILE, POW_TILE, PK, BLOCK_N))
    v2 = tl.sum(v2, axis=2)  # [POH_TILE, POW_TILE, BLOCK_N]
    pool_acc = tl.reshape(v2, (POOL_M, BLOCK_N)) / (PK * PK)

    # Store: output is [N, OC, POH, POW], contiguous
    # pooled coord for each row
    pm_idx = tl.arange(0, POOL_M)
    pm_h = pm_idx // POW_TILE
    pm_w = pm_idx % POW_TILE
    poh_coord = poh_start + pm_h  # [POOL_M]
    pow_coord = pow_start + pm_w
    pool_valid = (poh_coord < POH) & (pow_coord < POW)

    out_off = (pid_n * OC * POH * POW
               + oc_offs[None, :] * (POH * POW)
               + poh_coord[:, None] * POW
               + pow_coord[:, None])
    out_mask = pool_valid[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, pool_acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size = kernel_size
        self.kernel_size_pool = kernel_size_pool
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Pre-pack weight as [OC, KH, KW, IC] contiguous
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            w_packed = w.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]
            self.register_buffer("w_packed", w_packed.cuda())
            self.register_buffer("b_packed", self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, IC]

        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        PK = self.kernel_size_pool
        POH = OH // PK
        POW = OW // PK

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        # tile config
        POH_TILE = 8
        POW_TILE = 8
        BLOCK_N = 64
        BLOCK_K = 64  # IC=64 fits in one K-block

        n_pool_tiles_h = (POH + POH_TILE - 1) // POH_TILE
        n_pool_tiles_w = (POW + POW_TILE - 1) // POW_TILE

        BLOCK_M = POH_TILE * POW_TILE

        grid = (
            N,
            (OC + BLOCK_N - 1) // BLOCK_N,
            n_pool_tiles_h * n_pool_tiles_w,
        )

        stride_xn = H * W * IC
        stride_xh = W * IC
        stride_xw = IC
        stride_xc = 1

        fused_conv_tanh_pool_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, out,
            N, IC, H, W,
            OC,
            OH, OW,
            POH, POW,
            self.subtract1_value, self.subtract2_value,
            stride_xn, stride_xh, stride_xw, stride_xc,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            POH_TILE=POH_TILE,
            POW_TILE=POW_TILE,
            PK=PK,
            KH=KH,
            KW=KW,
            num_warps=4,
            num_stages=3,
        )
        return out