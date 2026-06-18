import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KT, KH, KW,
    stride_xn, stride_xd, stride_xh, stride_xw, stride_xc,
    stride_wk, stride_woc,
    stride_on, stride_od, stride_oh, stride_ow, stride_oc,
    OUT_SPATIAL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TOTAL: tl.constexpr,
):
    pid_m = tl.program_id(0)  # spatial tile
    pid_n = tl.program_id(1)  # oc tile
    pid_b = tl.program_id(2)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # decode spatial indices
    OH_OW = OH * OW
    od = offs_m // OH_OW
    rem = offs_m % OH_OW
    oh = rem // OW
    ow = rem % OW

    m_mask = offs_m < OUT_SPATIAL
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # x base for this batch
    x_batch = x_ptr + pid_b * stride_xn

    # Loop over K = KT*KH*KW*IC
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_TOTAL

        # decode k -> (kt, kh, kw, ic)
        kt = offs_k // (KH * KW * IC)
        krem = offs_k % (KH * KW * IC)
        kh = krem // (KW * IC)
        krem2 = krem % (KW * IC)
        kw = krem2 // IC
        ic = krem2 % IC

        # input spatial coords per (m, k)
        id_idx = od[:, None] + kt[None, :]  # (BLOCK_M, BLOCK_K)
        ih_idx = oh[:, None] + kh[None, :]
        iw_idx = ow[:, None] + kw[None, :]
        ic_idx = ic[None, :]

        x_offsets = (id_idx * stride_xd +
                     ih_idx * stride_xh +
                     iw_idx * stride_xw +
                     ic_idx * stride_xc)

        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_batch + x_offsets, mask=x_load_mask, other=0.0)

        # weight load: shape (BLOCK_K, BLOCK_N), w layout (K_TOTAL, OC)
        w_offsets = offs_k[:, None] * stride_wk + offs_n[None, :] * stride_woc
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_offsets, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # add bias
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b_vals[None, :]

    # fused mish then tanh
    # mish(x) = x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    mish = acc * th
    e2m = tl.exp(2.0 * mish)
    out = (e2m - 1.0) / (e2m + 1.0)

    # store
    out_batch = out_ptr + pid_b * stride_on
    out_offsets = (od[:, None] * stride_od +
                   oh[:, None] * stride_oh +
                   ow[:, None] * stride_ow +
                   offs_n[None, :] * stride_oc)
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_batch + out_offsets, out, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == 1 and padding == 0, "This kernel supports stride=1, padding=0"
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kT = self.kH = self.kW = kernel_size
        else:
            self.kT, self.kH, self.kW = kernel_size

        # use conv to init weights with correct distribution
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # weight: (OC, IC, kT, kH, kW) -> rearrange to (kT, kH, kW, IC, OC) flat (K_TOTAL, OC)
        w = conv.weight.data.detach()  # (OC, IC, kT, kH, kW)
        # permute to (kT, kH, kW, IC, OC)
        w_perm = w.permute(2, 3, 4, 1, 0).contiguous()
        K_TOTAL = self.kT * self.kH * self.kW * in_channels
        w_flat = w_perm.view(K_TOTAL, out_channels).contiguous()
        self.weight = nn.Parameter(w_flat)
        self.bias = nn.Parameter(conv.bias.data.detach().clone())

    def forward(self, x):
        # x: (N, IC, D, H, W) -> NDHWC
        N, IC, D, H, W = x.shape
        OD = D - self.kT + 1
        OH = H - self.kH + 1
        OW = W - self.kW + 1
        OC = self.out_channels

        # permute input to NDHWC (channels last)
        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()  # (N, D, H, W, IC)

        # output in NDHWC layout
        out_nhwc = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

        OUT_SPATIAL = OD * OH * OW
        K_TOTAL = self.kT * self.kH * self.kW * IC

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (
            triton.cdiv(OUT_SPATIAL, BLOCK_M),
            triton.cdiv(OC, BLOCK_N),
            N,
        )

        # strides in elements
        sxn, sxd, sxh, sxw, sxc = x_nhwc.stride()
        son, sod, soh, sow, soc = out_nhwc.stride()
        swk, swoc = self.weight.stride()

        conv3d_fused_kernel[grid](
            x_nhwc, self.weight, self.bias, out_nhwc,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            self.kT, self.kH, self.kW,
            sxn, sxd, sxh, sxw, sxc,
            swk, swoc,
            son, sod, soh, sow, soc,
            OUT_SPATIAL,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            K_TOTAL=K_TOTAL,
            num_warps=4, num_stages=2,
        )

        # permute back to NCDHW
        return out_nhwc.permute(0, 4, 1, 2, 3).contiguous()