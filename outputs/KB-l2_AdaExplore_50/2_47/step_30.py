import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, ID, IH, IW,
    OD, OH, OW,
    stride_xn, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    OUT_SPATIAL,
    IC: tl.constexpr,
    OC: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)  # spatial tile
    pid_b = tl.program_id(1)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, OC)
    offs_ic = tl.arange(0, IC)

    OH_OW = OH * OW
    od = offs_m // OH_OW
    rem = offs_m % OH_OW
    oh = rem // OW
    ow = rem % OW

    m_mask = offs_m < OUT_SPATIAL

    acc = tl.zeros((BLOCK_M, OC), dtype=tl.float32)

    x_batch = x_ptr + pid_b * stride_xn

    # nested loops over (kt, kh, kw); GEMM-K is IC
    for kt in tl.static_range(0, KT):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                id_idx = od + kt
                ih_idx = oh + kh
                iw_idx = ow + kw

                # input vals: (BLOCK_M, IC), NDHWC layout
                x_offsets = (id_idx[:, None] * stride_xd +
                             ih_idx[:, None] * stride_xh +
                             iw_idx[:, None] * stride_xw +
                             offs_ic[None, :])
                x_vals = tl.load(x_batch + x_offsets,
                                 mask=m_mask[:, None],
                                 other=0.0)

                # weight: (kT, kH, kW, IC, OC), flat layout
                # k_index = ((kt*KH + kh)*KW + kw)*IC + ic
                k_base = ((kt * KH + kh) * KW + kw) * IC
                w_offsets = (k_base + offs_ic[:, None]) * OC + offs_n[None, :]
                w_vals = tl.load(w_ptr + w_offsets)

                acc += tl.dot(x_vals, w_vals)

    # bias
    b_vals = tl.load(b_ptr + offs_n)
    acc += b_vals[None, :]

    # fused mish then tanh
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    mish = acc * th
    e2m = tl.exp(2.0 * mish)
    out = (e2m - 1.0) / (e2m + 1.0)

    # store directly into NCDHW output: out[n, oc, od, oh, ow]
    out_batch = out_ptr + pid_b * stride_on
    out_offsets = (offs_n[None, :] * stride_oc +
                   od[:, None] * stride_od +
                   oh[:, None] * stride_oh +
                   ow[:, None] * stride_ow)
    tl.store(out_batch + out_offsets, out, mask=m_mask[:, None])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == 1 and padding == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kT = self.kH = self.kW = kernel_size
        else:
            self.kT, self.kH, self.kW = kernel_size

        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        w = conv.weight.data.detach()  # (OC, IC, kT, kH, kW)
        # permute to (kT, kH, kW, IC, OC)
        w_perm = w.permute(2, 3, 4, 1, 0).contiguous()
        K_TOTAL = self.kT * self.kH * self.kW * in_channels
        w_flat = w_perm.view(K_TOTAL, out_channels).contiguous()
        self.weight = nn.Parameter(w_flat)
        self.bias = nn.Parameter(conv.bias.data.detach().clone())

    def forward(self, x):
        N, IC, D, H, W = x.shape
        OD = D - self.kT + 1
        OH = H - self.kH + 1
        OW = W - self.kW + 1
        OC = self.out_channels

        # permute input to NDHWC
        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()

        # output in NCDHW directly
        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        OUT_SPATIAL = OD * OH * OW

        BLOCK_M = 128

        grid = (triton.cdiv(OUT_SPATIAL, BLOCK_M), N)

        sxn, sxd, sxh, sxw, _ = x_nhwc.stride()
        son, soc, sod, soh, sow = out.stride()

        conv3d_fused_kernel[grid](
            x_nhwc, self.weight, self.bias, out,
            N, D, H, W,
            OD, OH, OW,
            sxn, sxd, sxh, sxw,
            son, soc, sod, soh, sow,
            OUT_SPATIAL,
            IC=IC, OC=OC,
            KT=self.kT, KH=self.kH, KW=self.kW,
            BLOCK_M=BLOCK_M,
            num_warps=8, num_stages=2,
        )

        return out