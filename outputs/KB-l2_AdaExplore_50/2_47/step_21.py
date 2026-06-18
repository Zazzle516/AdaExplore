import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K', 'OD', 'OH', 'OW', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_mish_tanh_kernel(
    x_ptr,    # [N, ID, IH, IW, IC] channels-last
    w_ptr,    # [OC, KD, KH, KW, IC] (out, kd, kh, kw, in)
    b_ptr,    # [OC]
    y_ptr,    # [N, OD, OH, OW, OC] channels-last
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    M, K,  # M = OD*OH*OW, K = IC*KD*KH*KW
    stride_xn, stride_xd, stride_xh, stride_xw,  # IC contiguous = 1
    stride_yn, stride_yd, stride_yh, stride_yw,  # OC contiguous = 1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_n_blk = tl.program_id(1)  # OC tile
    pid_batch = tl.program_id(2)

    num_m_blocks = tl.cdiv(M, BLOCK_M)
    pid_m_blk = pid_mn

    offs_m = pid_m_blk * BLOCK_M + tl.arange(0, BLOCK_M)   # output spatial
    offs_n = pid_n_blk * BLOCK_N + tl.arange(0, BLOCK_N)   # OC

    # Decompose offs_m into od, oh, ow
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    od = tmp // OH

    m_mask = offs_m < M
    n_mask = offs_n < OC

    # Base input d/h/w for each output position
    id_base = od * SD - PD
    ih_base = oh * SH - PH
    iw_base = ow * SW - PW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kd, kh, kw, and IC in chunks
    KDHW = KD * KH * KW
    # We iterate over k = kd*KH*KW*IC + kh*KW*IC + kw*IC + ic
    # Outer loops over kd,kh,kw (compile-time), inner over IC tiles

    x_batch_ptr = x_ptr + pid_batch * stride_xn

    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                cur_id = id_base + kd
                cur_ih = ih_base + kh
                cur_iw = iw_base + kw
                d_ok = (cur_id >= 0) & (cur_id < ID)
                h_ok = (cur_ih >= 0) & (cur_ih < IH)
                w_ok = (cur_iw >= 0) & (cur_iw < IW)
                spatial_ok = d_ok & h_ok & w_ok & m_mask

                # input ptr offset per m (without IC)
                x_off_m = cur_id * stride_xd + cur_ih * stride_xh + cur_iw * stride_xw  # [BLOCK_M]

                # weight ptr offset per n (without IC)
                # w layout: [OC, KD, KH, KW, IC] -> idx = oc*(KD*KH*KW*IC) + kd*(KH*KW*IC) + kh*(KW*IC) + kw*IC + ic
                w_off_n = offs_n * (KD * KH * KW * IC) + (kd * KH * KW + kh * KW + kw) * IC  # [BLOCK_N]

                for ic_start in range(0, IC, BLOCK_K):
                    offs_k = ic_start + tl.arange(0, BLOCK_K)
                    k_mask = offs_k < IC

                    # Load x: [BLOCK_M, BLOCK_K]
                    x_ptrs = x_batch_ptr + x_off_m[:, None] + offs_k[None, :]
                    x_load_mask = spatial_ok[:, None] & k_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                    # Load w: [BLOCK_K, BLOCK_N]
                    w_ptrs = w_ptr + w_off_n[None, :] + offs_k[:, None]
                    w_load_mask = n_mask[None, :] & k_mask[:, None]
                    w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals)

    # bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x)); softplus = log(1+exp(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th_sp = (e2 - 1.0) / (e2 + 1.0)
    mish = acc * th_sp
    # tanh(mish)
    e2m = tl.exp(2.0 * mish)
    out = (e2m - 1.0) / (e2m + 1.0)

    # Store: y[N, OD, OH, OW, OC]
    y_batch_ptr = y_ptr + pid_batch * stride_yn
    y_off_m = od * stride_yd + oh * stride_yh + ow * stride_yw  # [BLOCK_M]
    y_ptrs = y_batch_ptr + y_off_m[:, None] + offs_n[None, :]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, out, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.KD = self.KH = self.KW = kernel_size
        else:
            self.KD, self.KH, self.KW = kernel_size
        if isinstance(stride, int):
            self.SD = self.SH = self.SW = stride
        else:
            self.SD, self.SH, self.SW = stride
        if isinstance(padding, int):
            self.PD = self.PH = self.PW = padding
        else:
            self.PD, self.PH, self.PW = padding

        # Pre-permute weight to [OC, KD, KH, KW, IC] for channels-last input
        with torch.no_grad():
            w = self.conv.weight.detach().clone()  # [OC, IC, KD, KH, KW]
            w = w.permute(0, 2, 3, 4, 1).contiguous()  # [OC, KD, KH, KW, IC]
        self.register_buffer('weight_cl', w)
        self.register_buffer('bias_cl', self.conv.bias.detach().clone())

    def _refresh_weights(self):
        # If user modified self.conv params, refresh (cheap check)
        with torch.no_grad():
            w = self.conv.weight.detach().permute(0, 2, 3, 4, 1).contiguous()
            if (w.shape != self.weight_cl.shape) or (not torch.equal(w, self.weight_cl)):
                self.weight_cl = w
            if not torch.equal(self.conv.bias.detach(), self.bias_cl):
                self.bias_cl = self.conv.bias.detach().clone()

    def forward(self, x):
        # x: [N, IC, D, H, W]
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.KD, self.KH, self.KW
        SD, SH, SW = self.SD, self.SH, self.SW
        PD, PH, PW = self.PD, self.PH, self.PW
        OC = self.out_channels
        OD = (ID + 2 * PD - KD) // SD + 1
        OH = (IH + 2 * PH - KH) // SH + 1
        OW = (IW + 2 * PW - KW) // SW + 1

        # Convert input to channels-last (NDHWC)
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()  # [N, ID, IH, IW, IC]

        w_cl = self.weight_cl.to(device=x.device, dtype=x.dtype)
        b_cl = self.bias_cl.to(device=x.device, dtype=x.dtype)

        y_cl = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

        M = OD * OH * OW
        K = IC * KD * KH * KW

        # Strides for channels-last
        stride_xn = ID * IH * IW * IC
        stride_xd = IH * IW * IC
        stride_xh = IW * IC
        stride_xw = IC

        stride_yn = OD * OH * OW * OC
        stride_yd = OH * OW * OC
        stride_yh = OW * OC
        stride_yw = OC

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(OC, META['BLOCK_N']),
            N,
        )

        conv3d_mish_tanh_kernel[grid](
            x_cl, w_cl, b_cl, y_cl,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            M, K,
            stride_xn, stride_xd, stride_xh, stride_xw,
            stride_yn, stride_yd, stride_yh, stride_yw,
        )

        # Convert back to NCDHW
        y = y_cl.permute(0, 4, 1, 2, 3).contiguous()
        return y