import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3d_mish_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    M, K_total, N_oc,
    stride_xn, stride_xd, stride_xh, stride_xw, stride_xc,
    stride_wkd, stride_wkh, stride_wkw, stride_wic, stride_woc,
    stride_on, stride_od, stride_oh, stride_ow, stride_oc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_eff = M
    N_eff = N_oc

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # over output spatial+batch
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # over OC

    # Decompose offs_m into (n, od, oh, ow)
    # M = N * OD * OH * OW
    OHW = OH * OW
    ODHW = OD * OHW
    n_idx = offs_m // ODHW
    rem = offs_m % ODHW
    od_idx = rem // OHW
    rem2 = rem % OHW
    oh_idx = rem2 // OW
    ow_idx = rem2 % OW

    m_mask = offs_m < M_eff
    n_mask = offs_n < N_eff

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K_total = IC * KD * KH * KW
    # Loop over K in BLOCK_K chunks
    KHKW = KH * KW
    KDKHKW = KD * KHKW

    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total

        # decompose k -> (ic, kd, kh, kw)
        ic_k = offs_k // KDKHKW
        rk = offs_k % KDKHKW
        kd_k = rk // KHKW
        rk2 = rk % KHKW
        kh_k = rk2 // KW
        kw_k = rk2 % KW

        # input coords for each (m, k): id = od + kd, etc. (no padding, stride 1)
        id_mk = od_idx[:, None] + kd_k[None, :]  # [BM, BK]
        ih_mk = oh_idx[:, None] + kh_k[None, :]
        iw_mk = ow_idx[:, None] + kw_k[None, :]
        n_mk = n_idx[:, None]
        ic_mk = ic_k[None, :]

        x_off = (n_mk * stride_xn
                 + id_mk * stride_xd
                 + ih_mk * stride_xh
                 + iw_mk * stride_xw
                 + ic_mk * stride_xc)
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # weight: [KD, KH, KW, IC, OC] in NDHWC-style
        w_off = (kd_k[:, None] * stride_wkd
                 + kh_k[:, None] * stride_wkh
                 + kw_k[:, None] * stride_wkw
                 + ic_k[:, None] * stride_wic
                 + offs_n[None, :] * stride_woc)
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # bias
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # mish: x * tanh(softplus(x)); tanh via exp
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    t1 = (e2 - 1.0) / (e2 + 1.0)
    mish = acc * t1
    e2b = tl.exp(2.0 * mish)
    out = (e2b - 1.0) / (e2b + 1.0)

    # store - output is NDHWC
    out_off = (n_idx[:, None] * stride_on
               + od_idx[:, None] * stride_od
               + oh_idx[:, None] * stride_oh
               + ow_idx[:, None] * stride_ow
               + offs_n[None, :] * stride_oc)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Only use custom kernel for stride=1, padding=0
        if self.stride != 1 or self.padding != 0:
            x = self.conv(x)
            return torch.tanh(F.mish(x))

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kd, self.kh, self.kw
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Convert input to NDHWC (channels_last3d)
        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()  # [N, ID, IH, IW, IC]

        # Convert weight from [OC, IC, KD, KH, KW] to [KD, KH, KW, IC, OC]
        w = self.conv.weight  # [OC, IC, KD, KH, KW]
        w_perm = w.permute(2, 3, 4, 1, 0).contiguous()  # [KD, KH, KW, IC, OC]

        b = self.conv.bias.contiguous()

        # Output in NDHWC
        out_nhwc = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

        M = N * OD * OH * OW
        K_total = IC * KD * KH * KW
        N_oc = OC

        # strides (in elements)
        stride_xn = ID * IH * IW * IC
        stride_xd = IH * IW * IC
        stride_xh = IW * IC
        stride_xw = IC
        stride_xc = 1

        stride_wkd = KH * KW * IC * OC
        stride_wkh = KW * IC * OC
        stride_wkw = IC * OC
        stride_wic = OC
        stride_woc = 1

        stride_on = OD * OH * OW * OC
        stride_od = OH * OW * OC
        stride_oh = OW * OC
        stride_ow = OC
        stride_oc = 1

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N_oc, META['BLOCK_N']),
        )

        conv3d_mish_tanh_kernel[grid](
            x_nhwc, w_perm, b, out_nhwc,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            M, K_total, N_oc,
            stride_xn, stride_xd, stride_xh, stride_xw, stride_xc,
            stride_wkd, stride_wkh, stride_wkw, stride_wic, stride_woc,
            stride_on, stride_od, stride_oh, stride_ow, stride_oc,
        )

        # Convert back to NCDHW
        out = out_nhwc.permute(0, 4, 1, 2, 3).contiguous()
        return out