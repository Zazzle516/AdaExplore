import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OUT_HW', 'IC_KHKW'],
)
@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W, OC, KH, KW, OH, OW,
    scale,
    OUT_HW, IC_KHKW,
    stride_xn, stride_xh, stride_xw, stride_xc,  # NHWC strides
    stride_wkh, stride_wkw, stride_wi, stride_wo,  # KH,KW,IC,OC weight layout
    BLOCK_OC: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)
    sp_mask = offs_sp < OUT_HW
    oh = offs_sp // OW
    ow = offs_sp % OW

    offs_oc = tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < OC

    x_batch_ptr = x_ptr + pid_n * stride_xn

    KHKW = KH * KW

    acc = tl.zeros([BLOCK_OC, BLOCK_N], dtype=tl.float32)

    # iterate over reduction K = IC*KH*KW
    for k_start in range(0, IC_KHKW, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < IC_KHKW

        # decode K -> (kh, kw, ic) since weight layout is (KH, KW, IC, OC)
        # here we order K as kh*KW*IC + kw*IC + ic for contiguous IC
        kh = offs_k // (KW * IC)
        rem = offs_k % (KW * IC)
        kw = rem // IC
        ic = rem % IC

        # weight pointers: [BLOCK_K, BLOCK_OC]
        w_ptrs = (w_ptr
                  + kh[:, None] * stride_wkh
                  + kw[:, None] * stride_wkw
                  + ic[:, None] * stride_wi
                  + offs_oc[None, :] * stride_wo)
        w_load_mask = k_mask[:, None] & oc_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

        # input pointers: [BLOCK_N, BLOCK_K], NHWC layout
        h_idx = oh[:, None] + kh[None, :]  # [BLOCK_N, BLOCK_K]
        w_idx = ow[:, None] + kw[None, :]
        x_ptrs = (x_batch_ptr
                  + h_idx * stride_xh
                  + w_idx * stride_xw
                  + ic[None, :] * stride_xc)
        x_load_mask = sp_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

        # acc[BLOCK_OC, BLOCK_N] += w[K, OC]^T @ x[N, K]^T
        # tl.dot(w_vals.T, x_vals.T) -> need [OC, K] @ [K, N]
        acc += tl.dot(tl.trans(w_vals), tl.trans(x_vals))

    # add bias
    b_vals = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc = acc + b_vals[:, None]
    acc = acc * scale

    # mask out invalid OC rows
    acc = tl.where(oc_mask[:, None], acc, float('inf'))

    # reduce along OC axis
    min_val = tl.min(acc, axis=0)

    out_ptrs = out_ptr + pid_n * OUT_HW + offs_sp
    tl.store(out_ptrs, min_val, mask=sp_mask)


def conv_scale_min(x_nhwc, weight_kkio, bias, scale, N, IC, H, W, OC, KH, KW):
    OH = H - KH + 1
    OW = W - KW + 1
    OUT_HW = OH * OW
    IC_KHKW = IC * KH * KW

    out = torch.empty((N, 1, OH, OW), device=x_nhwc.device, dtype=torch.float32)

    # next power of 2 >= OC for BLOCK_OC
    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2

    grid = lambda meta: (N, triton.cdiv(OUT_HW, meta['BLOCK_N']))

    conv_scale_min_kernel[grid](
        x_nhwc, weight_kkio, bias, out,
        N, IC, H, W, OC, KH, KW, OH, OW,
        scale,
        OUT_HW, IC_KHKW,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        weight_kkio.stride(0), weight_kkio.stride(1), weight_kkio.stride(2), weight_kkio.stride(3),
        BLOCK_OC=BLOCK_OC,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to (KH, KW, IC, OC) for contiguous IC-major K reduction
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            w_perm = w.permute(2, 3, 1, 0).contiguous()  # (KH, KW, IC, OC)
        self.register_buffer('weight_perm', w_perm)
        self.register_buffer('bias_buf', self.conv.bias.detach().contiguous())

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size

        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w = self.weight_perm
        b = self.bias_buf
        if w.device != x.device:
            w = w.to(x.device)
            b = b.to(x.device)
            self.weight_perm = w
            self.bias_buf = b

        return conv_scale_min(x_nhwc, w, b, float(self.scale_factor), N, IC, H, W, OC, KH, KW)