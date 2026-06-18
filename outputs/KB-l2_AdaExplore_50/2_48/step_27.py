import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr,           # input (N, D, H, W, IC) channels-last
    w_ptr,           # weight (OC, kT, kH, kW, IC)
    cb_ptr,          # conv bias (OC,)
    scale_ptr,       # scaling factor (OC,)
    bias_ptr,        # bias (OC,)
    out_ptr,         # output (N, OC, D_out, H_out, W_out)
    N, D_in, H_in, W_in,
    D_out, H_out, W_out,
    IC: tl.constexpr,
    OC: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)  # batch index

    # Total spatial outputs per batch
    DHW = D_out * H_out * W_out

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < DHW

    # Decompose m_offs into (d, h, w)
    w_idx = m_offs % W_out
    tmp = m_offs // W_out
    h_idx = tmp % H_out
    d_idx = tmp // H_out

    # Channel offsets for OC
    oc_offs = tl.arange(0, OC)  # (OC,)

    acc = tl.zeros((BLOCK_M, OC), dtype=tl.float32)

    # x base for this batch: pid_n * D_in * H_in * W_in * IC
    x_batch_ptr = x_ptr + pid_n * D_in * H_in * W_in * IC

    # Loop over kernel positions
    for kt in tl.static_range(KT):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                # input spatial coords
                in_d = d_idx + kt  # (BLOCK_M,)
                in_h = h_idx + kh
                in_w = w_idx + kw
                # offset into x: (in_d * H_in + in_h) * W_in + in_w, then * IC
                x_spatial = (in_d * H_in + in_h) * W_in + in_w  # (BLOCK_M,)
                x_base = x_spatial[:, None] * IC + tl.arange(0, IC)[None, :]  # (BLOCK_M, IC)

                x_vals = tl.load(x_batch_ptr + x_base, mask=m_mask[:, None], other=0.0)  # (BLOCK_M, IC)

                # weight offset: oc * (KT*KH*KW*IC) + (kt*KH+kh)*KW*IC + kw*IC + ic
                w_base = oc_offs[:, None] * (KT * KH * KW * IC) + \
                         ((kt * KH + kh) * KW + kw) * IC + tl.arange(0, IC)[None, :]
                w_vals = tl.load(w_ptr + w_base)  # (OC, IC)

                # acc += x_vals @ w_vals.T  => (BLOCK_M, OC)
                acc += tl.dot(x_vals, tl.trans(w_vals))

    # Add conv bias
    cb = tl.load(cb_ptr + oc_offs)  # (OC,)
    acc = acc + cb[None, :]

    # Apply scaling factor
    sc = tl.load(scale_ptr + oc_offs)  # (OC,)
    v = acc * sc[None, :]

    # tanh
    e = tl.exp(2.0 * v)
    t = (e - 1.0) / (e + 1.0)

    # multiply by bias
    bs = tl.load(bias_ptr + oc_offs)
    y = t * bs[None, :]

    # sigmoid
    out = 1.0 / (1.0 + tl.exp(-y))

    # Store: output is (N, OC, D_out, H_out, W_out)
    # offset: pid_n * OC * DHW + oc * DHW + m_offs
    out_batch_ptr = out_ptr + pid_n * OC * DHW
    out_offs = oc_offs[None, :] * DHW + m_offs[:, None]  # (BLOCK_M, OC)
    tl.store(out_batch_ptr + out_offs, out, mask=m_mask[:, None])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (N, IC, D, H, W)
        x = x.contiguous()
        N, IC, D_in, H_in, W_in = x.shape
        OC = self.out_channels
        KT = KH = KW = self.kernel_size
        D_out = D_in - KT + 1
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1

        # Permute input to (N, D, H, W, IC) channels-last
        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()

        # Permute weight (OC, IC, KT, KH, KW) -> (OC, KT, KH, KW, IC)
        w = self.conv.weight.permute(0, 2, 3, 4, 1).contiguous()
        cb = self.conv.bias.contiguous()

        scale = self.scaling_factor.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

        DHW = D_out * H_out * W_out
        BLOCK_M = 64
        grid = ((DHW + BLOCK_M - 1) // BLOCK_M, N)

        # Pad IC to power of 2 for tl.dot if needed. IC=3 -> need to handle.
        # tl.dot requires K dim >= 16 typically. Use IC_PAD=16 with zero padding.
        IC_PAD = 16
        if IC < IC_PAD:
            x_padded = torch.zeros((N, D_in, H_in, W_in, IC_PAD), device=x.device, dtype=x.dtype)
            x_padded[..., :IC] = x_nhwc
            w_padded = torch.zeros((OC, KT, KH, KW, IC_PAD), device=x.device, dtype=x.dtype)
            w_padded[..., :IC] = w
            x_use = x_padded
            w_use = w_padded
            IC_USE = IC_PAD
        else:
            x_use = x_nhwc
            w_use = w
            IC_USE = IC

        conv3d_fused_kernel[grid](
            x_use, w_use, cb, scale, bias, out,
            N, D_in, H_in, W_in,
            D_out, H_out, W_out,
            IC=IC_USE, OC=OC, KT=KT, KH=KH, KW=KW,
            BLOCK_M=BLOCK_M,
            num_warps=4,
            num_stages=2,
        )
        return out