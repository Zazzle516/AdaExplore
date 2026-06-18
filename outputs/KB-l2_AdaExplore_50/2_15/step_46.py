import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_transpose3d_kernel(
    x_ptr,        # [N, IC, D_in, H_in, W_in]
    w_ptr,        # [IC, OC, KD, KH, KW]
    bias_ptr,     # [OC]
    out_ptr,      # [N, OC, D_out, H_out, W_out]
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    IC_C: tl.constexpr,
    KD_C: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    HW_out = H_out * W_out
    DHW_out = D_out * HW_out

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < DHW_out

    d_o = sp_offs // HW_out
    rem = sp_offs % HW_out
    h_o = rem // W_out
    w_o = rem % W_out

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load bias
    bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32) + bias[None, :]

    # For each output position, find input positions:
    # d_in * SD - PD + kd = d_o => d_in = (d_o + PD - kd) / SD
    # need (d_o + PD - kd) % SD == 0 and 0 <= d_in < D_in

    for kd in tl.static_range(0, KD_C):
        d_in_num = d_o + PD - kd
        d_in = d_in_num // SD
        d_valid = (d_in_num % SD == 0) & (d_in >= 0) & (d_in < D_in)

        for kh in tl.static_range(0, KH_C):
            h_in_num = h_o + PH - kh
            h_in = h_in_num // SH
            h_valid = (h_in_num % SH == 0) & (h_in >= 0) & (h_in < H_in)

            for kw in tl.static_range(0, KW_C):
                w_in_num = w_o + PW - kw
                w_in = w_in_num // SW
                w_valid = (w_in_num % SW == 0) & (w_in >= 0) & (w_in < W_in)

                valid = d_valid & h_valid & w_valid & sp_mask

                # Load x[N, IC, d_in, h_in, w_in] for all IC
                # Compute base x address
                x_base = (
                    pid_n * IC * D_in * H_in * W_in
                    + d_in * H_in * W_in
                    + h_in * W_in
                    + w_in
                )
                ic_offs = tl.arange(0, IC_C)
                # x: [BLOCK_SP, IC_C]
                x_addrs = x_base[:, None] + ic_offs[None, :] * (D_in * H_in * W_in)
                x_mask = valid[:, None] & (ic_offs[None, :] < IC)
                x_vals = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

                # Load w[IC, OC, kd, kh, kw] -> [IC_C, BLOCK_OC]
                w_base = (kd * KH * KW + kh * KW + kw)
                w_addrs = (
                    ic_offs[:, None] * (OC * KD * KH * KW)
                    + oc_offs[None, :] * (KD * KH * KW)
                    + w_base
                )
                w_mask = (ic_offs[:, None] < IC) & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_addrs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals, allow_tf32=False)

    # Store output
    out_base = (
        pid_n * OC * DHW_out
        + oc_offs[None, :] * DHW_out
        + sp_offs[:, None]
    )
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_base, acc, mask=out_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, D_in, H_in, W_in = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride if isinstance(stride, int) else stride[0]
    PD = PH = PW = padding if isinstance(padding, int) else padding[0]

    D_out = (D_in - 1) * SD - 2 * PD + KD
    H_out = (H_in - 1) * SH - 2 * PH + KH
    W_out = (W_in - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(D_out * H_out * W_out, BLOCK_SP))

    _conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        IC_C=IC,
        KD_C=KD,
        KH_C=KH,
        KW_C=KW,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def _scale_mean_sub_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    row_offset = pid * S
    scale = tl.load(scale_ptr + c)

    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    mean = acc / S

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_offset + offs, scale * (vals - mean), mask=mask)


def scale_mean_sub(x, scale):
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    BLOCK_S = 1024
    _scale_mean_sub_kernel[grid](x, scale, out, N, C, S, BLOCK_S=BLOCK_S, num_warps=4, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        if self.conv_transpose.bias is not None:
            bias = self.conv_transpose.bias.contiguous()
        else:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)

        if self.training:
            bn = self.batch_norm
            with torch.no_grad():
                dims = (0, 2, 3, 4)
                batch_mean = y.mean(dim=dims)
                batch_var = y.var(dim=dims, unbiased=False)
                if bn.track_running_stats:
                    m = bn.momentum if bn.momentum is not None else 0.1
                    n_elems = y.numel() / y.size(1)
                    bn.running_mean.mul_(1 - m).add_(batch_mean, alpha=m)
                    unbiased_var = batch_var * (n_elems / (n_elems - 1)) if n_elems > 1 else batch_var
                    bn.running_var.mul_(1 - m).add_(unbiased_var, alpha=m)
                    if bn.num_batches_tracked is not None:
                        bn.num_batches_tracked.add_(1)
            scale = bn.weight / torch.sqrt(batch_var + bn.eps)
            return scale_mean_sub(y, scale.contiguous())
        else:
            bn = self.batch_norm
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            return scale_mean_sub(y, scale.contiguous())