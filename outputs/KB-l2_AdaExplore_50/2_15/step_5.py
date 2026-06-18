import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _convt3d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    OS = OD * OHW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < OS
    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_SP, BLOCK_OC], dtype=tl.float32)

    # Add positions od+PD, oh+PH, ow+PW. id_num = od+PD - kd
    # iterate kd, kh, kw
    for kd in range(0, KD):
        id_num = od + PD - kd  # [BLOCK_SP]
        id_q = id_num // SD
        id_r = id_num - id_q * SD
        d_ok = (id_r == 0) & (id_q >= 0) & (id_q < ID)
        for kh in range(0, KH):
            ih_num = oh + PH - kh
            ih_q = ih_num // SH
            ih_r = ih_num - ih_q * SH
            h_ok = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in range(0, KW):
                iw_num = ow + PW - kw
                iw_q = iw_num // SW
                iw_r = iw_num - iw_q * SW
                w_ok = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                spatial_ok = d_ok & h_ok & w_ok & sp_mask  # [BLOCK_SP]

                # input index: ((n*IC + ic)*ID + id_q)*IH*IW + ih_q*IW + iw_q
                base_in_sp = id_q * (IH * IW) + ih_q * IW + iw_q  # [BLOCK_SP]

                # Loop over IC in tiles
                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    # x_ptrs : [BLOCK_SP, BLOCK_IC]
                    x_off = (pid_n * IC + ic_offs[None, :]) * (ID * IH * IW) + base_in_sp[:, None]
                    x_load_mask = spatial_ok[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)

                    # w shape: (IC, OC, KD, KH, KW) -> index ic, oc, kd, kh, kw
                    w_off = (ic_offs[:, None] * OC + oc_offs[None, :]) * (KD * KH * KW) + (kd * KH * KW + kh * KW + kw)
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals)

    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
        acc += bias[None, :]

    # Store: out shape [N, OC, OS], we want layout [N, OC, OD, OH, OW]
    # Output index = ((n*OC + oc)*OS) + sp
    out_off = (pid_n * OC + oc_offs[None, :]) * OS + sp_offs[:, None]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=store_mask)


def convt3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w
    SD, SH, SW = stride if isinstance(stride, tuple) else (stride, stride, stride)
    PD, PH, PW = padding if isinstance(padding, tuple) else (padding, padding, padding)

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 64
    BLOCK_IC = 16

    OS = OD * OH * OW
    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OS, BLOCK_SP))

    _convt3d_gather_kernel[grid](
        x, weight, bias if bias is not None else x,  # placeholder; we'll handle None below
        out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP, BLOCK_IC=BLOCK_IC,
        num_warps=4, num_stages=2,
    )
    return out


# Better: handle bias None properly
@triton.jit
def _convt3d_gather_kernel_nb(
    x_ptr, w_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    OS = OD * OHW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < OS
    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_SP, BLOCK_OC], dtype=tl.float32)

    for kd in range(0, KD):
        id_num = od + PD - kd
        id_q = id_num // SD
        id_r = id_num - id_q * SD
        d_ok = (id_r == 0) & (id_q >= 0) & (id_q < ID)
        for kh in range(0, KH):
            ih_num = oh + PH - kh
            ih_q = ih_num // SH
            ih_r = ih_num - ih_q * SH
            h_ok = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in range(0, KW):
                iw_num = ow + PW - kw
                iw_q = iw_num // SW
                iw_r = iw_num - iw_q * SW
                w_ok = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                spatial_ok = d_ok & h_ok & w_ok & sp_mask

                base_in_sp = id_q * (IH * IW) + ih_q * IW + iw_q

                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    x_off = (pid_n * IC + ic_offs[None, :]) * (ID * IH * IW) + base_in_sp[:, None]
                    x_load_mask = spatial_ok[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)

                    w_off = (ic_offs[:, None] * OC + oc_offs[None, :]) * (KD * KH * KW) + (kd * KH * KW + kh * KW + kw)
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals)

    out_off = (pid_n * OC + oc_offs[None, :]) * OS + sp_offs[:, None]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=store_mask)


def convt3d_triton_v2(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD, SH, SW = stride if isinstance(stride, tuple) else (stride, stride, stride)
    PD, PH, PW = padding if isinstance(padding, tuple) else (padding, padding, padding)

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 64
    BLOCK_IC = 16

    OS = OD * OH * OW
    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OS, BLOCK_SP))

    _convt3d_gather_kernel_nb[grid](
        x, weight, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP, BLOCK_IC=BLOCK_IC,
        num_warps=4, num_stages=2,
    )

    if bias is not None:
        out = out + bias.view(1, OC, 1, 1, 1)

    return out


@triton.jit
def _per_nc_reduce_kernel(
    x_ptr, sum_ptr, sumsq_ptr,
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    row_off = pid * S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        acc += vals
        acc_sq += vals * vals
    s = tl.sum(acc, axis=0)
    sq = tl.sum(acc_sq, axis=0)
    tl.store(sum_ptr + pid, s)
    tl.store(sumsq_ptr + pid, sq)


@triton.jit
def _fused_epilogue_kernel(
    x_ptr, out_ptr,
    sum_nc_ptr,
    scale_ptr,
    bias_ptr,
    N, C, S,
    inv_S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    row_off = pid * S

    s_nc = tl.load(sum_nc_ptr + pid)
    m_nc = s_nc * inv_S
    scale = tl.load(scale_ptr + c)
    bshift = tl.load(bias_ptr + c)

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0)
        # bn(x) - mean( bn(x) ) = scale*(x - m_nc) since constant shift cancels
        out = scale * (vals - m_nc)
        tl.store(out_ptr + row_off + offs, out, mask=mask)


def fused_bn_subtract_mean(x, gamma, beta, running_mean, running_var,
                           training, momentum, eps):
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    M = N * S

    sum_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)
    sumsq_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)

    BLOCK_S = 1024
    grid = (N * C,)
    _per_nc_reduce_kernel[grid](x, sum_nc, sumsq_nc, S, BLOCK_S=BLOCK_S, num_warps=4)

    if training:
        sum_c = sum_nc.sum(dim=0)
        sumsq_c = sumsq_nc.sum(dim=0)
        mu_c = sum_c / M
        var_c = sumsq_c / M - mu_c * mu_c
        with torch.no_grad():
            if running_mean is not None:
                running_mean.mul_(1 - momentum).add_(mu_c, alpha=momentum)
            if running_var is not None and M > 1:
                var_unbiased = var_c * (M / (M - 1))
                running_var.mul_(1 - momentum).add_(var_unbiased, alpha=momentum)
        scale_c = gamma / torch.sqrt(var_c + eps)
    else:
        scale_c = gamma / torch.sqrt(running_var + eps)

    scale_c = scale_c.contiguous().to(x.dtype)
    bias_c = torch.zeros_like(scale_c)  # not used since constant cancels
    out = torch.empty_like(x)
    inv_S = 1.0 / S
    _fused_epilogue_kernel[grid](
        x, out, sum_nc, scale_c, bias_c, N, C, S, inv_S,
        BLOCK_S=BLOCK_S, num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias
        x = convt3d_triton_v2(x, weight, bias, self.stride, self.padding)
        x = x.contiguous()
        bn = self.batch_norm
        x = fused_bn_subtract_mean(
            x,
            bn.weight, bn.bias,
            bn.running_mean, bn.running_var,
            bn.training, bn.momentum, bn.eps,
        )
        return x