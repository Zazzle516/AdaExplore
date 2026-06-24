import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_kernel(
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
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    sp_block = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = sp_block * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ic_offs = tl.arange(0, BLOCK_IC)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OD * OH * OW)
    ic_mask = ic_offs < IC

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Hoisted strides
    x_n_off = n * (IC * ID * IH * IW)
    KHW = KH * KW
    KDHW = KD * KHW
    IHIW = IH * IW
    IDIHIW = ID * IHIW
    OCKDHW = OC * KDHW

    # Hoisted oc * KDHW
    oc_w_base = oc_offs * KDHW  # [BLOCK_OC]

    for kd in range(KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in range(KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in range(KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val >= 0) & (iw_val < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask
                in_spatial_idx = id_val * IHIW + ih_val * IW + iw_val
                w_kern_idx = kd * KHW + kh * KW + kw

                # x_tile: [BLOCK_IC, BLOCK_SP]
                x_idx = x_n_off + ic_offs[:, None] * IDIHIW + in_spatial_idx[None, :]
                x_m = ic_mask[:, None] & spatial_valid[None, :]
                x_tile = tl.load(x_ptr + x_idx, mask=x_m, other=0.0)

                # w_tile: [BLOCK_OC, BLOCK_IC]
                w_idx = ic_offs[None, :] * OCKDHW + oc_w_base[:, None] + w_kern_idx
                w_m = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptr + w_idx, mask=w_m, other=0.0)

                acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    acc += bias[:, None]

    out_idx = n * (OC * OD * OH * OW) + oc_offs[:, None] * (OD * OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w
    SD, SH, SW = stride
    PD, PH, PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    SP = OD * OH * OW
    # Round IC up to next power of two for tl.dot K dim
    BLOCK_IC = 1
    while BLOCK_IC < IC:
        BLOCK_IC *= 2
    if BLOCK_IC < 16:
        BLOCK_IC = 16
    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(SP, META['BLOCK_SP']))

    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_IC=BLOCK_IC,
    )
    return out


@triton.jit
def bn_meansub_eval_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, SP,
    BLOCK_SP: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    base = n * C * SP + c * SP

    sum_val = 0.0
    for off in range(0, SP, BLOCK_SP):
        idx = off + tl.arange(0, BLOCK_SP)
        mask = idx < SP
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift
        sum_val += tl.sum(tl.where(mask, y, 0.0), axis=0)

    mean = sum_val / SP

    for off in range(0, SP, BLOCK_SP):
        idx = off + tl.arange(0, BLOCK_SP)
        mask = idx < SP
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = x * scale + shift - mean
        tl.store(out_ptr + base + idx, y, mask=mask)


def bn_meansub_eval_triton(x, scale, shift):
    N, C, D, H, W = x.shape
    SP = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    BLOCK_SP = 1024
    bn_meansub_eval_kernel[grid](
        x, out, scale, shift,
        N, C, SP,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
    )
    return out


@triton.jit
def per_nc_stats_kernel(
    x_ptr, sum_ptr, sumsq_ptr,
    N, C, SP,
    BLOCK_SP: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * SP + c * SP

    s = 0.0
    ss = 0.0
    for off in range(0, SP, BLOCK_SP):
        idx = off + tl.arange(0, BLOCK_SP)
        mask = idx < SP
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        s += tl.sum(tl.where(mask, x, 0.0), axis=0)
        ss += tl.sum(tl.where(mask, x * x, 0.0), axis=0)

    tl.store(sum_ptr + n * C + c, s)
    tl.store(sumsq_ptr + n * C + c, ss)


@triton.jit
def apply_meansub_scale_kernel(
    x_ptr, out_ptr,
    spatial_mean_ptr, scale_ptr,
    N, C, SP,
    BLOCK_SP: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    sm = tl.load(spatial_mean_ptr + n * C + c)
    sc = tl.load(scale_ptr + c)

    base = n * C * SP + c * SP

    for off in range(0, SP, BLOCK_SP):
        idx = off + tl.arange(0, BLOCK_SP)
        mask = idx < SP
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = (x - sm) * sc
        tl.store(out_ptr + base + idx, y, mask=mask)


def bn_meansub_train_triton(x, bn):
    """
    Fused BN(train) + spatial mean subtraction.
    Output = (x - spatial_mean[n,c]) * (gamma[c] / sqrt(batch_var[c] + eps))
    Also updates bn.running_mean and bn.running_var.
    """
    N, C, D, H, W = x.shape
    SP = D * H * W

    sum_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)
    sumsq_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)

    grid = (N * C,)
    BLOCK_SP = 1024
    per_nc_stats_kernel[grid](
        x, sum_nc, sumsq_nc,
        N, C, SP,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
    )

    spatial_mean = sum_nc / float(SP)  # [N, C]

    batch_sum = sum_nc.sum(dim=0)  # [C]
    batch_sumsq = sumsq_nc.sum(dim=0)  # [C]
    M = float(N * SP)
    batch_mean = batch_sum / M
    batch_var = batch_sumsq / M - batch_mean * batch_mean

    eps = bn.eps
    gamma = bn.weight if bn.weight is not None else torch.ones(C, device=x.device, dtype=x.dtype)
    scale = (gamma / torch.sqrt(batch_var + eps)).contiguous()

    # Update running stats (BN training behavior)
    if bn.running_mean is not None and bn.running_var is not None:
        momentum = bn.momentum if bn.momentum is not None else 0.1
        with torch.no_grad():
            # unbiased var for running_var
            if M > 1:
                unbiased_var = batch_var * (M / (M - 1))
            else:
                unbiased_var = batch_var
            bn.running_mean.mul_(1 - momentum).add_(batch_mean.detach(), alpha=momentum)
            bn.running_var.mul_(1 - momentum).add_(unbiased_var.detach(), alpha=momentum)
            if bn.num_batches_tracked is not None:
                bn.num_batches_tracked.add_(1)

    out = torch.empty_like(x)
    apply_meansub_scale_kernel[grid](
        x, out,
        spatial_mean.contiguous(), scale,
        N, C, SP,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, x):
        x = x.cuda().contiguous().float()
        weight = self.conv_transpose.weight.cuda().contiguous().float()
        if self.conv_transpose.bias is not None:
            bias = self.conv_transpose.bias.cuda().contiguous().float()
        else:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=torch.float32)

        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)

        bn = self.batch_norm
        if bn.training:
            y = bn_meansub_train_triton(y, bn)
            return y
        else:
            running_mean = bn.running_mean
            running_var = bn.running_var
            eps = bn.eps
            w = bn.weight if bn.weight is not None else torch.ones_like(running_mean)
            b = bn.bias if bn.bias is not None else torch.zeros_like(running_mean)
            invstd = torch.rsqrt(running_var + eps)
            scale = (w * invstd).contiguous()
            shift = (b - running_mean * w * invstd).contiguous()
            y = bn_meansub_eval_triton(y, scale, shift)
            return y