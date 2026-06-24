import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    total_out,
    BLOCK: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_out

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    oc = tmp % OC
    n = tmp // OC

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    ic_offs = tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
    ic_mask = ic_offs < IC  # [BLOCK_IC]

    IHIW = IH * IW
    IDIHIW = ID * IHIW
    KHKW = KH * KW
    KDKHKW = KD * KHKW
    OC_KDKHKW = OC * KDKHKW

    # x base per output element (n contribution)
    n_base = n * IC * IDIHIW  # [BLOCK]
    # weight base per output element (oc contribution)
    oc_base = oc * KDKHKW  # [BLOCK]

    for kd in range(0, KD):
        id_num = od + PD - kd
        id_ = id_num // SD
        id_valid = (id_num % SD == 0) & (id_ >= 0) & (id_ < ID)
        for kh in range(0, KH):
            ih_num = oh + PH - kh
            ih = ih_num // SH
            ih_valid = (ih_num % SH == 0) & (ih >= 0) & (ih < IH)
            for kw in range(0, KW):
                iw_num = ow + PW - kw
                iw = iw_num // SW
                iw_valid = (iw_num % SW == 0) & (iw >= 0) & (iw < IW)
                valid = id_valid & ih_valid & iw_valid & mask  # [BLOCK]

                # x_idx[block, ic] = n_base + ic*IDIHIW + id_*IHIW + ih*IW + iw
                spatial_in = id_ * IHIW + ih * IW + iw  # [BLOCK]
                x_base = n_base + spatial_in  # [BLOCK]
                x_idx = x_base[:, None] + ic_offs[None, :] * IDIHIW  # [BLOCK, BLOCK_IC]

                # w_idx[block, ic] = ic*OC_KDKHKW + oc*KDKHKW + kd*KHKW + kh*KW + kw
                k_off = kd * KHKW + kh * KW + kw
                w_base = oc_base + k_off  # [BLOCK]
                w_idx = w_base[:, None] + ic_offs[None, :] * OC_KDKHKW  # [BLOCK, BLOCK_IC]

                load_mask = valid[:, None] & ic_mask[None, :]
                xv = tl.load(x_ptr + x_idx, mask=load_mask, other=0.0)
                wv = tl.load(w_ptr + w_idx, mask=load_mask, other=0.0)
                acc += tl.sum(xv * wv, axis=1)

    bv = tl.load(b_ptr + oc, mask=mask, other=0.0)
    acc += bv
    tl.store(out_ptr + offs, acc, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD, SH, SW = stride, stride, stride
    PD, PH, PW = padding, padding, padding
    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
    total = N * OC * OD * OH * OW
    BLOCK = 256
    # BLOCK_IC rounded up to next power of two >= IC
    BLOCK_IC = 1
    while BLOCK_IC < IC:
        BLOCK_IC *= 2
    grid = (triton.cdiv(total, BLOCK),)
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        total,
        BLOCK=BLOCK,
        BLOCK_IC=BLOCK_IC,
        num_warps=8,
        num_stages=2,
    )
    return out


@triton.jit
def bn_meansub_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, SPATIAL,
    BLOCK: tl.constexpr,
):
    # one program per (n, c) - compute mean of (BN(x)) then subtract
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    base = (n * C + c) * SPATIAL

    # first pass: compute sum of normalized
    sum_val = tl.zeros([], dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < SPATIAL
        v = tl.load(x_ptr + base + idx, mask=m, other=0.0)
        nv = v * scale + shift
        sum_val += tl.sum(tl.where(m, nv, 0.0))

    mean = sum_val / SPATIAL

    # second pass: write out
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < SPATIAL
        v = tl.load(x_ptr + base + idx, mask=m, other=0.0)
        nv = v * scale + shift
        res = nv - mean
        tl.store(out_ptr + base + idx, res, mask=m)


def bn_meansub_triton(x, scale, shift):
    N, C, D, H, W = x.shape
    SPATIAL = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    BLOCK = 1024
    bn_meansub_kernel[grid](
        x, out, scale, shift,
        N, C, SPATIAL,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        if self.conv_transpose.bias is not None:
            bias = self.conv_transpose.bias.contiguous()
        else:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)

        # BatchNorm3d - in training mode it computes batch stats. Replicate behavior.
        # Use F.batch_norm to update running stats and get normalized output, but
        # we want to fuse normalization + mean-subtraction. Use a two-step:
        # 1) compute batch mean/var per-channel, update running stats
        # 2) normalize via fused kernel
        if self.training:
            # compute per-channel mean and var over (N, D, H, W)
            dims = (0, 2, 3, 4)
            mean = y.mean(dim=dims)
            var = y.var(dim=dims, unbiased=False)
            with torch.no_grad():
                momentum = self.batch_norm.momentum
                self.batch_norm.running_mean.mul_(1 - momentum).add_(mean.detach(), alpha=momentum)
                # unbiased var for running stats
                N = y.shape[0] * y.shape[2] * y.shape[3] * y.shape[4]
                unbiased_var = var.detach() * N / (N - 1) if N > 1 else var.detach()
                self.batch_norm.running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
                self.batch_norm.num_batches_tracked.add_(1)
        else:
            mean = self.batch_norm.running_mean
            var = self.batch_norm.running_var

        eps = self.batch_norm.eps
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        invstd = torch.rsqrt(var + eps)
        scale = (gamma * invstd).contiguous()
        shift = (beta - mean * gamma * invstd).contiguous()

        out = bn_meansub_triton(y, scale, shift)
        return out