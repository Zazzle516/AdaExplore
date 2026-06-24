import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Gather-based ConvTranspose3d using tl.dot to leverage tensor cores.
# Each program computes [BLOCK_SP, BLOCK_OC] for one (n, oc_block, sp_block).
# We loop over (kd, kh, kw) and inside each iteration build an
# [BLOCK_SP, IC] tile of input values (gathered) and an [IC, BLOCK_OC] tile of
# weight values, then accumulate via tl.dot.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC: tl.constexpr, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ic_offs = tl.arange(0, IC)

    DHW = OD * OH * OW
    HW = OH * OW

    od = sp_offs // HW
    rem = sp_offs % HW
    oh = rem // OW
    ow = rem % OW

    sp_mask = sp_offs < DHW
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    x_base_n = pid_n * IC * ID * IH * IW
    IDHW = ID * IH * IW
    IHW = IH * IW

    WOC = OC * KD * KH * KW
    WKDHW = KD * KH * KW
    WKHW = KH * KW

    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_ = id_num // SD
        id_valid = (id_num >= 0) & (id_num % SD == 0) & (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_ = ih_num // SH
            ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih_ >= 0) & (ih_ < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_ = iw_num // SW
                iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw_ >= 0) & (iw_ < IW)

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

                # x_tile: [BLOCK_SP, IC]
                x_off = (x_base_n
                         + ic_offs[None, :] * IDHW
                         + id_[:, None] * IHW
                         + ih_[:, None] * IW
                         + iw_[:, None])
                x_tile = tl.load(x_ptr + x_off, mask=spatial_valid[:, None], other=0.0)

                # w_tile: [IC, BLOCK_OC]
                # w layout: [IC, OC, KD, KH, KW]
                w_off = (ic_offs[:, None] * WOC
                         + oc_offs[None, :] * WKDHW
                         + kd * WKHW + kh * KW + kw)
                w_tile = tl.load(w_ptr + w_off, mask=oc_mask[None, :], other=0.0)

                acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    # store
    out_off = (pid_n * OC * DHW
               + oc_offs[None, :] * DHW
               + sp_offs[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose3d_triton(x, w, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC2, OC, KD, KH, KW = w.shape
    assert IC == IC2
    SD, SH, SW = stride, stride, stride
    PD, PH, PW = padding, padding, padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OD * OH * OW, META['BLOCK_SP']))

    conv_transpose3d_kernel[grid](
        x, w, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
    )
    return out


# Fused BN(train) + bias add + subtract spatial mean.
# y_conv has no bias yet. Add bias, then compute per-channel mean/var across (N,D,H,W).
# Final = scale * (x_with_bias - spatial_mean_per_NC(x_with_bias))
#       = scale * (x - spatial_mean(x))   [bias is constant across spatial -> cancels]
# So we don't even need to add bias for the final output! But we still need var which
# depends on the conv output (bias just shifts; var(x+b)=var(x)). So bias add can be
# skipped entirely for the final result.

@triton.jit
def spatial_mean_kernel(
    x_ptr, mean_ptr,
    N, C, SP,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = n * C * SP + c * SP

    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, SP, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SP
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        s += v
    m = tl.sum(s) / SP
    tl.store(mean_ptr + pid, m)


@triton.jit
def apply_scale_submean_kernel(
    x_ptr, scale_ptr, mean_ptr, out_ptr,
    N, C, SP,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = tl.program_id(1)
    n = pid // C
    c = pid % C
    base = n * C * SP + c * SP

    scale = tl.load(scale_ptr + c)
    mean = tl.load(mean_ptr + pid)

    idx = blk * BLOCK + tl.arange(0, BLOCK)
    mask = idx < SP
    v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
    out = scale * (v - mean)
    tl.store(out_ptr + base + idx, out, mask=mask)


def fused_bn_submean(y, gamma, eps):
    N, C, D, H, W = y.shape
    SP = D * H * W
    out = torch.empty_like(y)

    # compute per-(N,C) spatial means
    means = torch.empty((N * C,), device=y.device, dtype=torch.float32)
    BLOCK1 = 1024
    spatial_mean_kernel[(N * C,)](
        y, means, N, C, SP, BLOCK=BLOCK1, num_warps=4, num_stages=2,
    )

    # compute var per-channel using torch (small): need across N, D, H, W
    # var = mean(y^2) - mean(y)^2 ; channel mean = mean over n of means[n,c] (since spatial weighted equally)
    # Actually channel_mean = mean over (N,D,H,W) of y = mean over n of means[n,c]
    means_nc = means.view(N, C)
    ch_mean = means_nc.mean(dim=0)  # [C]
    # var = E[y^2] - E[y]^2. Compute E[y^2] via torch:
    y_sq_mean = (y * y).mean(dim=(0, 2, 3, 4))
    var = y_sq_mean - ch_mean * ch_mean
    scale = gamma / torch.sqrt(var + eps)

    BLOCK2 = 512
    grid = (N * C, triton.cdiv(SP, BLOCK2))
    apply_scale_submean_kernel[grid](
        y, scale.contiguous(), means, out,
        N, C, SP, BLOCK=BLOCK2, num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous()

        y = conv_transpose3d_triton(x, w, self.stride, self.padding)
        # bias cancels in final output; skip adding it.

        eps = self.batch_norm.eps
        gamma = self.batch_norm.weight

        out = fused_bn_submean(y, gamma, eps)
        return out