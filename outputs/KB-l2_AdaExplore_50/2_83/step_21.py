import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SP': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_SP': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    OHW = OH * OW
    ODHW = OD * OHW
    IHW = IH * IW
    IDHW = ID * IHW
    KHW = KH * KW
    KDHW = KD * KHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < ODHW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    x_n_base = pid_n * IC * IDHW
    # spatial base offset in input for each output position (with kd=kh=kw=0)
    sp_in_base = od * IHW + oh * IW + ow

    w_oc_base = oc_offs * (IC * KDHW)

    for ic in range(0, IC):
        x_ic_base = x_n_base + ic * IDHW
        w_ic_base = w_oc_base + ic * KDHW
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    tap_off = kd * IHW + kh * IW + kw
                    x_off = x_ic_base + sp_in_base + tap_off
                    x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)
                    w_off = w_ic_base + kd * KHW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += w_val[:, None] * x_val[None, :]

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[:, None]

    out_base = pid_n * OC * ODHW
    out_off = out_base + oc_offs[:, None] * ODHW + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


def conv3d_triton(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    ODHW = OD * OH * OW
    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(ODHW, meta['BLOCK_SP']))
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
    )
    return out


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    C, S,
    channels_per_group,
    min_value: tl.constexpr, max_value: tl.constexpr, eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid encodes (n * groups + g)
    # base offset
    base = pid * GROUP_SIZE

    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, GROUP_SIZE, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < GROUP_SIZE
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    g = pid % (C // channels_per_group)
    c_base = g * channels_per_group

    for off in range(0, GROUP_SIZE, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < GROUP_SIZE
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        c_in_group = idx // S
        c = c_base + c_in_group
        w = tl.load(weight_ptr + c, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c, mask=mask, other=0.0)
        y = (x - mean) * rstd * w + b
        y = tl.maximum(tl.minimum(y, min_value), min_value)
        tl.store(out_ptr + base + idx, y, mask=mask)


def fused_gn_min_clamp(x, weight, bias, groups, min_value, max_value, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    channels_per_group = C // groups
    x = x.contiguous()
    out = torch.empty_like(x)
    GROUP_SIZE = channels_per_group * S
    BLOCK_SIZE = 2048
    grid = (N * groups,)
    fused_gn_min_clamp_kernel[grid](
        x, out, weight, bias,
        C, S, channels_per_group,
        float(min_value), float(max_value), float(eps),
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = x.contiguous()
        x = conv3d_triton(x, self.conv.weight, self.conv.bias)
        x = fused_gn_min_clamp(
            x,
            self.norm.weight,
            self.norm.bias,
            self.groups,
            self.min_value,
            self.max_value,
            self.norm.eps,
        )
        x = self.dropout(x)
        return x