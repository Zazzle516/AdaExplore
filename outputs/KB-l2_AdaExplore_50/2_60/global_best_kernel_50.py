import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128}, num_warps=8, num_stages=3),
    ],
    key=['N', 'IC', 'OC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv_transpose3d_swish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_spatial = tl.program_id(0)
    pid_owt = tl.program_id(1)

    oh = pid_spatial % OH
    tmp = pid_spatial // OH
    od = tmp % OD
    n = tmp // OD

    ow_offs = pid_owt * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, OC)

    acc = tl.zeros([OC, BLOCK_OW], dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_valid = (id_num >= 0) & ((id_num % STRIDE) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_val = ih_num // STRIDE
            ih_valid = (ih_num >= 0) & ((ih_num % STRIDE) == 0) & (ih_val >= 0) & (ih_val < IH)
            dh_valid = id_valid & ih_valid
            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PAD - kw
                iw_val = iw_num // STRIDE
                iw_valid = (iw_num >= 0) & ((iw_num % STRIDE) == 0) & (iw_val >= 0) & (iw_val < IW) & ow_mask
                if dh_valid:
                    for ic in tl.static_range(0, IC):
                        x_off = ((n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                        xval = tl.load(x_ptr + x_off, mask=iw_valid, other=0.0)
                        w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                        wval = tl.load(w_ptr + w_off)
                        acc += wval[:, None] * xval[None, :]

    if HAS_BIAS:
        bval = tl.load(b_ptr + oc_offs)
        acc += bval[:, None]

    sig = tl.sigmoid(acc)
    res = acc * sig

    out_off = ((n * OC + oc_offs[:, None]) * OD + od) * OH * OW + oh * OW + ow_offs[None, :]
    tl.store(out_ptr + out_off, res, mask=ow_mask[None, :])


# Stage 1: partial reductions for GroupNorm — split each (n, g) group across CHUNKS programs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['group_elems'],
)
@triton.jit
def gn_partial_reduce_kernel(
    x_ptr, partial_sum_ptr, partial_sumsq_ptr,
    N, C, G, C_per_G, S, group_elems, CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = pid % CHUNKS
    ng = pid // CHUNKS
    n = ng // G
    g = ng % G

    base = n * C * S + g * C_per_G * S

    # chunk range
    chunk_size = (group_elems + CHUNKS - 1) // CHUNKS
    start = chunk * chunk_size
    end = start + chunk_size
    if end > group_elems:
        end = group_elems

    sum_val = tl.zeros([1], dtype=tl.float32)
    sum_sq = tl.zeros([1], dtype=tl.float32)
    offs = tl.arange(0, BLOCK_SIZE)
    cur = start
    while cur < end:
        idx = cur + offs
        mask = idx < end
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        cur += BLOCK_SIZE

    out_idx = ng * CHUNKS + chunk
    tl.store(partial_sum_ptr + out_idx, tl.sum(sum_val, axis=0))
    tl.store(partial_sumsq_ptr + out_idx, tl.sum(sum_sq, axis=0))


# Stage 2: finalize mean/rstd by reducing the CHUNKS partials per (n, g)
@triton.jit
def gn_finalize_stats_kernel(
    partial_sum_ptr, partial_sumsq_ptr,
    mean_ptr, rstd_ptr,
    NG, group_elems, eps,
    CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= NG:
        return
    offs = tl.arange(0, CHUNKS)
    ps = tl.load(partial_sum_ptr + pid * CHUNKS + offs)
    pss = tl.load(partial_sumsq_ptr + pid * CHUNKS + offs)
    s = tl.sum(ps, axis=0)
    ssq = tl.sum(pss, axis=0)
    inv_n = 1.0 / group_elems
    mean = s * inv_n
    var = ssq * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


# Stage 3: normalize + affine + hardswish, using precomputed mean/rstd
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['group_elems'],
)
@triton.jit
def gn_normalize_hardswish_kernel(
    x_ptr, weight_ptr, bias_ptr, mean_ptr, rstd_ptr, out_ptr,
    N, C, G, C_per_G, S, group_elems, CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = pid % CHUNKS
    ng = pid // CHUNKS
    n = ng // G
    g = ng % G

    base = n * C * S + g * C_per_G * S
    mean = tl.load(mean_ptr + ng)
    rstd = tl.load(rstd_ptr + ng)

    chunk_size = (group_elems + CHUNKS - 1) // CHUNKS
    start = chunk * chunk_size
    end = start + chunk_size
    if end > group_elems:
        end = group_elems

    offs = tl.arange(0, BLOCK_SIZE)
    cur = start
    while cur < end:
        idx = cur + offs
        mask = idx < end
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        c_in_g = idx // S
        c_global = g * C_per_G + c_in_g
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)
        scale = rstd * w
        shift = b - mean * scale
        y = x * scale + shift
        t = y + 3.0
        t = tl.minimum(tl.maximum(t, 0.0), 6.0)
        out = y * t * 0.16666666666666666
        tl.store(out_ptr + base + idx, out, mask=mask)
        cur += BLOCK_SIZE


def triton_group_norm_hardswish(x, weight, bias, G, eps):
    x = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    C_per_G = C // G
    group_elems = C_per_G * S
    NG = N * G

    CHUNKS = 32

    partial_sum = torch.empty((NG, CHUNKS), device=x.device, dtype=torch.float32)
    partial_sumsq = torch.empty((NG, CHUNKS), device=x.device, dtype=torch.float32)
    mean = torch.empty((NG,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((NG,), device=x.device, dtype=torch.float32)
    out = torch.empty_like(x)

    grid1 = (NG * CHUNKS,)
    gn_partial_reduce_kernel[grid1](
        x, partial_sum, partial_sumsq,
        N, C, G, C_per_G, S, group_elems,
        CHUNKS=CHUNKS,
    )

    grid2 = (NG,)
    gn_finalize_stats_kernel[grid2](
        partial_sum, partial_sumsq, mean, rstd,
        NG, group_elems, eps,
        CHUNKS=CHUNKS,
    )

    grid3 = (NG * CHUNKS,)
    gn_normalize_hardswish_kernel[grid3](
        x, weight, bias, mean, rstd, out,
        N, C, G, C_per_G, S, group_elems,
        CHUNKS=CHUNKS,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        STRIDE = self.stride
        PAD = self.padding
        OC = self.out_channels
        OD = (ID - 1) * STRIDE - 2 * PAD + KD
        OH = (IH - 1) * STRIDE - 2 * PAD + KH
        OW = (IW - 1) * STRIDE - 2 * PAD + KW

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous() if self.has_bias else torch.empty(1, device=x.device, dtype=x.dtype)

        grid = lambda META: (N * OD * OH, triton.cdiv(OW, META['BLOCK_OW']))
        conv_transpose3d_swish_kernel[grid](
            x, w, b, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            STRIDE, PAD,
            HAS_BIAS=self.has_bias,
        )

        out = triton_group_norm_hardswish(
            out,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return out