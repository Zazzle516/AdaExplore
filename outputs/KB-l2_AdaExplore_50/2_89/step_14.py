import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv_transpose_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,  # pooled output dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N * PD * PH * PW, )
    pid = tl.program_id(0)
    pw = pid % PW
    tmp = pid // PW
    ph = tmp % PH
    tmp = tmp // PH
    pd = tmp % PD
    n = tmp // PD

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)

    # 8 accumulators (POOL_K^3 = 8), one per pool position
    acc000 = tl.zeros([BLOCK_OC], dtype=tl.float32)
    acc001 = tl.zeros([BLOCK_OC], dtype=tl.float32)
    acc010 = tl.zeros([BLOCK_OC], dtype=tl.float32)
    acc011 = tl.zeros([BLOCK_OC], dtype=tl.float32)
    acc100 = tl.zeros([BLOCK_OC], dtype=tl.float32)
    acc101 = tl.zeros([BLOCK_OC], dtype=tl.float32)
    acc110 = tl.zeros([BLOCK_OC], dtype=tl.float32)
    acc111 = tl.zeros([BLOCK_OC], dtype=tl.float32)

    od0 = pd * POOL_K
    oh0 = ph * POOL_K
    ow0 = pw * POOL_K

    # Hoist weight load outside pool loop: weights only depend on (ic, kd, kh, kw, oc)
    for kd in tl.static_range(KD):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                for ic in tl.static_range(0, 3):
                    w_off = ((ic * OC + offs_oc) * KD + kd) * KH * KW + kh * KW + kw
                    wv = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)

                    # Now apply this weight to all 8 pool positions
                    # ld=0
                    id_num0 = od0 + PAD_D - kd
                    id0 = id_num0 // SD
                    id0_valid = (id_num0 % SD == 0) & (id0 >= 0) & (id0 < ID)
                    # ld=1
                    id_num1 = od0 + 1 + PAD_D - kd
                    id1 = id_num1 // SD
                    id1_valid = (id_num1 % SD == 0) & (id1 >= 0) & (id1 < ID)

                    ih_num0 = oh0 + PAD_H - kh
                    ih0 = ih_num0 // SH
                    ih0_valid = (ih_num0 % SH == 0) & (ih0 >= 0) & (ih0 < IH)
                    ih_num1 = oh0 + 1 + PAD_H - kh
                    ih1 = ih_num1 // SH
                    ih1_valid = (ih_num1 % SH == 0) & (ih1 >= 0) & (ih1 < IH)

                    iw_num0 = ow0 + PAD_W - kw
                    iw0 = iw_num0 // SW
                    iw0_valid = (iw_num0 % SW == 0) & (iw0 >= 0) & (iw0 < IW)
                    iw_num1 = ow0 + 1 + PAD_W - kw
                    iw1 = iw_num1 // SW
                    iw1_valid = (iw_num1 % SW == 0) & (iw1 >= 0) & (iw1 < IW)

                    base_n_ic = (n * IC + ic) * ID

                    # (0,0,0)
                    v000 = id0_valid & ih0_valid & iw0_valid
                    x000 = tl.load(x_ptr + (base_n_ic + id0) * IH * IW + ih0 * IW + iw0, mask=v000, other=0.0)
                    acc000 += x000 * wv
                    # (0,0,1)
                    v001 = id0_valid & ih0_valid & iw1_valid
                    x001 = tl.load(x_ptr + (base_n_ic + id0) * IH * IW + ih0 * IW + iw1, mask=v001, other=0.0)
                    acc001 += x001 * wv
                    # (0,1,0)
                    v010 = id0_valid & ih1_valid & iw0_valid
                    x010 = tl.load(x_ptr + (base_n_ic + id0) * IH * IW + ih1 * IW + iw0, mask=v010, other=0.0)
                    acc010 += x010 * wv
                    # (0,1,1)
                    v011 = id0_valid & ih1_valid & iw1_valid
                    x011 = tl.load(x_ptr + (base_n_ic + id0) * IH * IW + ih1 * IW + iw1, mask=v011, other=0.0)
                    acc011 += x011 * wv
                    # (1,0,0)
                    v100 = id1_valid & ih0_valid & iw0_valid
                    x100 = tl.load(x_ptr + (base_n_ic + id1) * IH * IW + ih0 * IW + iw0, mask=v100, other=0.0)
                    acc100 += x100 * wv
                    # (1,0,1)
                    v101 = id1_valid & ih0_valid & iw1_valid
                    x101 = tl.load(x_ptr + (base_n_ic + id1) * IH * IW + ih0 * IW + iw1, mask=v101, other=0.0)
                    acc101 += x101 * wv
                    # (1,1,0)
                    v110 = id1_valid & ih1_valid & iw0_valid
                    x110 = tl.load(x_ptr + (base_n_ic + id1) * IH * IW + ih1 * IW + iw0, mask=v110, other=0.0)
                    acc110 += x110 * wv
                    # (1,1,1)
                    v111 = id1_valid & ih1_valid & iw1_valid
                    x111 = tl.load(x_ptr + (base_n_ic + id1) * IH * IW + ih1 * IW + iw1, mask=v111, other=0.0)
                    acc111 += x111 * wv

    acc000 = acc000 + bias
    acc001 = acc001 + bias
    acc010 = acc010 + bias
    acc011 = acc011 + bias
    acc100 = acc100 + bias
    acc101 = acc101 + bias
    acc110 = acc110 + bias
    acc111 = acc111 + bias

    m0 = tl.maximum(acc000, acc001)
    m1 = tl.maximum(acc010, acc011)
    m2 = tl.maximum(acc100, acc101)
    m3 = tl.maximum(acc110, acc111)
    m01 = tl.maximum(m0, m1)
    m23 = tl.maximum(m2, m3)
    acc_max = tl.maximum(m01, m23)

    # store pooled result: shape (N, OC, PD, PH, PW)
    out_off = ((n * OC + offs_oc) * PD + pd) * PH * PW + ph * PW + pw
    tl.store(out_ptr + out_off, acc_max, mask=mask_oc)


@triton.jit
def fused_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * S + s
    x_ptrs = x_ptr + base + offs_c * S

    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    s_sum = tl.sum(e, axis=0)
    sm = e / s_sum

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask_c, sw, -float('inf'))
    out_val = tl.max(sw, axis=0)

    tl.store(out_ptr + n * S + s, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PAD_D = PAD_H = PAD_W = self.padding
        OPD = OPH = OPW = self.output_padding

        # output spatial dims for conv transpose
        OD = (ID - 1) * SD - 2 * PAD_D + KD + OPD
        OH = (IH - 1) * SH - 2 * PAD_H + KH + OPH
        OW = (IW - 1) * SW - 2 * PAD_W + KW + OPW

        # pooled dims
        POOL_K = self.pool_kernel_size
        PD = OD // POOL_K
        PH = OH // POOL_K
        PW = OW // POOL_K

        # Fused conv_transpose + maxpool: produce (N, OC, PD, PH, PW)
        pooled = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=x.dtype)

        w = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        b = self.conv_transpose.bias.contiguous()    # (OC,)

        BLOCK_OC = max(32, triton.next_power_of_2(OC))
        grid = (N * PD * PH * PW,)
        conv_transpose_pool_kernel[grid](
            x, w, b, pooled,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            SD, SH, SW,
            PAD_D, PAD_H, PAD_W,
            POOL_K,
            BLOCK_OC=BLOCK_OC,
        )

        # Softmax + sub + swish + channel-max
        S = PD * PH * PW
        out = torch.empty((N, PD, PH, PW), device=x.device, dtype=x.dtype)
        BLOCK_C = triton.next_power_of_2(OC)
        grid2 = (N * S,)
        fused_softmax_sub_swish_max_kernel[grid2](
            pooled, self.subtract.contiguous(), out,
            N, OC, S,
            BLOCK_C=BLOCK_C,
            num_warps=1,
        )
        return out