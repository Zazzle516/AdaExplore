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
    BLOCK_SPATIAL: tl.constexpr,
):
    # program ids: (n, oc, spatial_tile)
    n = tl.program_id(0)
    oc = tl.program_id(1)
    sp_id = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OHW

    sp_offs = sp_id * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    sp_mask = sp_offs < ODHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    # bias
    bias = tl.load(b_ptr + oc)
    acc = tl.zeros((BLOCK_SPATIAL,), dtype=tl.float32) + bias

    # For ConvTranspose3d with no padding/stride=1:
    # out[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, od-kd, oh-kh, ow-kw] * w[ic, oc, kd, kh, kw]
    # where the input indices must be in valid range
    for ic in range(0, IC):
        for kd in range(0, KD):
            id_ = od - kd
            id_valid = (id_ >= 0) & (id_ < ID)
            for kh in range(0, KH):
                ih_ = oh - kh
                ih_valid = (ih_ >= 0) & (ih_ < IH)
                for kw in range(0, KW):
                    iw_ = ow - kw
                    iw_valid = (iw_ >= 0) & (iw_ < IW)
                    valid = id_valid & ih_valid & iw_valid & sp_mask

                    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                    w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw

                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    out_off = ((n * OC + oc) * ODHW) + sp_offs
    tl.store(out_ptr + out_off, acc, mask=sp_mask)


@triton.jit
def fused_scale_bn_pool_kernel(
    out_ptr,  # input from conv_transpose
    scale_factor,
    bn_weight_ptr, bn_bias_ptr,
    bn_mean_ptr, bn_var_ptr,
    eps,
    final_out_ptr,
    N, OC, SPATIAL,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    oc = tl.program_id(1)

    # load BN params
    bn_w = tl.load(bn_weight_ptr + oc)
    bn_b = tl.load(bn_bias_ptr + oc)
    mean = tl.load(bn_mean_ptr + oc)
    var = tl.load(bn_var_ptr + oc)
    invstd = 1.0 / tl.sqrt(var + eps)

    # combined transform: y = ((x*scale - mean) * invstd) * bn_w + bn_b
    #                     = x * (scale * invstd * bn_w) + (-mean * invstd * bn_w + bn_b)
    a = scale_factor * invstd * bn_w
    b_term = -mean * invstd * bn_w + bn_b

    base = (n * OC + oc) * SPATIAL

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for start in range(0, SPATIAL, BLOCK_S):
        offs = start + tl.arange(0, BLOCK_S)
        mask = offs < SPATIAL
        x = tl.load(out_ptr + base + offs, mask=mask, other=0.0)
        y = x * a + b_term
        y = tl.where(mask, y, 0.0)
        acc += y

    total = tl.sum(acc, axis=0)
    result = total / SPATIAL
    tl.store(final_out_ptr + n * OC + oc, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.eps = float(eps)
        self.momentum = momentum

        # Match ConvTranspose3d default init
        ct = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(ct.weight.detach().clone())
        self.conv_bias = nn.Parameter(ct.bias.detach().clone())

        bn = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.bn_weight = nn.Parameter(bn.weight.detach().clone())
        self.bn_bias = nn.Parameter(bn.bias.detach().clone())
        self.register_buffer('bn_running_mean', bn.running_mean.clone())
        self.register_buffer('bn_running_var', bn.running_var.clone())

        self.KD = kernel_size
        self.KH = kernel_size
        self.KW = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels

        # Use cuDNN-backed conv_transpose3d for performance
        conv_out = F.conv_transpose3d(x, self.conv_weight, self.conv_bias)
        OD, OH, OW = conv_out.shape[2], conv_out.shape[3], conv_out.shape[4]

        # Training mode: need to update running stats and use batch stats
        if self.training:
            y = conv_out * self.scale_factor
            y = F.batch_norm(y, self.bn_running_mean, self.bn_running_var,
                             self.bn_weight, self.bn_bias, training=True,
                             momentum=self.momentum, eps=self.eps)
            y = F.adaptive_avg_pool3d(y, (1, 1, 1))
            return y

        # Eval mode: fused kernel
        final_out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)
        SPATIAL = OD * OH * OW
        BLOCK_S = 1024
        grid2 = (N, OC)
        fused_scale_bn_pool_kernel[grid2](
            conv_out, self.scale_factor,
            self.bn_weight, self.bn_bias,
            self.bn_running_mean, self.bn_running_var,
            self.eps,
            final_out,
            N, OC, SPATIAL,
            BLOCK_S=BLOCK_S,
            num_warps=8,
        )
        return final_out