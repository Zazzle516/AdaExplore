import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_scale_bn_pool_kernel(
    out_ptr,
    bn_weight_ptr, bn_bias_ptr,
    bn_mean_ptr, bn_var_ptr,
    final_out_ptr,
    scale_factor,
    eps,
    N, OC, SPATIAL,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    oc = tl.program_id(1)

    bn_w = tl.load(bn_weight_ptr + oc)
    bn_b = tl.load(bn_bias_ptr + oc)
    mean = tl.load(bn_mean_ptr + oc)
    var = tl.load(bn_var_ptr + oc)
    invstd = 1.0 / tl.sqrt(var + eps)

    a = scale_factor * invstd * bn_w
    b_term = -mean * invstd * bn_w + bn_b

    base = (n * OC + oc) * SPATIAL

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for start in range(0, SPATIAL, BLOCK_S):
        offs = start + tl.arange(0, BLOCK_S)
        mask = offs < SPATIAL
        x = tl.load(out_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.where(mask, x, 0.0)

    total = tl.sum(acc, axis=0)
    mean_x = total / SPATIAL
    result = mean_x * a + b_term
    tl.store(final_out_ptr + n * OC + oc, result)


@triton.jit
def channel_sum_kernel(
    x_ptr,
    sum_ptr, sumsq_ptr,
    SPATIAL,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    oc = tl.program_id(1)
    OC = tl.num_programs(1)

    base = (n * OC + oc) * SPATIAL

    s = tl.zeros((BLOCK_S,), dtype=tl.float32)
    sq = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for start in range(0, SPATIAL, BLOCK_S):
        offs = start + tl.arange(0, BLOCK_S)
        mask = offs < SPATIAL
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        xv = tl.where(mask, x, 0.0)
        s += xv
        sq += xv * xv

    sum_val = tl.sum(s, axis=0)
    sumsq_val = tl.sum(sq, axis=0)
    tl.store(sum_ptr + n * OC + oc, sum_val)
    tl.store(sumsq_ptr + n * OC + oc, sumsq_val)


@triton.jit
def train_finalize_kernel(
    sum_per_nc_ptr,
    batch_mean_ptr, batch_invstd_ptr,
    bn_weight_ptr, bn_bias_ptr,
    final_out_ptr,
    scale_factor,
    N, OC, SPATIAL,
    BLOCK_OC: tl.constexpr,
):
    n = tl.program_id(0)
    oc_id = tl.program_id(1)
    oc_offs = oc_id * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask = oc_offs < OC

    sum_nc = tl.load(sum_per_nc_ptr + n * OC + oc_offs, mask=mask, other=0.0)
    bmean = tl.load(batch_mean_ptr + oc_offs, mask=mask, other=0.0)
    binvstd = tl.load(batch_invstd_ptr + oc_offs, mask=mask, other=0.0)
    gamma = tl.load(bn_weight_ptr + oc_offs, mask=mask, other=0.0)
    beta = tl.load(bn_bias_ptr + oc_offs, mask=mask, other=0.0)

    mean_nc = sum_nc / SPATIAL
    y = (scale_factor * mean_nc - bmean) * binvstd * gamma + beta
    tl.store(final_out_ptr + n * OC + oc_offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.eps = float(eps)
        self.momentum = momentum

        ct = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.conv_weight = nn.Parameter(ct.weight.detach().clone())
        self.conv_bias = nn.Parameter(ct.bias.detach().clone())

        bn = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.bn_weight = nn.Parameter(bn.weight.detach().clone())
        self.bn_bias = nn.Parameter(bn.bias.detach().clone())
        self.register_buffer('bn_running_mean', bn.running_mean.clone())
        self.register_buffer('bn_running_var', bn.running_var.clone())

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels

        conv_out = F.conv_transpose3d(x, self.conv_weight, self.conv_bias)
        OD, OH, OW = conv_out.shape[2], conv_out.shape[3], conv_out.shape[4]
        SPATIAL = OD * OH * OW

        if self.training:
            sum_nc = torch.empty((N, OC), device=x.device, dtype=torch.float32)
            sumsq_nc = torch.empty((N, OC), device=x.device, dtype=torch.float32)
            channel_sum_kernel[(N, OC)](
                conv_out, sum_nc, sumsq_nc,
                SPATIAL,
                BLOCK_S=2048,
                num_warps=8, num_stages=3,
            )

            total = N * SPATIAL
            sum_c = sum_nc.sum(dim=0)
            sumsq_c = sumsq_nc.sum(dim=0)
            mean_c = sum_c / total
            var_c = sumsq_c / total - mean_c * mean_c
            scaled_mean = self.scale_factor * mean_c
            scaled_var = (self.scale_factor * self.scale_factor) * var_c
            unbiased_var = scaled_var * (total / (total - 1)) if total > 1 else scaled_var
            invstd = 1.0 / torch.sqrt(scaled_var + self.eps)

            with torch.no_grad():
                self.bn_running_mean.mul_(1 - self.momentum).add_(scaled_mean, alpha=self.momentum)
                self.bn_running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)

            final_flat = torch.empty((N, OC), device=x.device, dtype=x.dtype)
            BLOCK_OC = 128
            grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC)
            train_finalize_kernel[grid](
                sum_nc, scaled_mean, invstd,
                self.bn_weight, self.bn_bias,
                final_flat,
                self.scale_factor,
                N, OC, SPATIAL,
                BLOCK_OC=BLOCK_OC,
                num_warps=4,
            )
            return final_flat.view(N, OC, 1, 1, 1)

        final_out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)
        fused_scale_bn_pool_kernel[(N, OC)](
            conv_out,
            self.bn_weight, self.bn_bias,
            self.bn_running_mean, self.bn_running_var,
            final_out,
            self.scale_factor,
            self.eps,
            N, OC, SPATIAL,
            BLOCK_S=8192,
            num_warps=4, num_stages=2,
        )
        return final_out