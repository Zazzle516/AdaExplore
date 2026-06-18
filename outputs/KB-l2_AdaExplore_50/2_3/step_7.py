import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_kernel(
    x_ptr,           # input from conv_transpose: [N, C, D, H, W]
    out_ptr,         # output: [N, C, D//2, H//2, W//2]
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    sum_weight,      # scalar fp32
    eps,             # scalar fp32
    inv_C,           # 1.0 / C
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
):
    # one program per output element (n, od, oh, ow)
    pid = tl.program_id(0)
    total = tl.program_id(1)  # unused
    
    # decode pid -> (n, od, oh, ow)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n  = tmp // OD
    
    c_idx = tl.arange(0, BLOCK_C)
    c_mask = c_idx < C
    
    # We need to compute output[n, c, od, oh, ow] for all c
    # = GELU( avg_pool over 2x2x2 of LayerNorm(x[n,c,...] + sum_weight) )
    # AvgPool window: input positions (2*od+a, 2*oh+b, 2*ow+c2) for a,b,c2 in {0,1}
    # LayerNorm is across C dimension (norm_shape=(C,))
    
    # For each of 8 input positions, we need to compute LayerNorm across C
    # then average across the 8 positions. Then GELU.
    
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    
    for a in tl.static_range(0, 2):
        for b in tl.static_range(0, 2):
            for c2 in tl.static_range(0, 2):
                id_ = 2 * od + a
                ih_ = 2 * oh + b
                iw_ = 2 * ow + c2
                # load x[n, :, id_, ih_, iw_]
                base = n * stride_n + id_ * stride_d + ih_ * stride_h + iw_ * stride_w
                ptrs = x_ptr + base + c_idx * stride_c
                x = tl.load(ptrs, mask=c_mask, other=0.0).to(tl.float32)
                x = x + sum_weight
                # mean over C
                x_safe = tl.where(c_mask, x, 0.0)
                mean = tl.sum(x_safe, axis=0) * inv_C
                diff = tl.where(c_mask, x - mean, 0.0)
                var = tl.sum(diff * diff, axis=0) * inv_C
                rstd = 1.0 / tl.sqrt(var + eps)
                # apply gamma, beta
                gamma = tl.load(gamma_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
                beta = tl.load(beta_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
                normed = (x - mean) * rstd * gamma + beta
                acc = acc + normed
    
    acc = acc * 0.125  # divide by 8 for avg pool
    
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    
    # store to output
    out_base = n * (C * OD * OH * OW) + od * (OH * OW) + oh * OW + ow
    out_ptrs = out_ptr + out_base + c_idx * (OD * OH * OW)
    tl.store(out_ptrs, gelu, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        self.out_channels = out_channels
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        # x shape: [N, C, D, H, W]
        N, C, D, H, W = x.shape
        pk = self.pool_kernel_size
        OD = D // pk[0]
        OH = H // pk[1]
        OW = W // pk[2]
        
        # Only handle the case where pool_kernel_size = (2,2,2)
        if pk != (2, 2, 2):
            x = x + self.sum_weight
            x = self.norm(x)
            x = self.avg_pool(x)
            x = self.gelu(x)
            return x
        
        x = x.contiguous()
        out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
        
        # BLOCK_C must be >= C, power of 2
        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2
        
        total = N * OD * OH * OW
        grid = (total, 1)
        
        sn = x.stride(0)
        sc = x.stride(1)
        sd = x.stride(2)
        sh = x.stride(3)
        sw = x.stride(4)
        
        fused_post_kernel[grid](
            x, out,
            self.norm.weight, self.norm.bias,
            float(self.sum_weight.item()),
            float(self.norm.eps),
            1.0 / C,
            N, C, D, H, W,
            OD, OH, OW,
            sn, sc, sd, sh, sw,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out