import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_maxpool_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    S_out = D_out * H_out * W_out
    n = pid // S_out
    rem = pid % S_out
    d_out = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    h_out = rem2 // W_out
    w_out = rem2 % W_out
    
    d_in0 = d_out * 2
    h_in0 = h_out * 2
    w_in0 = w_out * 2
    
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    
    S_in = D_in * H_in * W_in
    base_n = n * C * S_in
    c_stride = S_in
    
    # 2x2x2 maxpool window
    neg_inf = -float('inf')
    acc = tl.full([BLOCK_C], neg_inf, dtype=tl.float32)
    
    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            for ww in tl.static_range(0, 2):
                d_in = d_in0 + dd
                h_in = h_in0 + hh
                w_in = w_in0 + ww
                spatial_off = d_in * H_in * W_in + h_in * W_in + w_in
                x_offs = base_n + offs_c * c_stride + spatial_off
                v = tl.load(x_ptr + x_offs, mask=mask_c, other=neg_inf)
                acc = tl.maximum(acc, v)
    
    x = acc
    
    # softmax across channels
    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    exp_x = tl.exp(x_shift)
    exp_x = tl.where(mask_c, exp_x, 0.0)
    sum_exp = tl.sum(exp_x, axis=0)
    sm = exp_x / sum_exp
    
    # subtract
    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub
    
    # swish
    sig = 1.0 / (1.0 + tl.exp(-y))
    swish = sig * y
    
    swish = tl.where(mask_c, swish, neg_inf)
    result = tl.max(swish, axis=0)
    
    tl.store(out_ptr + n * S_out + rem, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride,
                                      padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse max_pool into our kernel
        
        N, C, D_in, H_in, W_in = x.shape
        D_out = D_in // 2
        H_out = H_in // 2
        W_out = W_in // 2
        
        x_contig = x.contiguous()
        sub_contig = self.subtract.contiguous()
        
        out = torch.empty((N, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
        
        BLOCK_C = triton.next_power_of_2(C)
        S_out = D_out * H_out * W_out
        
        grid = (N * S_out,)
        fused_maxpool_softmax_sub_swish_max_kernel[grid](
            x_contig, sub_contig, out,
            N, C,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        
        return out