import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_gelu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    H_in, W_in, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # grid: (N, OC, H_out * ceil(W_out/BLOCK_W))
    n = tl.program_id(0)
    oc = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    num_w_blocks = (W_out + BLOCK_W - 1) // BLOCK_W
    h_out = pid_hw // num_w_blocks
    w_block = pid_hw % num_w_blocks
    
    w_offs = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W_out
    
    # Accumulator
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    # Add bias
    bias = tl.load(b_ptr + oc).to(tl.float32)
    acc += bias
    
    # ConvTranspose2d with stride=1, padding=0:
    # y[n,oc,h_out,w_out] = sum_{ic,kh,kw} x[n,ic,h_out-kh, w_out-kw] * w[ic,oc,kh,kw]
    # where h_out-kh, w_out-kw must be in [0, H_in)
    
    x_n_base = n * IC * H_in * W_in
    
    for ic in tl.static_range(0, IC):
        x_ic_base = x_n_base + ic * H_in * W_in
        w_ic_base = ic * OC * KH * KW + oc * KH * KW
        for kh in tl.static_range(0, KH):
            h_in = h_out - kh
            h_valid = (h_in >= 0) & (h_in < H_in)
            for kw in tl.static_range(0, KW):
                w_in = w_offs - kw
                in_mask = h_valid & (w_in >= 0) & (w_in < W_in) & w_mask
                x_offset = x_ic_base + h_in * W_in + w_in
                x_val = tl.load(x_ptr + x_offset, mask=in_mask, other=0.0).to(tl.float32)
                w_val = tl.load(w_ptr + w_ic_base + kh * KW + kw).to(tl.float32)
                acc += x_val * w_val
    
    # Apply GELU
    inv_sqrt2 = 0.7071067811865475
    gx = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    
    # Store
    y_offset = n * OC * H_out * W_out + oc * H_out * W_out + h_out * W_out + w_offs
    tl.store(y_ptr + y_offset, gx, mask=w_mask)


@triton.jit
def groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, G, CPG: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_elems = CPG * HW
    base = n * C * HW + g * CPG * HW

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for c in tl.static_range(0, CPG):
        c_base = base + c * HW
        for off in range(0, HW, BLOCK_SIZE):
            offs = off + tl.arange(0, BLOCK_SIZE)
            mask = offs < HW
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0).to(tl.float32)
            x_masked = tl.where(mask, x, 0.0)
            sum_val += x_masked
            sumsq_val += x_masked * x_masked

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sumsq_val, axis=0)

    mean = s / group_elems
    var = sq / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    g_off = g * CPG
    for c in tl.static_range(0, CPG):
        c_base = base + c * HW
        w = tl.load(weight_ptr + g_off + c).to(tl.float32)
        b = tl.load(bias_ptr + g_off + c).to(tl.float32)
        scale = rstd * w
        shift = b - mean * scale
        for off in range(0, HW, BLOCK_SIZE):
            offs = off + tl.arange(0, BLOCK_SIZE)
            mask = offs < HW
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * scale + shift
            tl.store(y_ptr + c_base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        H_out = H_in + KH - 1  # stride=1, padding=0
        W_out = W_in + KW - 1
        
        # Output of conv_transpose + GELU
        gelu_out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)
        
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        bias = self.conv_transpose.bias.contiguous()
        
        BLOCK_W = 128
        num_w_blocks = (W_out + BLOCK_W - 1) // BLOCK_W
        grid = (N, OC, H_out * num_w_blocks)
        
        conv_transpose_gelu_kernel[grid](
            x, weight, bias, gelu_out,
            N, IC, OC,
            H_in, W_in, H_out, W_out,
            KH, KW,
            BLOCK_W=BLOCK_W,
            num_warps=4,
            num_stages=2,
        )
        
        # GroupNorm
        y = torch.empty_like(gelu_out)
        HW = H_out * W_out
        G = self.num_groups
        CPG = OC // G
        
        if HW >= 16384:
            BLOCK_SIZE = 2048
            num_warps = 8
        elif HW >= 4096:
            BLOCK_SIZE = 1024
            num_warps = 8
        else:
            BLOCK_SIZE = 512
            num_warps = 4
        
        grid2 = (N * G,)
        groupnorm_kernel[grid2](
            gelu_out, y,
            self.group_norm.weight, self.group_norm.bias,
            N, OC, HW, G, CPG,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )
        return y