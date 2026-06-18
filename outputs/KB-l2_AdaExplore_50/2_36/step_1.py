import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_min_sum_gelu_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW, SH, SW, PH, PW,
    BLOCK_W: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # grid: (N, OW_tiles)
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    w_start = pid_w * BLOCK_W
    ow_offs = w_start + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    # For each output position (n, ow), we need:
    # min over oc of sum over oh of conv_out[n, oc, oh, ow]
    # Wait: it's min over channels first, then sum over H.
    # So: for each (n, ow): result = sum_{oh} min_{oc} conv[n, oc, oh, ow]
    
    # We need to compute conv[n, oc, oh, ow] for all oc, all oh, then min over oc, sum over oh.
    # conv[n, oc, oh, ow] = bias[oc] + sum_{ic, kh, kw} x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih*SH = oh + PH - kh, iw*SW = ow + PW - kw

    # Strategy: accumulate sum_oh of min_oc(conv[n,oc,oh,ow]) for each ow in tile.
    # We loop over oh, and for each oh, we compute conv[n, oc, oh, ow] for all oc and ow in tile,
    # then take min over oc, then add to running sum.

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # Loop over oh
    for oh in range(0, OH):
        # Compute conv[n, :, oh, ow_tile] for all OC
        # We need an array of shape (OC, BLOCK_W). Then min over OC -> (BLOCK_W,)
        # OC could be large (128), so we tile OC.
        
        # Initialize min vector with +inf
        min_vec = tl.full((BLOCK_W,), float('inf'), dtype=tl.float32)
        
        # Loop over OC in chunks of 1 to keep things simple, accumulating into min
        # Actually let's just iterate one OC at a time
        for oc in range(0, OC):
            # conv[n, oc, oh, ow] = bias[oc] + sum over ic, kh, kw of x[n, ic, ih, iw] * w[ic, oc, kh, kw]
            # ih = (oh + PH - kh) / SH, must be integer and in [0, IH)
            
            b_val = tl.load(b_ptr + oc)
            conv_val = tl.full((BLOCK_W,), b_val, dtype=tl.float32)
            
            for kh in range(0, KH):
                ih_num = oh + PH - kh
                ih = ih_num // SH
                ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih >= 0) & (ih < IH)
                
                for kw in range(0, KW):
                    iw_num = ow_offs + PW - kw
                    iw = iw_num // SW
                    iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw >= 0) & (iw < IW)
                    valid = ih_valid & iw_valid & ow_mask
                    
                    # Loop over IC in blocks
                    for ic_start in range(0, IC, BLOCK_IC):
                        ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                        ic_mask = ic_offs < IC
                        
                        # x[n, ic, ih, iw] - shape (BLOCK_IC, BLOCK_W)
                        x_offs = (pid_n * IC * IH * IW
                                  + ic_offs[:, None] * IH * IW
                                  + ih * IW
                                  + iw[None, :])
                        x_mask = ic_mask[:, None] & valid[None, :]
                        x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)
                        
                        # w[ic, oc, kh, kw] - shape (BLOCK_IC,)
                        w_offs = (ic_offs * OC * KH * KW
                                  + oc * KH * KW
                                  + kh * KW
                                  + kw)
                        w_vals = tl.load(w_ptr + w_offs, mask=ic_mask, other=0.0)
                        
                        # Multiply and accumulate
                        prod = x_vals * w_vals[:, None]
                        conv_val += tl.sum(prod, axis=0)
            
            # Update min
            min_vec = tl.minimum(min_vec, conv_val)
        
        acc += min_vec
    
    # acc shape: (BLOCK_W,) - sum over OH of min over OC
    # Apply GELU
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu_out = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))
    
    # Add bias (bias_shape is (1,1,1) so it's a scalar)
    bias_val = tl.load(bias_ptr)
    result = gelu_out + bias_val
    
    # Store: output shape is (N, 1, 1, OW)
    out_offs = pid_n * OW + ow_offs
    tl.store(out_ptr + out_offs, result, mask=ow_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OP = self.output_padding
        OH = (IH - 1) * SH - 2 * PH + KH + OP
        OW = (IW - 1) * SW - 2 * PW + KW + OP

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)
        conv_bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias = self.bias.contiguous()

        out = torch.empty((N, 1, 1, OW), device=x.device, dtype=x.dtype)

        BLOCK_W = 32
        BLOCK_IC = 16
        grid = (N, triton.cdiv(OW, BLOCK_W))

        conv_transpose_min_sum_gelu_kernel[grid](
            x, weight, conv_bias, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, SH, SW, PH, PW,
            BLOCK_W=BLOCK_W,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
        )

        return out