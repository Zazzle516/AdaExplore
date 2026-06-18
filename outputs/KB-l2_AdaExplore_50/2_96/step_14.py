import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,          # (N, IC, ID, IH, IW)
    w_ptr,          # (IC, OC, KD, KH, KW)
    b_ptr,          # (OC,)
    out_ptr,        # (N, OC)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,  # pooled dims
    SCALE: tl.constexpr,
    INV_COUNT: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, pooled spatial tile in flattened form)
    # We'll do one program per (n, oc) pair, and loop over pooled positions
    # Actually: this is heavy. Let's do: one program per (n) processing all OC, all pooled positions.
    # But that's a lot. Better: per (n, oc_block).
    
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    
    # Load bias for these OCs
    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    
    # Accumulator for sum over pooled positions, per OC
    sum_acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    
    pooled_total = PD * PH * PW
    
    # For each pooled output position
    for pidx in range(0, pooled_total):
        pw = pidx % PW
        ptmp = pidx // PW
        ph = ptmp % PH
        pd = ptmp // PH
        
        # The pool window in conv-transpose output space is [pd*2:pd*2+2, ph*2:ph*2+2, pw*2:pw*2+2]
        # Compute max over 8 positions for each OC
        max_vals = tl.full((BLOCK_OC,), -1e30, dtype=tl.float32)
        
        for dd in range(0, 2):
            for hh in range(0, 2):
                for ww in range(0, 2):
                    od = pd * 2 + dd
                    oh = ph * 2 + hh
                    ow = pw * 2 + ww
                    
                    # Compute conv_transpose output at (n, oc_offs, od, oh, ow)
                    # out[n,oc,od,oh,ow] = sum_{ic, kd, kh, kw} x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
                    # where id*STRIDE - PAD + kd = od => id = (od + PAD - kd) / STRIDE, must be integer
                    
                    val = tl.zeros((BLOCK_OC,), dtype=tl.float32) + bias_vals
                    
                    for kd in range(0, KD):
                        id_num = od + PAD - kd
                        id_q = id_num // STRIDE
                        id_r = id_num - id_q * STRIDE
                        valid_d = (id_r == 0) & (id_q >= 0) & (id_q < ID)
                        
                        for kh in range(0, KH):
                            ih_num = oh + PAD - kh
                            ih_q = ih_num // STRIDE
                            ih_r = ih_num - ih_q * STRIDE
                            valid_h = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
                            
                            for kw in range(0, KW):
                                iw_num = ow + PAD - kw
                                iw_q = iw_num // STRIDE
                                iw_r = iw_num - iw_q * STRIDE
                                valid_w = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                                
                                valid = valid_d & valid_h & valid_w
                                
                                if valid:
                                    # Accumulate over IC
                                    for ic in range(0, IC):
                                        x_off = ((pid_n * IC + ic) * ID + id_q) * IH * IW + ih_q * IW + iw_q
                                        x_val = tl.load(x_ptr + x_off)
                                        w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                        val += x_val * w_val
                    
                    val = val * SCALE
                    max_vals = tl.maximum(max_vals, val)
        
        sum_acc += max_vals
    
    mean = sum_acc * INV_COUNT
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    
    out_off = pid_n * OC + oc_offs
    tl.store(out_ptr + out_off, mean, mask=oc_mask)


@triton.jit
def fused_maxpool_mean_clamp_kernel(
    x_ptr,
    out_ptr,
    N, C, D, H, W,
    Dp, Hp, Wp,
    scale,
    inv_count,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    nc_base = (n * C + c) * D * H * W
    pooled_total = Dp * Hp * Wp
    
    acc = 0.0
    for off in range(0, pooled_total, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < pooled_total
        pw = idx % Wp
        tmp = idx // Wp
        ph = tmp % Hp
        pd = tmp // Hp
        
        d0 = pd * 2
        h0 = ph * 2
        w0 = pw * 2
        
        b000 = nc_base + d0 * H * W + h0 * W + w0
        b001 = b000 + 1
        b010 = b000 + W
        b011 = b010 + 1
        b100 = b000 + H * W
        b101 = b100 + 1
        b110 = b100 + W
        b111 = b110 + 1
        
        v0 = tl.load(x_ptr + b000, mask=mask, other=-1e30)
        v1 = tl.load(x_ptr + b001, mask=mask, other=-1e30)
        v2 = tl.load(x_ptr + b010, mask=mask, other=-1e30)
        v3 = tl.load(x_ptr + b011, mask=mask, other=-1e30)
        v4 = tl.load(x_ptr + b100, mask=mask, other=-1e30)
        v5 = tl.load(x_ptr + b101, mask=mask, other=-1e30)
        v6 = tl.load(x_ptr + b110, mask=mask, other=-1e30)
        v7 = tl.load(x_ptr + b111, mask=mask, other=-1e30)
        
        m = tl.maximum(tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3)),
                       tl.maximum(tl.maximum(v4, v5), tl.maximum(v6, v7)))
        m = tl.where(mask, m, 0.0)
        acc += tl.sum(m, axis=0)
    
    mean = acc * inv_count * scale
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * C + c, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1
        self.maxpool_kernel_size = maxpool_kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        
        N, C, D, H, W = x.shape
        k = self.maxpool_kernel_size
        Dp, Hp, Wp = D // k, H // k, W // k
        
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        
        if k == 2:
            pooled_total = Dp * Hp * Wp
            if pooled_total <= 256:
                BLOCK = 256
            elif pooled_total <= 512:
                BLOCK = 512
            elif pooled_total <= 1024:
                BLOCK = 1024
            else:
                BLOCK = 2048
            
            inv_count = 1.0 / float(pooled_total)
            grid = (N * C,)
            fused_maxpool_mean_clamp_kernel[grid](
                x, out,
                N, C, D, H, W,
                Dp, Hp, Wp,
                float(self.scale),
                inv_count,
                BLOCK=BLOCK,
                num_warps=4,
            )
        else:
            x = self.maxpool(x)
            x = x * self.scale
            x = self.global_avg_pool(x)
            out = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return out