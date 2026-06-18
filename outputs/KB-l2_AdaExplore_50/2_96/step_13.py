import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convt_pool_kernel(
    x_ptr,        # input (N, IC, ID, IH, IW)
    w_ptr,        # weight (IC, OC, KD, KH, KW)
    b_ptr,        # bias (OC,)
    out_ptr,      # output (N, OC)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,    # conv_transpose output dims
    PD, PH, PW,        # pool output dims = OD//2, OH//2, OW//2
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # one program per (n, oc) pair; iterate over pooled-output positions
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC
    
    bias = tl.load(b_ptr + oc)
    
    inv_total = 1.0 / (PD * PH * PW)
    
    acc_sum = 0.0
    
    # iterate over each pooled output position
    for pd in range(0, PD):
        for ph in range(0, PH):
            for pw in range(0, PW):
                # for this pooled position, max over 2x2x2 conv_transpose output positions
                max_val = -float('inf')
                for dd in range(0, 2):
                    for dh in range(0, 2):
                        for dw in range(0, 2):
                            od = pd * 2 + dd
                            oh = ph * 2 + dh
                            ow = pw * 2 + dw
                            
                            # compute conv_transpose output at (n, oc, od, oh, ow)
                            # output[n,oc,od,oh,ow] = sum over (ic,kd,kh,kw) of
                            #   input[n,ic,id,ih,iw] * weight[ic,oc,kd,kh,kw]
                            # where id*stride - pad + kd = od  =>  id = (od + pad - kd)/stride
                            # need (od + pad - kd) divisible by stride and id in [0,ID)
                            
                            val = bias
                            for kd in range(0, KD):
                                num_d = od + PAD - kd
                                id_ = num_d // STRIDE
                                valid_d = ((num_d - id_ * STRIDE) == 0) & (id_ >= 0) & (id_ < ID)
                                for kh in range(0, KH):
                                    num_h = oh + PAD - kh
                                    ih_ = num_h // STRIDE
                                    valid_h = ((num_h - ih_ * STRIDE) == 0) & (ih_ >= 0) & (ih_ < IH)
                                    for kw in range(0, KW):
                                        num_w = ow + PAD - kw
                                        iw_ = num_w // STRIDE
                                        valid_w = ((num_w - iw_ * STRIDE) == 0) & (iw_ >= 0) & (iw_ < IW)
                                        valid = valid_d & valid_h & valid_w
                                        
                                        # sum over input channels - vectorized
                                        ic_offs = tl.arange(0, BLOCK_IC)
                                        ic_mask = ic_offs < IC
                                        
                                        x_off = ((n * IC + ic_offs) * ID + id_) * IH * IW + ih_ * IW + iw_
                                        w_off = (ic_offs * OC + oc) * KD * KH * KW + kd * KH * KW + kh * KW + kw
                                        
                                        x_v = tl.load(x_ptr + x_off, mask=ic_mask & valid, other=0.0)
                                        w_v = tl.load(w_ptr + w_off, mask=ic_mask & valid, other=0.0)
                                        
                                        val += tl.sum(x_v * w_v, axis=0)
                            
                            val = val * SCALE
                            max_val = tl.maximum(max_val, val)
                
                acc_sum += max_val
    
    avg = acc_sum * inv_total
    avg = tl.minimum(tl.maximum(avg, 0.0), 1.0)
    tl.store(out_ptr + n * OC + oc, avg)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        S = self.stride
        P = self.padding
        
        OD = (ID - 1) * S - 2 * P + KD
        OH = (IH - 1) * S - 2 * P + KH
        OW = (IW - 1) * S - 2 * P + KW
        
        PK = self.maxpool_kernel_size
        PD = OD // PK
        PH = OH // PK
        PW = OW // PK
        
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        bias = self.conv_transpose.bias.contiguous()
        
        out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)
        
        # pick BLOCK_IC as next power of 2 >= IC
        BLOCK_IC = 1
        while BLOCK_IC < IC:
            BLOCK_IC *= 2
        BLOCK_IC = max(BLOCK_IC, 4)
        
        grid = (N * OC,)
        fused_convt_pool_kernel[grid](
            x, weight, bias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            S, P,
            float(self.scale),
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
        )
        return out