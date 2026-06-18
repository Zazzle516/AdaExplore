import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convT_pool_kernel(
    x_ptr,           # (N, IC, ID, IH, IW)
    w_ptr,           # (IC, OC, KD, KH, KW)
    b_ptr,           # (OC,)
    out_ptr,         # (N, OC)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,      # pooled output spatial size = OD//2, OH//2, OW//2
    scale, inv_count,
    BLOCK_PD: tl.constexpr,
    BLOCK_PHW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
):
    # one program per (n, oc)
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC
    
    bias = tl.load(b_ptr + oc)
    
    PHW = PH * PW
    pooled_total = PD * PHW
    
    acc = 0.0
    
    # iterate over pooled output positions
    for pd_start in range(0, PD, BLOCK_PD):
        pd_offs = pd_start + tl.arange(0, BLOCK_PD)  # [BLOCK_PD]
        pd_mask = pd_offs < PD
        
        for phw_start in range(0, PHW, BLOCK_PHW):
            phw_offs = phw_start + tl.arange(0, BLOCK_PHW)  # [BLOCK_PHW]
            phw_mask = phw_offs < PHW
            ph_offs = phw_offs // PW
            pw_offs = phw_offs % PW
            
            # combined mask: [BLOCK_PD, BLOCK_PHW]
            mask2d = pd_mask[:, None] & phw_mask[None, :]
            
            # For each pooled position, we need max over 2x2x2 window of conv output
            # OD index for the 8 corners: 2*pd, 2*pd+1
            # OH index: 2*ph, 2*ph+1
            # OW index: 2*pw, 2*pw+1
            
            # We'll compute conv output for all 8 corners, then reduce via max
            # Initialize 8 accumulators with bias (will add conv)
            
            # Output coords for 8 corners
            od0 = 2 * pd_offs       # [BLOCK_PD]
            od1 = od0 + 1
            oh0 = 2 * ph_offs       # [BLOCK_PHW]
            oh1 = oh0 + 1
            ow0 = 2 * pw_offs       # [BLOCK_PHW]
            ow1 = ow0 + 1
            
            # Each conv output value:
            # out[n,oc,od,oh,ow] = sum_{ic,kd,kh,kw} x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
            # where: od + PAD = id*STRIDE + kd  =>  id = (od + PAD - kd) / STRIDE if divisible
            
            # For maxpool 2: od0,od1 differ by 1.
            # For each (kd,kh,kw), we compute:
            #   id_for_od0 = (od0 + PAD - kd) / STRIDE   if (od0+PAD-kd) % STRIDE == 0
            
            # Initialize 8 accumulators
            v000 = tl.zeros((BLOCK_PD, BLOCK_PHW), dtype=tl.float32)
            v001 = tl.zeros((BLOCK_PD, BLOCK_PHW), dtype=tl.float32)
            v010 = tl.zeros((BLOCK_PD, BLOCK_PHW), dtype=tl.float32)
            v011 = tl.zeros((BLOCK_PD, BLOCK_PHW), dtype=tl.float32)
            v100 = tl.zeros((BLOCK_PD, BLOCK_PHW), dtype=tl.float32)
            v101 = tl.zeros((BLOCK_PD, BLOCK_PHW), dtype=tl.float32)
            v110 = tl.zeros((BLOCK_PD, BLOCK_PHW), dtype=tl.float32)
            v111 = tl.zeros((BLOCK_PD, BLOCK_PHW), dtype=tl.float32)
            
            x_n_base = n * IC * ID * IH * IW
            
            for ic in range(0, IC):
                x_base = x_n_base + ic * ID * IH * IW
                w_base = ic * OC * KD * KH * KW + oc * KD * KH * KW
                
                for kd in tl.static_range(0, KD):
                    # compute id for od0 and od1
                    num_d0 = od0 + PAD - kd  # [BLOCK_PD]
                    num_d1 = od1 + PAD - kd
                    id0 = num_d0 // STRIDE
                    id1 = num_d1 // STRIDE
                    valid_d0 = (num_d0 >= 0) & ((num_d0 % STRIDE) == 0) & (id0 >= 0) & (id0 < ID)
                    valid_d1 = (num_d1 >= 0) & ((num_d1 % STRIDE) == 0) & (id1 >= 0) & (id1 < ID)
                    
                    for kh in tl.static_range(0, KH):
                        num_h0 = oh0 + PAD - kh
                        num_h1 = oh1 + PAD - kh
                        ih0 = num_h0 // STRIDE
                        ih1 = num_h1 // STRIDE
                        valid_h0 = (num_h0 >= 0) & ((num_h0 % STRIDE) == 0) & (ih0 >= 0) & (ih0 < IH)
                        valid_h1 = (num_h1 >= 0) & ((num_h1 % STRIDE) == 0) & (ih1 >= 0) & (ih1 < IH)
                        
                        for kw in tl.static_range(0, KW):
                            num_w0 = ow0 + PAD - kw
                            num_w1 = ow1 + PAD - kw
                            iw0 = num_w0 // STRIDE
                            iw1 = num_w1 // STRIDE
                            valid_w0 = (num_w0 >= 0) & ((num_w0 % STRIDE) == 0) & (iw0 >= 0) & (iw0 < IW)
                            valid_w1 = (num_w1 >= 0) & ((num_w1 % STRIDE) == 0) & (iw1 >= 0) & (iw1 < IW)
                            
                            wval = tl.load(w_ptr + w_base + kd * KH * KW + kh * KW + kw)
                            
                            # 8 corners
                            # (od0, oh0, ow0)
                            v_d = valid_d0[:, None]
                            v_h0 = valid_h0  # [BLOCK_PHW]
                            v_w0 = valid_w0  # [BLOCK_PHW]
                            v_h1 = valid_h1
                            v_w1 = valid_w1
                            
                            # idx for (id0, ih0, iw0)
                            idx000 = x_base + id0[:, None] * IH * IW + (ih0 * IW + iw0)[None, :]
                            mask000 = v_d & (v_h0 & v_w0)[None, :]
                            x000 = tl.load(x_ptr + idx000, mask=mask000, other=0.0)
                            v000 += x000 * wval
                            
                            idx001 = x_base + id0[:, None] * IH * IW + (ih0 * IW + iw1)[None, :]
                            mask001 = v_d & (v_h0 & v_w1)[None, :]
                            x001 = tl.load(x_ptr + idx001, mask=mask001, other=0.0)
                            v001 += x001 * wval
                            
                            idx010 = x_base + id0[:, None] * IH * IW + (ih1 * IW + iw0)[None, :]
                            mask010 = v_d & (v_h1 & v_w0)[None, :]
                            x010 = tl.load(x_ptr + idx010, mask=mask010, other=0.0)
                            v010 += x010 * wval
                            
                            idx011 = x_base + id0[:, None] * IH * IW + (ih1 * IW + iw1)[None, :]
                            mask011 = v_d & (v_h1 & v_w1)[None, :]
                            x011 = tl.load(x_ptr + idx011, mask=mask011, other=0.0)
                            v011 += x011 * wval
                            
                            v_d = valid_d1[:, None]
                            idx100 = x_base + id1[:, None] * IH * IW + (ih0 * IW + iw0)[None, :]
                            mask100 = v_d & (v_h0 & v_w0)[None, :]
                            x100 = tl.load(x_ptr + idx100, mask=mask100, other=0.0)
                            v100 += x100 * wval
                            
                            idx101 = x_base + id1[:, None] * IH * IW + (ih0 * IW + iw1)[None, :]
                            mask101 = v_d & (v_h0 & v_w1)[None, :]
                            x101 = tl.load(x_ptr + idx101, mask=mask101, other=0.0)
                            v101 += x101 * wval
                            
                            idx110 = x_base + id1[:, None] * IH * IW + (ih1 * IW + iw0)[None, :]
                            mask110 = v_d & (v_h1 & v_w0)[None, :]
                            x110 = tl.load(x_ptr + idx110, mask=mask110, other=0.0)
                            v110 += x110 * wval
                            
                            idx111 = x_base + id1[:, None] * IH * IW + (ih1 * IW + iw1)[None, :]
                            mask111 = v_d & (v_h1 & v_w1)[None, :]
                            x111 = tl.load(x_ptr + idx111, mask=mask111, other=0.0)
                            v111 += x111 * wval
            
            # add bias and scale
            v000 = (v000 + bias) * scale
            v001 = (v001 + bias) * scale
            v010 = (v010 + bias) * scale
            v011 = (v011 + bias) * scale
            v100 = (v100 + bias) * scale
            v101 = (v101 + bias) * scale
            v110 = (v110 + bias) * scale
            v111 = (v111 + bias) * scale
            
            # max over the 8 corners
            m1 = tl.maximum(tl.maximum(v000, v001), tl.maximum(v010, v011))
            m2 = tl.maximum(tl.maximum(v100, v101), tl.maximum(v110, v111))
            m = tl.maximum(m1, m2)
            
            m = tl.where(mask2d, m, 0.0)
            acc += tl.sum(m)
    
    mean = acc * inv_count
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * OC + oc, mean)


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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        stride = self.stride
        padding = self.padding
        
        # transposed conv output spatial dims
        OD = (ID - 1) * stride - 2 * padding + KD
        OH = (IH - 1) * stride - 2 * padding + KH
        OW = (IW - 1) * stride - 2 * padding + KW
        
        k = self.maxpool_kernel_size
        
        if k == 2 and stride == 2 and padding == 1 and KD == 3:
            PD, PH, PW = OD // 2, OH // 2, OW // 2
            pooled_total = PD * PH * PW
            inv_count = 1.0 / float(pooled_total)
            
            x = x.contiguous()
            w = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
            b = self.conv_transpose.bias.contiguous()
            
            out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)
            
            grid = (N * OC,)
            
            BLOCK_PD = 16
            BLOCK_PHW = 64
            
            fused_convT_pool_kernel[grid](
                x, w, b, out,
                N, IC, ID, IH, IW,
                OC, OD, OH, OW,
                PD, PH, PW,
                float(self.scale), inv_count,
                BLOCK_PD=BLOCK_PD,
                BLOCK_PHW=BLOCK_PHW,
                KD=KD, KH=KH, KW=KW,
                STRIDE=stride, PAD=padding,
                num_warps=4,
                num_stages=2,
            )
            return out
        else:
            x = self.conv_transpose(x)
            x = x * self.scale
            x = self.maxpool(x)
            x = self.global_avg_pool(x)
            x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
            return x