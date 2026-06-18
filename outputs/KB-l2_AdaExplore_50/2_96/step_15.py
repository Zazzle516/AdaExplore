import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_kernel(
    y_ptr,        # conv output (N, OC, OD, OH, OW)
    out_ptr,      # output (N, OC)
    OD, OH, OW,
    PD, PH, PW,
    OC,
    SCALE: tl.constexpr,
    PK: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, oc)
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC
    
    base = (n * OC + oc) * OD * OH * OW
    inv_total = 1.0 / (PD * PH * PW)
    
    acc_sum = 0.0
    
    w_offs = tl.arange(0, BLOCK_W)
    
    for pd in range(0, PD):
        for ph in range(0, PH):
            # accumulate per-pw maxes across the W dimension as a vector
            # We process all pw positions in parallel via vectorization
            # max over a 2x2x2 window per pw
            row_max = tl.full([BLOCK_W], -float('inf'), dtype=tl.float32)
            for dd in range(0, PK):
                od = pd * PK + dd
                for dh in range(0, PK):
                    oh = ph * PK + dh
                    row_base = base + od * OH * OW + oh * OW
                    for dw in range(0, PK):
                        # for each pw, ow = pw*PK + dw; pw in [0, PW)
                        # offsets for all pw: w_offs * PK + dw
                        ow_idx = w_offs * PK + dw
                        mask = w_offs < PW
                        v = tl.load(y_ptr + row_base + ow_idx, mask=mask, other=-float('inf'))
                        row_max = tl.maximum(row_max, v)
            # sum over pw positions (only the valid ones)
            mask_pw = w_offs < PW
            row_max = tl.where(mask_pw, row_max, 0.0)
            acc_sum += tl.sum(row_max, axis=0)
    
    avg = acc_sum * SCALE * inv_total
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
        N = x.shape[0]
        OC = self.out_channels
        PK = self.maxpool_kernel_size
        
        # Run conv_transpose using cuDNN (heavy op preserved)
        y = self.conv_transpose(x)  # (N, OC, OD, OH, OW)
        y = y.contiguous()
        _, _, OD, OH, OW = y.shape
        PD = OD // PK
        PH = OH // PK
        PW = OW // PK
        
        out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)
        
        # Choose BLOCK_W as next power of 2 >= PW
        BLOCK_W = 1
        while BLOCK_W < PW:
            BLOCK_W *= 2
        BLOCK_W = max(BLOCK_W, 4)
        
        grid = (N * OC,)
        fused_pool_kernel[grid](
            y, out,
            OD, OH, OW,
            PD, PH, PW,
            OC,
            SCALE=float(self.scale),
            PK=PK,
            BLOCK_W=BLOCK_W,
            num_warps=4,
        )
        return out