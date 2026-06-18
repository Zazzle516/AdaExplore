import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Fused: (x + sum_weight) -> LayerNorm over last dim (length L) -> AvgPool3d(2,2,2) -> GELU
# Input layout: [N, C, D, H, W], LN normalizes over W (last dim).
# Pool window 2x2x2 -> output [N, C, D/2, H/2, W/2]
# Output ow indexes 0..OW-1, where OW = W/2.
# Strategy: one program per (n, c, od, oh). Inside, for each of 4 input rows
#   (id in {2*od, 2*od+1}, ih in {2*oh, 2*oh+1}), load full row of length W,
#   add sum_weight, compute mean/var across W, normalize with gamma/beta,
#   then for each output ow, average the 8 normalized values across the 2x2x2 window.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 256}, num_warps=4, num_stages=2),
    ],
    key=['W'],
)
@triton.jit
def fused_post_kernel(
    x_ptr,           # input: [N, C, D, H, W] contiguous
    out_ptr,         # output: [N, C, OD, OH, OW] contiguous
    gamma_ptr,       # [W]
    beta_ptr,        # [W]
    sum_weight,      # scalar fp32
    eps,             # scalar fp32
    inv_W,           # 1.0 / W
    N, C, D, H, W,
    OD, OH, OW,
    BLOCK_W: tl.constexpr,
):
    # pid -> (n, c, od, oh)
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    od = tmp % OD
    tmp = tmp // OD
    c  = tmp % C
    n  = tmp // C
    
    w_idx = tl.arange(0, BLOCK_W)
    w_mask = w_idx < W
    
    gamma = tl.load(gamma_ptr + w_idx, mask=w_mask, other=0.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + w_idx, mask=w_mask, other=0.0).to(tl.float32)
    
    # Compute base index per (n, c)
    base_nc = (n * C + c) * D * H * W
    
    # Accumulator over the 2x2 (id, ih) sub-window of 4 rows; we'll accumulate
    # into a per-ow output of length OW. For 2x2x2 pool: avg = sum_8 / 8.
    # We'll accumulate normed-row values, summing pairs along W as we go,
    # then sum 4 rows together, then divide by 8 and apply GELU.
    
    # We'll keep an accumulator of length BLOCK_W (per-input-w). After all 4
    # rows, fold pairs along W to get OW outputs, /8, GELU, store.
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    
    for a in tl.static_range(0, 2):
        for b in tl.static_range(0, 2):
            id_ = 2 * od + a
            ih_ = 2 * oh + b
            row_base = base_nc + (id_ * H + ih_) * W
            ptrs = x_ptr + row_base + w_idx
            x = tl.load(ptrs, mask=w_mask, other=0.0).to(tl.float32)
            x = x + sum_weight
            x_safe = tl.where(w_mask, x, 0.0)
            mean = tl.sum(x_safe, axis=0) * inv_W
            diff = tl.where(w_mask, x - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) * inv_W
            rstd = 1.0 / tl.sqrt(var + eps)
            normed = (x - mean) * rstd * gamma + beta
            acc = acc + normed
    
    # Now acc holds sum over 4 input rows of normed values, shape [BLOCK_W].
    # We need, for each ow in [0, OW), avg over (2*ow, 2*ow+1) -> sum/8 then GELU.
    # Process in chunks of BLOCK_W, but typically BLOCK_W >= W so we handle all at once.
    # Reshape via gather: for ow in arange(OW), out[ow] = (acc[2*ow] + acc[2*ow+1]) / 8.
    # Use the fact that BLOCK_W >= W = 2*OW.
    
    ow_idx = tl.arange(0, BLOCK_W // 2)
    ow_mask = ow_idx < OW
    # Get even and odd indexed elements via masked loads from acc... but acc is a register tensor.
    # We can compute by treating acc as paired: acc has BLOCK_W elements; pair k corresponds to
    # acc[2k] and acc[2k+1]. We can use tl.reshape if available, else recompute by reloading.
    
    # Simpler: re-do the sum by loading shifted. But we already reduced. Use tl.reshape:
    acc2 = tl.reshape(acc, (BLOCK_W // 2, 2))
    pair_sum = tl.sum(acc2, axis=1)  # [BLOCK_W//2]
    
    pooled = pair_sum * 0.125  # /8
    
    # GELU tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    x3 = pooled * pooled * pooled
    inner = k0 * (pooled + k1 * x3)
    # tanh via sigmoid: tanh(y) = 2*sigmoid(2y) - 1
    two_y = 2.0 * inner
    sig = 1.0 / (1.0 + tl.exp(-two_y))
    tanh_v = 2.0 * sig - 1.0
    gelu = 0.5 * pooled * (1.0 + tanh_v)
    
    # Store to out[n, c, od, oh, :]
    out_base = ((n * C + c) * OD + od) * OH * OW + oh * OW
    out_ptrs = out_ptr + out_base + ow_idx
    tl.store(out_ptrs, gelu, mask=ow_mask)


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
        self.norm_shape = tuple(norm_shape) if not isinstance(norm_shape, int) else (norm_shape,)

    def forward(self, x):
        x = self.conv_transpose(x)
        # x shape: [N, C, D, H, W]
        N, C, D, H, W = x.shape
        pk = self.pool_kernel_size
        
        # Fused fast path: pool=(2,2,2), norm over last dim (W), W is power-of-two and divisible by 2
        ns = self.norm_shape
        fast = (
            pk == (2, 2, 2)
            and len(ns) == 1
            and ns[0] == W
            and (D % 2 == 0) and (H % 2 == 0) and (W % 2 == 0)
        )
        if not fast:
            x = x + self.sum_weight
            x = self.norm(x)
            x = self.avg_pool(x)
            x = self.gelu(x)
            return x
        
        OD = D // 2
        OH = H // 2
        OW = W // 2
        
        x = x.contiguous()
        out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
        
        total = N * C * OD * OH
        grid = (total,)
        
        fused_post_kernel[grid](
            x, out,
            self.norm.weight, self.norm.bias,
            float(self.sum_weight.item()),
            float(self.norm.eps),
            1.0 / W,
            N, C, D, H, W,
            OD, OH, OW,
        )
        return out