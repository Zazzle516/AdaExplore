import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# The key observation: 
# x -> Linear (B, out) -> subtract -> mean over dim=1 keepdim -> shape (B, 1)
# logsumexp over dim=1 keepdim on (B,1) is just identity (log(exp(x)) = x)
# Then GELU applied to (B, 1), then broadcast-add to original_x (B, in_features)
#
# So scalar per row s_b = mean_j(W_j @ x_b + bias_j - subtract_j)
#                       = (1/out) * sum_j (W_j @ x_b) + (1/out)*sum_j(bias_j - subtract_j)
#                       = ( (sum_j W_j) @ x_b ) / out + const
#
# But safety contract: must not collapse heavy op into reduction at init.
# So we must execute the gemm at runtime. We'll do a fused kernel that computes
# linear + subtract + mean per row at runtime as a matmul.
#
# Strategy: one program per row (batch element). Each program loops over K (in_features)
# and N (out_features) tiles. We compute partial dot products and accumulate sum across N.
# Actually simpler: compute matvec-like reduction: for each batch row b, accumulate
#   sum_n ( sum_k W[n,k] * x[b,k] + bias[n] - subtract[n] )
# = sum_k x[b,k] * (sum_n W[n,k]) + sum_n(bias[n]-subtract[n])
# That would be the algebraic shortcut forbidden by safety contract.
#
# So we need to actually execute the GEMM at runtime. Let's do a tiled GEMM that
# computes the full (B, out) matrix then reduces. Or fuse: compute (B, out) tile by tile,
# accumulate the row-sum into a scalar, then write only the scalar.
#
# Use a kernel that processes one batch row per program, tiled over N (out_features),
# and inside accumulates over K (in_features). This is essentially a matvec per row,
# but doing the full work.


@triton.jit
def fused_gemm_mean_kernel(
    X_ptr, W_ptr, B_ptr, S_ptr, OUT_ptr,
    BATCH, IN_F, OUT_F,
    stride_xb, stride_xk,
    stride_wn, stride_wk,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per batch row
    b = tl.program_id(0)
    
    # Pointers to row b of X
    x_row_ptr = X_ptr + b * stride_xb
    
    # Accumulator: scalar sum over n of (W_n . x_b + bias_n - subtract_n)
    row_sum = 0.0
    
    # Loop over N (output features) tiles
    for n_start in range(0, OUT_F, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < OUT_F
        
        # For this tile of N, compute dot product with x for each n
        # Accumulator: [BLOCK_N]
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        
        for k_start in range(0, IN_F, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < IN_F
            
            # Load x[b, k_start:k_start+BLOCK_K]: shape [BLOCK_K]
            x_vals = tl.load(x_row_ptr + offs_k * stride_xk, mask=mask_k, other=0.0)
            
            # Load W[n_start:n_start+BLOCK_N, k_start:k_start+BLOCK_K]: shape [BLOCK_N, BLOCK_K]
            w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_mask = mask_n[:, None] & mask_k[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
            
            # acc += sum_k W[n,k] * x[k]
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
        
        # Add bias and subtract subtract
        bias_vals = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
        sub_vals = tl.load(S_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias_vals - sub_vals
        
        # Mask out invalid n
        acc = tl.where(mask_n, acc, 0.0)
        
        # Sum into row_sum
        row_sum += tl.sum(acc, axis=0)
    
    # Mean over N
    mean_val = row_sum / OUT_F
    
    # LogSumExp over a single element is identity
    # GELU
    # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Or exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu_val = 0.5 * mean_val * (1.0 + tl.math.erf(mean_val * inv_sqrt2))
    
    # Store scalar
    tl.store(OUT_ptr + b, gelu_val)


@triton.jit
def add_scalar_to_rows_kernel(
    X_ptr, S_ptr, OUT_ptr,
    BATCH, FEATS,
    BLOCK: tl.constexpr,
):
    # 2D grid: (batch, feat_tile)
    b = tl.program_id(0)
    t = tl.program_id(1)
    
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < FEATS
    
    x_vals = tl.load(X_ptr + b * FEATS + offs, mask=mask, other=0.0)
    s_val = tl.load(S_ptr + b)
    
    out_vals = x_vals + s_val
    tl.store(OUT_ptr + b * FEATS + offs, out_vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        IN_F = self.in_features
        OUT_F = self.out_features
        
        W = self.gemm.weight.contiguous()  # (out, in)
        bias = self.gemm.bias.contiguous() if self.gemm.bias is not None else torch.zeros(OUT_F, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()
        
        # Scalar per row
        scalar = torch.empty(B, device=x.device, dtype=torch.float32)
        
        BLOCK_K = 128
        BLOCK_N = 64
        
        grid = (B,)
        fused_gemm_mean_kernel[grid](
            x, W, bias, sub, scalar,
            B, IN_F, OUT_F,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        
        # Now scalar has shape (B,), add it to each row of x
        out = torch.empty_like(x)
        BLOCK = 1024
        grid2 = (B, (IN_F + BLOCK - 1) // BLOCK)
        add_scalar_to_rows_kernel[grid2](
            x, scalar, out,
            B, IN_F,
            BLOCK=BLOCK,
            num_warps=4,
        )
        
        return out