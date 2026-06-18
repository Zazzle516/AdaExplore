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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['BATCH', 'IN_F', 'OUT_F'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    X_ptr, W_ptr, B_ptr, S_ptr, ROW_SUM_ptr,
    BATCH, IN_F, OUT_F,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(BATCH, BLOCK_M)
    num_pid_n = tl.cdiv(OUT_F, BLOCK_N)
    
    # GROUP_M swizzle
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # X pointers: [BLOCK_M, BLOCK_K]
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W pointers: W is (OUT_F, IN_F). We want b = W^T tile [BLOCK_K, BLOCK_N] where b[k,n] = W[n,k]
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    
    mask_m = offs_m < BATCH
    mask_n = offs_n < OUT_F
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, IN_F, BLOCK_K):
        k_remain = IN_F - k_start
        mask_k = offs_k < k_remain
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # Add bias and subtract subtract per column
    bias_vals = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    sub_vals = tl.load(S_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + (bias_vals - sub_vals)[None, :]
    
    # Zero out invalid columns
    acc = tl.where(mask_n[None, :], acc, 0.0)
    
    # Row-sum reduction over N tile -> [BLOCK_M]
    partial = tl.sum(acc, axis=1)
    
    # Atomic add into row_sum[m]
    tl.atomic_add(ROW_SUM_ptr + offs_m, partial, mask=mask_m)


@triton.jit
def epilogue_kernel(
    X_ptr, ROW_SUM_ptr, OUT_ptr,
    BATCH, FEATS,
    INV_OUT_F,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    
    # Compute scalar for this row
    row_sum = tl.load(ROW_SUM_ptr + b)
    mean_val = row_sum * INV_OUT_F
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu_val = 0.5 * mean_val * (1.0 + tl.math.erf(mean_val * inv_sqrt2))
    
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < FEATS
    
    x_vals = tl.load(X_ptr + b * FEATS + offs, mask=mask, other=0.0)
    out_vals = x_vals + gelu_val
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
        
        # Row sum accumulator (zero-init for atomic adds)
        row_sum = torch.zeros(B, device=x.device, dtype=torch.float32)
        
        grid = lambda META: (triton.cdiv(B, META['BLOCK_M']) * triton.cdiv(OUT_F, META['BLOCK_N']),)
        fused_gemm_rowsum_kernel[grid](
            x, W, bias, sub, row_sum,
            B, IN_F, OUT_F,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )
        
        # Epilogue: out[b, f] = x[b, f] + gelu(row_sum[b] / OUT_F)
        out = torch.empty_like(x)
        BLOCK = 1024
        grid2 = (B, (IN_F + BLOCK - 1) // BLOCK)
        inv_out_f = 1.0 / float(OUT_F)
        epilogue_kernel[grid2](
            x, row_sum, out,
            B, IN_F,
            inv_out_f,
            BLOCK=BLOCK,
            num_warps=4,
        )
        
        return out