import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------- Fused GEMM + BN affine + scale ----------------
# Computes: out[m,n] = ((x @ W.T)[m,n] + bias[n] - bn_mean[n]) * bn_scale[n] + bn_shift[n], all multiplied by scale
# where bn_scale[n] = bn_weight[n] / sqrt(bn_var[n] + eps)
# and   bn_shift[n] = bn_bias[n]
# Final per-element: y = scale * (bn_scale * (linear_out - bn_mean) + bn_bias)
#                      = (linear_out) * (scale * bn_scale) + (scale * (bn_bias - bn_scale * bn_mean))
# We precompute: A[n] = scale * bn_scale[n]
#                B[n] = scale * (bn_bias[n] - bn_scale[n] * bn_mean[n])
# But linear_out = x @ W.T + bias. We can also fold bias into B:
#                B'[n] = scale * (bn_bias[n] + bn_scale[n] * (bias[n] - bn_mean[n]))
# Then y = (x @ W.T) * A[n] + B'[n]

GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_gemm_bn_kernel(
    x_ptr, w_ptr, A_ptr, Bp_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k0 in range(0, K, BLOCK_K):
        k_remaining = K - k0
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    A = tl.load(A_ptr + offs_n, mask=mask_n, other=0.0)
    Bp = tl.load(Bp_ptr + offs_n, mask=mask_n, other=0.0)

    out = acc * A[None, :] + Bp[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


# ---------------- Softmax kernel ----------------

@triton.jit
def softmax_kernel(
    x_ptr, out_ptr, n_cols,
    stride_x_row, stride_o_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x_row
    o_row = out_ptr + row * stride_o_row

    # First pass: max
    offs = tl.arange(0, BLOCK)
    max_val = -float('inf')
    for c0 in range(0, n_cols, BLOCK):
        cols = c0 + offs
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=-float('inf'))
        cur_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, cur_max)

    # Second pass: sum of exp
    sum_val = 0.0
    for c0 in range(0, n_cols, BLOCK):
        cols = c0 + offs
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val)
        sum_val += tl.sum(tl.where(mask, e, 0.0), axis=0)

    inv_sum = 1.0 / sum_val

    # Third pass: write
    for c0 in range(0, n_cols, BLOCK):
        cols = c0 + offs
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val) * inv_sum
        tl.store(o_row + cols, e, mask=mask)


def fused_gemm_bn(x, weight, A, Bp):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    fused_gemm_bn_kernel[grid](
        x, weight, A, Bp, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def triton_softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = 2048
    grid = (M,)
    softmax_kernel[grid](
        x, out, N,
        x.stride(0), out.stride(0),
        BLOCK=BLOCK, num_warps=8, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)
        self.in_features = in_features
        self.out_features = out_features

    def _build_fused_params(self):
        # Use BN running stats (eval-mode fold). If training, fall back to actual BN behavior.
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_eps = self.bn.eps
        bn_w = self.bn.weight
        bn_b = self.bn.bias

        bn_scale = bn_w / torch.sqrt(bn_var + bn_eps)  # [N]
        # y = scale * (bn_scale * (linear - bn_mean) + bn_b)
        # linear = x @ W.T + bias
        # => y = (x@W.T) * (scale*bn_scale) + scale * (bn_scale*(bias - bn_mean) + bn_b)
        scale = self.scale  # shape (1,) or broadcastable
        A = (scale * bn_scale).contiguous()
        Bp = (scale * (bn_scale * (self.gemm.bias - bn_mean) + bn_b)).contiguous()
        return A, Bp

    def forward(self, x):
        x = x.cuda().contiguous()
        if self.training:
            # fall back to standard ops to ensure BN running stats updated correctly
            y = self.gemm(x)
            y = self.bn(y)
            y = self.scale * y
            y = self.softmax(y)
            return y

        A, Bp = self._build_fused_params()
        W = self.gemm.weight.contiguous()
        y = fused_gemm_bn(x, W, A, Bp)
        y = triton_softmax(y)
        return y