import torch
import torch.nn as nn
import triton
import triton.language as tl


# Fused GEMM + bias + scale*2 + clamp, then row-wise logsumexp.
# Since hidden_size is large (8192) and we need logsumexp across the full row,
# we'll do this in two passes:
#   Pass 1: compute the matmul row tiles, fused with bias/scale/clamp, and reduce online to find max + sum(exp(x-max)).
#   Actually, simpler: compute the full output Y (batch, hidden) and then logsumexp + mish.

# We'll fuse: matmul -> y = clamp(2 * scale * (x @ W^T + b), cmin, cmax)
# Then a separate kernel for logsumexp(y, dim=1) + mish epilogue.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_clamp_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    SCALE2: tl.constexpr,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # W is (K, N), contiguous along N
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = acc * SCALE2
    acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def logsumexp_mish_kernel(
    y_ptr, out_ptr,
    M, N,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_ptr = y_ptr + pid * stride_ym

    # Single-pass online logsumexp: track running max and sum of exp(x - max)
    max_val = -float('inf')
    sum_exp = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        vals = tl.load(row_ptr + offs * stride_yn, mask=mask, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        new_max = tl.maximum(max_val, block_max)
        # rescale previous sum
        sum_exp = sum_exp * tl.exp(max_val - new_max)
        e = tl.exp(vals - new_max)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)
        max_val = new_max

    lse = max_val + tl.log(sum_exp)

    # Mish: x * mish(x) = x * x * tanh(softplus(x)) = lse * lse * tanh(log(1+exp(lse)))
    sp = tl.log(1.0 + tl.exp(lse))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    mish = lse * tanh_sp
    out = lse * mish

    tl.store(out_ptr + pid, out)


def fused_linear_clamp(x, weight, bias, scale2, cmin, cmax):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    fused_linear_clamp_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        SCALE2=float(scale2),
        CMIN=float(cmin),
        CMAX=float(cmax),
    )
    return y


def logsumexp_mish(y):
    M, N = y.shape
    out = torch.empty((M, 1), device=y.device, dtype=torch.float32)
    BLOCK_N = 2048
    grid = (M,)
    logsumexp_mish_kernel[grid](
        y, out,
        M, N,
        y.stride(0), y.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out


def fused_linear_clamp_kt(x, weight_kt, bias, scale2, cmin, cmax):
    # weight_kt: (K, N) contiguous along N
    M, K = x.shape
    K2, N = weight_kt.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    fused_linear_clamp_kernel[grid](
        x, weight_kt, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_kt.stride(0), weight_kt.stride(1),
        y.stride(0), y.stride(1),
        SCALE2=float(scale2),
        CMIN=float(cmin),
        CMAX=float(cmax),
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # Pre-transpose weight to (K, N) contiguous for efficient B-loads
        self.register_buffer('weight_kt', self.matmul.weight.detach().t().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        bias = self.matmul.bias.contiguous()
        scale2 = self.scale_factor * 2.0
        y = fused_linear_clamp_kt(x, self.weight_kt, bias, scale2, self.clamp_min, self.clamp_max)
        out = logsumexp_mish(y)
        return out