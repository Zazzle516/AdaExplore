import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# Tiled GEMM kernel that computes: out = clamp((x @ W^T + b) * scale * 2, clamp_min, clamp_max)
# Then we do a row-wise logsumexp + mish fusion in a second kernel.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_scale_clamp_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    SCALE: tl.constexpr,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is now [K, N] contiguous: stride_wk along K (rows), stride_wn=1 along N
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # scale * 2 (because x = x*scale, x = x+x => 2*scale*x)
    acc = acc * (SCALE * 2.0)

    # clamp
    acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def logsumexp_mish_kernel(
    in_ptr, out_ptr,
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_ptr = in_ptr + pid * stride_m

    # find max
    offs = tl.arange(0, BLOCK_N)
    max_val = tl.full((), -float('inf'), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs
        mask = idx < N
        v = tl.load(row_ptr + idx, mask=mask, other=-float('inf')).to(tl.float32)
        cur_max = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, cur_max)

    # sum exp
    sum_exp = tl.full((), 0.0, dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs
        mask = idx < N
        v = tl.load(row_ptr + idx, mask=mask, other=-float('inf')).to(tl.float32)
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    lse = max_val + tl.log(sum_exp)

    # mish: lse * (lse * tanh(softplus(lse)))
    # softplus(x) = log(1 + exp(x)), stable: max(x,0) + log1p(exp(-|x|))
    sp = tl.where(lse > 0, lse, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(lse)))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    mish_val = lse * tanh_sp
    result = lse * mish_val

    tl.store(out_ptr + pid, result)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._wt_cache = None

    def _get_wt(self):
        # cache the transposed weight on CUDA: shape [K, N] contiguous
        w = self.matmul.weight
        if (self._wt_cache is None
                or self._wt_cache.device != w.device
                or self._wt_cache.shape[0] != w.shape[1]
                or self._wt_cache.shape[1] != w.shape[0]):
            self._wt_cache = w.detach().t().contiguous().cuda()
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        Wt = self._get_wt()  # [K, N]
        b = self.matmul.bias.detach().contiguous().cuda()    # [N]

        M, K = x.shape
        N = Wt.shape[1]

        intermediate = torch.empty((M, N), device=x.device, dtype=torch.float16)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        matmul_scale_clamp_kernel[grid](
            x, Wt, b, intermediate,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            intermediate.stride(0), intermediate.stride(1),
            SCALE=self.scale_factor,
            CMIN=self.clamp_min,
            CMAX=self.clamp_max,
        )

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        BLOCK_N = 2048
        logsumexp_mish_kernel[(M,)](
            intermediate, out,
            M, N,
            intermediate.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )

        return out