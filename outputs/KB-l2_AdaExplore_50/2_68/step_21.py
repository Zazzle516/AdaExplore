import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 2}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_iters = tl.cdiv(K, BLOCK_K * SPLIT_K)
    for k in range(0, k_iters):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * SPLIT_K * stride_xk
        w_ptrs += BLOCK_K * SPLIT_K * stride_wk

    offs_m_o = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_o = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + offs_m_o[:, None] * stride_om + offs_n_o[None, :] * stride_on
    out_mask = (offs_m_o[:, None] < M) & (offs_n_o[None, :] < N)

    if SPLIT_K == 1:
        b = tl.load(b_ptr + offs_n_o, mask=offs_n_o < N, other=0.0)
        acc += b[None, :]
        acc = tl.minimum(acc, C) - C
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        tl.atomic_add(out_ptrs, acc, mask=out_mask)


@triton.jit
def epilogue_kernel(
    out_ptr, b_ptr,
    M, N,
    stride_om, stride_on,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    val = tl.load(out_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    val += b[None, :]
    val = tl.minimum(val, C) - C
    tl.store(out_ptrs, val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.in_features = in_features
        self.out_features = out_features
        with torch.no_grad():
            w_t = self.linear.weight.detach().t().contiguous()
        self.register_buffer('weight_t', w_t)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()

        w = self.weight_t
        if w.device != x.device:
            w = w.to(x.device)
            self.weight_t = w

        b = self.linear.bias.contiguous()
        c = float(self.constant.detach().item())

        M, K = x.shape
        N = self.out_features

        def grid(meta):
            return (
                triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
                meta['SPLIT_K'],
            )

        # Pre-fill output with zeros for atomic_add path; if SPLIT_K==1 the kernel overwrites.
        out = torch.zeros((M, N), device=x.device, dtype=x.dtype)

        fused_gemm_kernel[grid](
            x, w, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            c,
        )

        # Check which config was used to decide on epilogue
        best_config = fused_gemm_kernel.best_config
        split_k = best_config.kwargs['SPLIT_K']
        if split_k > 1:
            BM, BN = 64, 256
            grid2 = (triton.cdiv(M, BM), triton.cdiv(N, BN))
            epilogue_kernel[grid2](
                out, b, M, N,
                out.stride(0), out.stride(1),
                c, BM, BN,
            )
        return out