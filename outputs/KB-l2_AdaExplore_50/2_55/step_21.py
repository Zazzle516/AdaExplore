import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    x_ptr, wt_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
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

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = wt_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        a = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N_GROUPS'],
)
@triton.jit
def pool_sum_kernel(
    y_ptr, out_ptr,
    BATCH, OUT_FEATURES, N_GROUPS,
    KERNEL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    row_ptr = y_ptr + pid_b * OUT_FEATURES

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k_start in range(0, N_GROUPS, BLOCK):
        offs_g = k_start + tl.arange(0, BLOCK)
        mask_g = offs_g < N_GROUPS
        # load KERNEL_SIZE values per group and take max
        # offsets into Y: g * KERNEL_SIZE + j
        # We assume KERNEL_SIZE is small (constexpr); unroll via tl.max over a 2D load.
        offs_y = offs_g[:, None] * KERNEL_SIZE + tl.arange(0, KERNEL_SIZE)[None, :]
        y_mask = mask_g[:, None] & (tl.arange(0, KERNEL_SIZE)[None, :] < KERNEL_SIZE)
        vals = tl.load(row_ptr + offs_y, mask=y_mask, other=-float('inf'))
        pooled = tl.max(vals, axis=1)
        # mask out-of-range groups
        pooled = tl.where(mask_g, pooled, 0.0)
        acc += pooled

    s = tl.sum(acc, axis=0) * SCALE
    tl.store(out_ptr + pid_b, s)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)
        # Cache W^T lazily on first forward (after .cuda())
        self._wt_cache = None

    def _get_wt(self, device):
        W = self.matmul.weight  # [OUT_FEATURES, IN_FEATURES]
        if (self._wt_cache is None
                or self._wt_cache.device != device
                or self._wt_cache.data_ptr() == 0):
            self._wt_cache = W.detach().to(device).t().contiguous()
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        device = x.device
        WT = self._get_wt(device)  # [IN_FEATURES, OUT_FEATURES]
        bias = self.matmul.bias.to(device).contiguous()
        B = x.shape[0]
        M, K, N = B, self.in_features, self.out_features

        assert self.out_features % self.kernel_size == 0
        n_groups = self.out_features // self.kernel_size

        Y = torch.empty((B, N), device=device, dtype=torch.float32)
        out = torch.empty(B, device=device, dtype=torch.float32)

        def grid_gemm(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

        gemm_kernel[grid_gemm](
            x, WT, bias, Y,
            M, N, K,
            x.stride(0), x.stride(1),
            WT.stride(0), WT.stride(1),
            Y.stride(0), Y.stride(1),
        )

        pool_sum_kernel[(B,)](
            Y, out,
            B, N, n_groups,
            KERNEL_SIZE=self.kernel_size,
            SCALE=self.scale_factor,
        )
        return out