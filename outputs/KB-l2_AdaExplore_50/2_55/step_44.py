import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['IN_FEATURES', 'OUT_FEATURES'],
)
@triton.jit
def stage1_kernel(
    x_ptr, wt_ptr, b_ptr, pooled_ptr,
    BATCH, IN_FEATURES, OUT_FEATURES, N_GROUPS,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(BATCH, BLOCK_M)
    num_pid_n = tl.cdiv(OUT_FEATURES, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < BATCH
    n_mask = offs_n < OUT_FEATURES

    # x: [BATCH, IN_FEATURES], row-major
    # wt: [IN_FEATURES, OUT_FEATURES], row-major (transposed weight)
    a_ptrs = x_ptr + offs_m[:, None] * IN_FEATURES + offs_k[None, :]
    b_ptrs = wt_ptr + offs_k[:, None] * OUT_FEATURES + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # BATCH=128, OUT_FEATURES=32768, IN_FEATURES=32768 divisible by all tile sizes -> no masks
    for k_start in range(0, IN_FEATURES, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * OUT_FEATURES

    # add bias
    b_vals = tl.load(b_ptr + offs_n)
    acc = acc + b_vals[None, :]

    # max pool over kernel_size=2 along N axis
    # reshape [BLOCK_M, BLOCK_N] -> [BLOCK_M, BLOCK_N/KERNEL_SIZE, KERNEL_SIZE]
    POOLED_N: tl.constexpr = BLOCK_N // KERNEL_SIZE
    acc3 = tl.reshape(acc, (BLOCK_M, POOLED_N, KERNEL_SIZE))
    pooled = tl.max(acc3, axis=2)  # [BLOCK_M, POOLED_N]

    # write to pooled buffer [BATCH, N_GROUPS]
    offs_g = pid_n * POOLED_N + tl.arange(0, POOLED_N)
    g_mask = offs_g < N_GROUPS
    out_ptrs = pooled_ptr + offs_m[:, None] * N_GROUPS + offs_g[None, :]
    tl.store(out_ptrs, pooled, mask=m_mask[:, None] & g_mask[None, :])


@triton.jit
def stage2_kernel(
    pooled_ptr, out_ptr,
    BATCH, N_GROUPS,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    row_ptr = pooled_ptr + pid_b * N_GROUPS

    offs = tl.arange(0, BLOCK)
    mask = offs < N_GROUPS
    v = tl.load(row_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(v, axis=0) * SCALE
    tl.store(out_ptr + pid_b, s)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)
        self._wt_cache = None
        self._wt_version = None

    def _get_wt(self, device):
        w = self.matmul.weight
        if (self._wt_cache is None or
                self._wt_cache.device != device or
                self._wt_version != w._version):
            self._wt_cache = w.detach().to(device).t().contiguous()
            self._wt_version = w._version
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        device = x.device
        wt = self._get_wt(device)
        b = self.matmul.bias.to(device).contiguous()
        B = x.shape[0]

        assert self.out_features % self.kernel_size == 0
        n_groups = self.out_features // self.kernel_size

        pooled = torch.empty((B, n_groups), device=device, dtype=torch.float32)
        out = torch.empty(B, device=device, dtype=torch.float32)

        def grid1(meta):
            return (triton.cdiv(B, meta['BLOCK_M']) * triton.cdiv(self.out_features, meta['BLOCK_N']),)

        stage1_kernel[grid1](
            x, wt, b, pooled,
            B, self.in_features, self.out_features, n_groups,
            KERNEL_SIZE=self.kernel_size,
        )

        # N_GROUPS=16384 fits in a single block
        BLOCK2 = triton.next_power_of_2(n_groups)
        stage2_kernel[(B,)](
            pooled, out,
            B, n_groups,
            SCALE=self.scale_factor,
            BLOCK=BLOCK2,
            num_warps=8,
            num_stages=2,
        )
        return out