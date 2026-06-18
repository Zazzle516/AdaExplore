import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Fused kernel: per (batch_row, out_tile of pooled dim).
# For each row, compute matmul output for K consecutive output features (K = pool_kernel_size),
# average them, apply GELU, scale, and reduce max over the pooled dim.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32, 'BLOCK_P': 8,  'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'BLOCK_P': 8,  'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_P': 8,  'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32, 'BLOCK_P': 8,  'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 8,  'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 16, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_P': 16, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'BLOCK_P': 16, 'GROUP_M': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'BLOCK_P': 8, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_P': 8, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['IN_FEATURES', 'POOLED_LEN', 'POOL_K'],
)
@triton.jit
def fused_kernel(
    x_ptr,         # [B, IN_FEATURES]
    wt_ptr,        # [IN_FEATURES, OUT_FEATURES]  (pre-transposed)
    b_ptr,         # [OUT_FEATURES]
    out_ptr,       # [B]
    B,
    IN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    POOLED_LEN: tl.constexpr,
    POOL_K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_m = tl.cdiv(B, BLOCK_M)
    num_pt = POOLED_LEN // BLOCK_P  # exact division expected

    # GROUP_M swizzle over (m, pt) so groups of m share weight tiles in L2
    num_pid_in_group = GROUP_M * num_pt
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_pt = (pid % num_pid_in_group) // group_size_m

    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    PK_TOTAL: tl.constexpr = BLOCK_P * POOL_K
    of_start = pid_pt * PK_TOTAL
    of_offs = of_start + tl.arange(0, PK_TOTAL)  # [PK_TOTAL]

    acc = tl.zeros((BLOCK_M, PK_TOTAL), dtype=tl.float32)

    # Inner GEMM loop: x [BLOCK_M, BLOCK_K] @ wt [BLOCK_K, PK_TOTAL]
    for k0 in range(0, IN_FEATURES, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IN_FEATURES

        x_ptrs = x_ptr + rows[:, None] * IN_FEATURES + k_offs[None, :]
        x_mask = row_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # wt is [IN_FEATURES, OUT_FEATURES], contiguous along OUT_FEATURES
        w_ptrs = wt_ptr + k_offs[:, None] * OUT_FEATURES + of_offs[None, :]
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)

        acc += tl.dot(x_tile, w_tile)

    bias = tl.load(b_ptr + of_offs)
    acc = acc + bias[None, :]

    acc = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL_K))
    avg = tl.sum(acc, axis=2) / POOL_K  # [BLOCK_M, BLOCK_P]

    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))
    scaled = gelu * SCALE
    local_max = tl.max(scaled, axis=1)  # [BLOCK_M]

    out_ptrs = out_ptr + rows
    tl.atomic_max(out_ptrs, local_max, mask=row_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        self.in_features = in_features
        self.out_features = out_features
        self.pooled_len = out_features // pool_kernel_size
        # Pre-transpose weight: [OUT_FEATURES, IN_FEATURES] -> [IN_FEATURES, OUT_FEATURES]
        self.register_buffer('_wt', None, persistent=False)

    def _get_wt(self):
        if self._wt is None or self._wt.device != self.matmul.weight.device:
            self._wt = self.matmul.weight.detach().t().contiguous().cuda()
        return self._wt

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        wt = self._get_wt()
        b = self.matmul.bias.contiguous()

        out = torch.full((B,), float('-inf'), device=x.device, dtype=torch.float32)

        def grid(meta):
            num_m = triton.cdiv(B, meta['BLOCK_M'])
            num_pt = self.pooled_len // meta['BLOCK_P']
            return (num_m * num_pt,)

        fused_kernel[grid](
            x, wt, b, out,
            B,
            self.in_features,
            self.out_features,
            self.pooled_len,
            self.pool_kernel_size,
            self.scale_factor,
        )
        return out