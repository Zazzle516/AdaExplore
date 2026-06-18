import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64,  'BLOCK_P': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64,  'BLOCK_P': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128, 'BLOCK_P': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128, 'BLOCK_P': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64,  'BLOCK_P': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64,  'BLOCK_P': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 256, 'BLOCK_P': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128,'BLOCK_K': 64,  'BLOCK_P': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128,'BLOCK_K': 128, 'BLOCK_P': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128,'BLOCK_K': 64,  'BLOCK_P': 32}, num_warps=8, num_stages=3),
    ],
    key=['IN_FEATURES', 'POOLED_LEN', 'POOL_K'],
)
@triton.jit
def fused_persistent_kernel(
    x_ptr,         # [B, IN_FEATURES]
    wt_ptr,        # [IN_FEATURES, OUT_FEATURES]
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
):
    pid_m = tl.program_id(0)

    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    PK_TOTAL: tl.constexpr = BLOCK_P * POOL_K
    NUM_PT: tl.constexpr = POOLED_LEN // BLOCK_P

    # Preload row tile of x in chunks (re-loaded per pt iter is wasteful; load once per K-block reused across pt)
    # Strategy: outer loop pt, inner loop K. But we want to reuse x across pt. So swap: outer K, inner pt.
    # However we need to accumulate per-pt separately. Use running max instead.
    running_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)

    for pt in range(0, NUM_PT):
        of_start = pt * PK_TOTAL
        of_offs = of_start + tl.arange(0, PK_TOTAL)

        acc = tl.zeros((BLOCK_M, PK_TOTAL), dtype=tl.float32)
        for k0 in range(0, IN_FEATURES, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offs < IN_FEATURES

            x_ptrs = x_ptr + rows[:, None] * IN_FEATURES + k_offs[None, :]
            x_mask = row_mask[:, None] & k_mask[None, :]
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_ptrs = wt_ptr + k_offs[:, None] * OUT_FEATURES + of_offs[None, :]
            w_tile = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)

            acc += tl.dot(x_tile, w_tile)

        bias = tl.load(b_ptr + of_offs)
        acc = acc + bias[None, :]

        acc_r = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL_K))
        avg = tl.sum(acc_r, axis=2) / POOL_K

        inv_sqrt2 = 0.70710678118654752440
        gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))
        scaled = gelu * SCALE
        local_max = tl.max(scaled, axis=1)
        running_max = tl.maximum(running_max, local_max)

    out_ptrs = out_ptr + rows
    tl.store(out_ptrs, running_max, mask=row_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        self.in_features = in_features
        self.out_features = out_features
        self.pooled_len = out_features // pool_kernel_size
        self._wt_cache = None
        self._wt_data_ptr = None

    def _get_wt(self):
        w = self.matmul.weight
        if (self._wt_cache is None
            or self._wt_data_ptr != w.data_ptr()
            or self._wt_cache.device != w.device):
            self._wt_cache = w.detach().t().contiguous()
            if not self._wt_cache.is_cuda:
                self._wt_cache = self._wt_cache.cuda()
            self._wt_data_ptr = w.data_ptr()
        return self._wt_cache

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        B = x.shape[0]
        wt = self._get_wt()
        b = self.matmul.bias
        if not b.is_cuda:
            b = b.cuda()
        b = b.contiguous()

        out = torch.empty((B,), device=x.device, dtype=torch.float32)

        def grid(meta):
            num_m = triton.cdiv(B, meta['BLOCK_M'])
            return (num_m,)

        fused_persistent_kernel[grid](
            x, wt, b, out,
            B,
            self.in_features,
            self.out_features,
            self.pooled_len,
            self.pool_kernel_size,
            self.scale_factor,
        )
        return out