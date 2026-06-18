import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_P': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32, 'BLOCK_P': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_P': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
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
    pid_pt = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    PK_TOTAL: tl.constexpr = BLOCK_P * POOL_K
    of_start = pid_pt * PK_TOTAL
    of_offs = of_start + tl.arange(0, PK_TOTAL)  # [PK_TOTAL]

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
            return (num_m, num_pt)

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