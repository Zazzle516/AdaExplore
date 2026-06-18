import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['IN_FEAT', 'OUT_FEAT'],
)
@triton.jit
def fused_kernel(
    x_ptr, wt_ptr, b_ptr, out_ptr,
    BATCH: tl.constexpr,
    IN_FEAT: tl.constexpr,
    OUT_FEAT: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < BATCH

    # wt has shape [IN_FEAT, OUT_FEAT], K-contiguous? No, N-contiguous (row-major after transpose).
    # x has shape [BATCH, IN_FEAT]
    x_ptrs = x_ptr + offs_m[:, None] * IN_FEAT + offs_k[None, :]
    wt_ptrs = wt_ptr + offs_k[:, None] * OUT_FEAT + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, IN_FEAT, BLOCK_K):
        x_vals = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w_vals = tl.load(wt_ptrs)
        acc += tl.dot(x_vals, w_vals)
        x_ptrs += BLOCK_K
        wt_ptrs += BLOCK_K * OUT_FEAT

    # Add bias
    b_vals = tl.load(b_ptr + offs_n)
    acc += b_vals[None, :]

    # Maxpool over KERNEL_SIZE groups along N
    # Reshape [BLOCK_M, BLOCK_N] -> [BLOCK_M, BLOCK_N // KERNEL_SIZE, KERNEL_SIZE]
    reshaped = tl.reshape(acc, (BLOCK_M, BLOCK_N // KERNEL_SIZE, KERNEL_SIZE))
    pooled = tl.max(reshaped, axis=2)  # [BLOCK_M, BLOCK_N // KERNEL_SIZE]
    partial = tl.sum(pooled, axis=1) * SCALE  # [BLOCK_M]

    # Atomic add into out[offs_m]
    tl.atomic_add(out_ptr + offs_m, partial, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        # Linear layer for parameters
        self.matmul = nn.Linear(in_features, out_features)
        self._wt_cache = None

    def _get_wt(self):
        # Pre-transpose weight to [IN_FEAT, OUT_FEAT] for K-contiguous-in-A, N-contiguous-in-B dot.
        w = self.matmul.weight  # [OUT_FEAT, IN_FEAT]
        if (self._wt_cache is None
                or self._wt_cache.device != w.device
                or self._wt_cache.dtype != w.dtype):
            self._wt_cache = w.t().contiguous()
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        wt = self._get_wt()
        b = self.matmul.bias.contiguous()
        batch_size = x.shape[0]
        out = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        def grid(meta):
            return (
                triton.cdiv(batch_size, meta['BLOCK_M']),
                triton.cdiv(self.out_features, meta['BLOCK_N']),
            )

        fused_kernel[grid](
            x, wt, b, out,
            batch_size,
            self.in_features,
            self.out_features,
            self.kernel_size,
            self.scale_factor,
        )
        return out