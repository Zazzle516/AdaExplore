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
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['IN_FEAT', 'OUT_FEAT'],
)
@triton.jit
def fused_kernel(
    x_ptr, wT_ptr, b_ptr, out_ptr,
    M, IN_FEAT: tl.constexpr, OUT_FEAT: tl.constexpr,
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

    mask_m = offs_m < M

    # x: [M, IN_FEAT]
    # wT: [IN_FEAT, OUT_FEAT]
    x_ptrs = x_ptr + offs_m[:, None] * IN_FEAT + offs_k[None, :]
    w_ptrs = wT_ptr + offs_k[:, None] * OUT_FEAT + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, IN_FEAT, BLOCK_K):
        x_vals = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        w_vals = tl.load(w_ptrs)
        acc += tl.dot(x_vals, w_vals)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * OUT_FEAT

    # Add bias
    b_vals = tl.load(b_ptr + offs_n)  # [BLOCK_N]
    acc += b_vals[None, :]

    # Maxpool over groups of KERNEL_SIZE along N axis
    # Reshape [BLOCK_M, BLOCK_N] -> [BLOCK_M, BLOCK_N // KERNEL_SIZE, KERNEL_SIZE]
    reshaped = tl.reshape(acc, (BLOCK_M, BLOCK_N // KERNEL_SIZE, KERNEL_SIZE))
    pooled = tl.max(reshaped, axis=2)  # [BLOCK_M, BLOCK_N // KERNEL_SIZE]
    # Sum along N
    partial = tl.sum(pooled, axis=1)  # [BLOCK_M]
    partial = partial * SCALE

    # Atomic add into out[offs_m]
    tl.atomic_add(out_ptr + offs_m, partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)
        self._wT_cache = None
        self._wT_version = -1

    def _get_wT(self):
        w = self.matmul.weight
        if (self._wT_cache is None
            or self._wT_cache.device != w.device
            or self._wT_version != w._version):
            self._wT_cache = w.t().contiguous()
            self._wT_version = w._version
        return self._wT_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        if self.matmul.weight.device != x.device:
            self.matmul.to(x.device)
        wT = self._get_wT()
        b = self.matmul.bias.contiguous()
        batch_size = x.shape[0]
        out = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_M']),
                             triton.cdiv(self.out_features, meta['BLOCK_N']))
        fused_kernel[grid](
            x, wT, b, out,
            batch_size,
            self.in_features,
            self.out_features,
            self.kernel_size,
            self.scale_factor,
        )
        return out