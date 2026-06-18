import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_swish_scale_kernel_fp16(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    SCALE: tl.constexpr,
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
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        w_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :].to(tl.float32)
    acc = acc * tl.sigmoid(acc)
    acc = acc * SCALE

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def linear_swish_scale_fp16(x_fp16, w_fp16_t, bias, scale, out_dtype):
    """
    x_fp16: (M, K) fp16
    w_fp16_t: (K, N) fp16 (transposed weight, K-major contiguous along N)
    bias: (N,) fp32
    """
    M, K = x_fp16.shape
    N = w_fp16_t.shape[1]
    out = torch.empty((M, N), device=x_fp16.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    linear_swish_scale_kernel_fp16[grid](
        x_fp16, w_fp16_t, bias, out,
        M, N, K,
        x_fp16.stride(0), x_fp16.stride(1),
        w_fp16_t.stride(1), w_fp16_t.stride(0),  # stride_wn (per N), stride_wk (per K)
        out.stride(0), out.stride(1),
        SCALE=float(scale),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        # Pre-cast weight to fp16 once: shape (K, N), K-major contiguous
        # weight is (out_features, in_features) = (N, K)
        with torch.no_grad():
            w = self.matmul.weight.detach()
            # Transpose to (K, N) and make contiguous
            w_t_fp16 = w.t().contiguous().to(torch.float16)
        self.register_buffer('weight_fp16_t', w_t_fp16, persistent=False)

    def _ensure_buffers(self, device):
        if self.weight_fp16_t.device != device:
            self.weight_fp16_t = self.weight_fp16_t.to(device)

    def forward(self, x):
        x = x.contiguous()
        self._ensure_buffers(x.device)
        x_fp16 = x.to(torch.float16)
        b = self.matmul.bias.contiguous()
        return linear_swish_scale_fp16(x_fp16, self.weight_fp16_t, b, self.scaling_factor, x.dtype)