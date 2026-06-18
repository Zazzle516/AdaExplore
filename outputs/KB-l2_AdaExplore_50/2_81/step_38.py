import torch
import torch.nn as nn
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def fused_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
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
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # Swish: x * sigmoid(x)
    acc = acc * tl.sigmoid(acc)
    # divide by 2
    acc = acc * 0.5
    # clamp [-1, 1]
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)
    # tanh: since input in [-0.5, 0.5], tanh stays in (-1, 1), but compute it
    # tanh(x) = (exp(2x)-1)/(exp(2x)+1)
    e2x = tl.exp(2.0 * acc)
    acc = (e2x - 1.0) / (e2x + 1.0)
    # final clamp - tanh output already in (-1,1) so no-op but apply for safety
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self._weight_t_cache = None
        self._bias_cache = None

    def _get_weight_t(self, device, dtype):
        w = self.gemm.weight
        if (self._weight_t_cache is None or
            self._weight_t_cache.device != device or
            self._weight_t_cache.dtype != dtype or
            self._weight_t_cache.data_ptr() == 0):
            # Pre-transpose to contiguous (K, N)
            self._weight_t_cache = w.detach().to(device=device, dtype=dtype).t().contiguous()
        return self._weight_t_cache

    def _get_bias(self, device, dtype):
        if self.gemm.bias is not None:
            if (self._bias_cache is None or
                self._bias_cache.device != device or
                self._bias_cache.dtype != dtype):
                self._bias_cache = self.gemm.bias.detach().to(device=device, dtype=dtype).contiguous()
            return self._bias_cache
        else:
            if self._bias_cache is None:
                self._bias_cache = torch.zeros(self.out_features, device=device, dtype=dtype)
            return self._bias_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        device = x.device
        dtype = x.dtype
        w_t = self._get_weight_t(device, dtype)  # (K, N) contiguous
        bias = self._get_bias(device, dtype)

        M, K = x.shape
        N = w_t.shape[1]

        out = torch.empty((M, N), device=device, dtype=dtype)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

        fused_gemm_kernel[grid](
            x, w_t, bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w_t.stride(0), w_t.stride(1),
            out.stride(0), out.stride(1),
        )
        return out