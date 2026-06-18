import torch
import torch.nn as nn
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def linear_sub_mul_relu_kernel(
    x_ptr, wt_ptr, beff_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    stride_om, stride_on,
    MUL: tl.constexpr,
    EVEN_K: tl.constexpr,
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

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

    x_ptrs = x_ptr + offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wt_ptrs = wt_ptr + offs_k[:, None] * stride_wtk + offs_bn[None, :] * stride_wtn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            x = tl.load(x_ptrs)
            w = tl.load(wt_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            mask_k = offs_k < k_remaining
            x = tl.load(x_ptrs, mask=mask_k[None, :], other=0.0)
            w = tl.load(wt_ptrs, mask=mask_k[:, None], other=0.0)
        acc = tl.dot(x, w, acc, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        wt_ptrs += BLOCK_K * stride_wtk

    beff = tl.load(beff_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * MUL + beff[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)
        self.in_features = in_features
        self.out_features = out_features
        self._wt_cache = None
        self._beff_cache = None

    def _get_wt(self, device, dtype):
        if (self._wt_cache is None
                or self._wt_cache.device != device
                or self._wt_cache.dtype != dtype
                or self._wt_cache.data_ptr() == 0):
            W = self.linear.weight.detach().to(device=device, dtype=dtype)
            # W is [N, K]; we want W_t of shape [K, N] contiguous (K-major rows of K, N along inner).
            self._wt_cache = W.t().contiguous()
        return self._wt_cache

    def _get_beff(self, device, dtype):
        if (self._beff_cache is None
                or self._beff_cache.device != device
                or self._beff_cache.dtype != dtype):
            b = self.linear.bias.detach().to(device=device, dtype=dtype)
            self._beff_cache = ((b - self.subtract_value) * self.multiply_value).contiguous()
        return self._beff_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        W_t = self._get_wt(x.device, x.dtype)
        beff = self._get_beff(x.device, x.dtype)

        M, K = x.shape
        N = W_t.shape[1]

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

        even_k = (K % 32 == 0)

        linear_sub_mul_relu_kernel[grid](
            x, W_t, beff, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W_t.stride(0), W_t.stride(1),
            out.stride(0), out.stride(1),
            MUL=self.multiply_value,
            EVEN_K=even_k,
        )
        return out