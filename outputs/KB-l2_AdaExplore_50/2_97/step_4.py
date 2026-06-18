import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_gemm_bn_swish_kernel(
    x_ptr, w_ptr, lin_b_ptr,
    scale_ptr, shift_ptr,
    bias_ptr, out_ptr,
    M, N, K,
    inv_div,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is in (K, N) layout
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remain, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remain, other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    lin_b = tl.load(lin_b_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(bias_ptr)

    acc = acc + lin_b[None, :]
    y = acc * scale[None, :] + shift[None, :]
    y = y + b
    y = y * inv_div
    y = y * tl.sigmoid(y)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.divide_value = float(divide_value)

        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        self._wT_cache = None
        self._wT_version = -1

    def _get_wT(self):
        w = self.matmul.weight
        if self._wT_cache is None or self._wT_version != w._version or self._wT_cache.shape[0] != w.shape[1]:
            self._wT_cache = w.t().contiguous()
            self._wT_version = w._version
        return self._wT_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        if self.training:
            y = self.matmul(x)
            y = self.bn(y)
            y = y + self.bias
            y = y / self.divide_value
            y = y * torch.sigmoid(y)
            return y

        wT = self._get_wT()  # (K, N)
        lin_b = self.matmul.bias

        rm = self.bn.running_mean
        rv = self.bn.running_var
        bn_w = self.bn.weight
        bn_b = self.bn.bias
        inv_std = torch.rsqrt(rv + self.bn_eps)
        scale = (bn_w * inv_std).contiguous()
        shift = (bn_b - rm * scale).contiguous()

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        inv_div = 1.0 / self.divide_value

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )

        fused_gemm_bn_swish_kernel[grid](
            x, wT, lin_b,
            scale, shift,
            self.bias, out,
            M, N, K,
            inv_div,
            x.stride(0), x.stride(1),
            wT.stride(0), wT.stride(1),
            out.stride(0), out.stride(1),
            GROUP_M=8,
        )
        return out