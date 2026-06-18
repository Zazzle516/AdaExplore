import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_bias_swish_kernel(
    X_ptr, Wt_ptr, Lbias_ptr,
    Scale_ptr, Shift_ptr,
    Bias_ptr,
    Out_ptr,
    M, N, K,
    INV_DIV: tl.constexpr,
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = Wt_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        mask_k = offs_k < k_remain
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add linear bias
    lb = tl.load(Lbias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += lb[None, :]

    # batch norm: scale * x + shift (precomputed for eval)
    scale = tl.load(Scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(Shift_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * scale[None, :] + shift[None, :]

    # add extra bias (scalar broadcast over N when bias_shape=(1,))
    b = tl.load(Bias_ptr)
    acc = acc + b

    # divide
    acc = acc * INV_DIV

    # swish: x * sigmoid(x)
    out = acc * tl.sigmoid(acc)

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = float(divide_value)
        self.bn_eps = bn_eps
        self._wt_cache = None
        self._wt_version = None

    def _get_wt(self):
        W = self.matmul.weight
        ver = W._version
        if self._wt_cache is None or self._wt_version != ver or self._wt_cache.device != W.device:
            self._wt_cache = W.t().contiguous()
            self._wt_version = ver
        return self._wt_cache

    def forward(self, x):
        x = x.cuda().contiguous()
        if self.training:
            # fall back to standard path to keep running stats correct
            y = self.matmul(x)
            y = self.bn(y)
            y = y + self.bias
            y = y / self.divide_value
            y = y * torch.sigmoid(y)
            return y

        M, K = x.shape
        N = self.out_features
        Wt = self._get_wt()  # (K, N) contiguous
        Lbias = self.matmul.bias  # (N,)

        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        bn_w = self.bn.weight
        bn_b = self.bn.bias

        invstd = torch.rsqrt(running_var + self.bn_eps)
        scale = bn_w * invstd
        shift = bn_b - running_mean * scale

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        inv_div = 1.0 / self.divide_value

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

        fused_linear_bn_bias_swish_kernel[grid](
            x, Wt, Lbias,
            scale, shift,
            self.bias,
            out,
            M, N, K,
            inv_div,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            out.stride(0), out.stride(1),
        )
        return out