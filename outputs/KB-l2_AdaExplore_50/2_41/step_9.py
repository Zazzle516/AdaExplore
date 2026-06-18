import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bn_gelu_relu_kernel(
    A_ptr, B_ptr, C_ptr,
    bias_ptr, scale_ptr, shift_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias add
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # batchnorm: scale * x + shift
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * scale[None, :] + shift[None, :]

    # GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    # ReLU after GELU
    out = tl.maximum(gelu, 0.0)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def fused_gemm_bn_gelu_relu(x, weight_t, bias, scale, shift):
    M, K = x.shape
    N = weight_t.shape[1]
    # weight_t is (K, N) row-major contiguous
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    gemm_bn_gelu_relu_kernel[grid](
        x, weight_t, out,
        bias, scale, shift,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        gemm_bias = self.gemm.bias.contiguous()

        bn = self.batch_norm
        if self.training:
            # fall back to reference path during training to update running stats
            x = self.gemm(x)
            x = self.batch_norm(x)
            x = F.gelu(x)
            x = torch.relu(x)
            return x

        # cache pre-transposed weight (K, N) row-major contiguous
        w = self.gemm.weight
        cache = getattr(self, '_wt_cache', None)
        if cache is None or cache[0] is not w:
            wt = w.t().contiguous()
            self._wt_cache = (w, wt)
        weight_t = self._wt_cache[1]

        # eval mode: fold BN into affine, cache scale/shift
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps
        bn_w = bn.weight if bn.weight is not None else None
        bn_b = bn.bias if bn.bias is not None else None

        ss_cache = getattr(self, '_ss_cache', None)
        key = (running_mean._version, running_var._version,
               bn_w._version if bn_w is not None else -1,
               bn_b._version if bn_b is not None else -1,
               id(running_mean), id(running_var))
        if ss_cache is None or ss_cache[0] != key:
            bw = bn_w if bn_w is not None else torch.ones_like(running_mean)
            bb = bn_b if bn_b is not None else torch.zeros_like(running_mean)
            inv_std = torch.rsqrt(running_var + eps)
            scale = (bw * inv_std).contiguous()
            shift = (bb - running_mean * scale).contiguous()
            self._ss_cache = (key, scale, shift)
        scale = self._ss_cache[1]
        shift = self._ss_cache[2]

        return fused_gemm_bn_gelu_relu(x, weight_t, gemm_bias, scale, shift)