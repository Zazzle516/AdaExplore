import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_swish_kernel(
    A_ptr, B_ptr, bias_ptr,
    scale_ptr, shift_ptr,
    Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    INV_DIV: tl.constexpr,
    EVEN_K: tl.constexpr,
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

    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    # Load scale and shift (per output column N)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)
    bias_extra = tl.load(bias_ptr)  # scalar bias

    y = acc * scale[None, :] + shift[None, :] + bias_extra
    y = y * INV_DIV
    # Swish: y * sigmoid(y)
    y = y * tl.sigmoid(y)

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
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

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        if self.training:
            # Run reference path during training to update BN running stats correctly
            y = self.matmul(x)
            y = self.bn(y)
            y = y + self.bias
            y = y / self.divide_value
            y = y * torch.sigmoid(y)
            return y

        # Eval mode: fuse Linear + BN(running) + bias + div + swish
        W = self.matmul.weight  # (N, K)
        b_lin = self.matmul.bias  # (N,)

        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias

        # BN: out = (lin - mean) / sqrt(var + eps) * gamma + beta
        # = lin * (gamma / sqrt(var+eps)) + (beta - mean * gamma / sqrt(var+eps))
        inv_std = torch.rsqrt(running_var + self.bn_eps)
        scale = (gamma * inv_std).contiguous()  # (N,)
        shift = (beta - running_mean * gamma * inv_std).contiguous()  # = bn(lin) when lin=0
        # add linear bias contribution into shift: (lin_bias) * scale + shift
        shift = (b_lin * scale + shift).contiguous()

        # Use W directly (N, K). Treat as B with shape (K, N) via strides:
        # B[k, n] = W[n, k] => stride_bk = 1, stride_bn = K (= W.stride(0))
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        bias_extra = self.bias.view(-1)[0:1].contiguous()
        inv_div = 1.0 / self.divide_value
        even_k = (K % 64 == 0)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        fused_linear_bn_swish_kernel[grid](
            x, W, bias_extra,
            scale, shift,
            out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(1), W.stride(0),  # stride_bk, stride_bn
            out.stride(0), out.stride(1),
            INV_DIV=inv_div,
            EVEN_K=even_k,
        )
        return out