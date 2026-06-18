import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_swish_kernel(
    A_ptr, W_ptr, bias_lin_ptr, scale_ptr, shift_ptr,
    OUT_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
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

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # W has shape (N, K) with stride_wn=K, stride_wk=1; load as (K, N) tile by transposing
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(w_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    offs_m2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m2 < M
    mask_n = offs_n2 < N

    bias_lin = tl.load(bias_lin_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)

    y = (acc + bias_lin[None, :]) * scale[None, :] + shift[None, :]
    out = y * tl.sigmoid(y)

    out_ptrs = OUT_ptr + offs_m2[:, None] * stride_om + offs_n2[None, :] * stride_on
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
        self.bn_momentum = bn_momentum

    def _compute_fold(self, mean, var):
        inv_std = 1.0 / torch.sqrt(var + self.bn_eps)
        gamma = self.bn.weight
        beta = self.bn.bias
        inv_div = 1.0 / self.divide_value
        scale = gamma * inv_std * inv_div
        extra = self.bias
        shift = (beta - mean * gamma * inv_std) * inv_div + extra * inv_div
        if shift.dim() == 0 or shift.numel() == 1:
            shift = shift.expand(self.out_features).contiguous()
        return scale.contiguous(), shift.contiguous()

    def forward(self, x):
        x = x.cuda().contiguous()
        if x.dtype != torch.float32:
            x = x.float()

        W = self.matmul.weight  # (N, K)
        b_lin = self.matmul.bias  # (N,)

        if self.training:
            y = torch.nn.functional.linear(x, W, b_lin)
            y = self.bn(y)
            y = y + self.bias
            y = y / self.divide_value
            y = y * torch.sigmoid(y)
            return y
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var
            scale, shift = self._compute_fold(mean, var)

            M, K = x.shape
            N = self.out_features
            out = torch.empty((M, N), device=x.device, dtype=torch.float32)

            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
            fused_linear_bn_swish_kernel[grid](
                x, W, b_lin, scale, shift,
                out,
                M, N, K,
                x.stride(0), x.stride(1),
                W.stride(0), W.stride(1),
                out.stride(0), out.stride(1),
            )
            return out