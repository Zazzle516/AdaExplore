import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_swish_kernel(
    A_ptr, B_ptr, bias_lin_ptr, scale_ptr, shift_ptr,
    OUT_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    INV_DIV: tl.constexpr,
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

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add linear bias (shape N)
    bias_lin = tl.load(bias_lin_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_lin[None, :]

    # Apply BN affine fold: y = acc * scale + shift  (where scale/shift already contain extra bias and division)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)
    y = acc * scale[None, :] + shift[None, :]

    # Swish: y * sigmoid(y)
    out = y * tl.sigmoid(y)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
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
        # y_bn = (x - mean) / sqrt(var + eps) * gamma + beta
        # then + extra_bias, then / divide_value
        # combined: y = x * scale + shift
        inv_std = 1.0 / torch.sqrt(var + self.bn_eps)
        gamma = self.bn.weight
        beta = self.bn.bias
        scale = (gamma * inv_std) / self.divide_value
        # extra bias broadcasted to N
        extra = self.bias  # shape bias_shape, typically (1,)
        shift = (beta - mean * gamma * inv_std) / self.divide_value + extra / self.divide_value
        # ensure shift is shape (N,)
        if shift.dim() == 0 or shift.numel() == 1:
            shift = shift.expand(self.out_features).contiguous()
        return scale.contiguous(), shift.contiguous()

    def forward(self, x):
        x = x.cuda().contiguous()
        if x.dtype != torch.float32:
            x = x.float()

        W = self.matmul.weight  # (out, in)
        b_lin = self.matmul.bias  # (out,)

        if self.training:
            # Compute via PyTorch path to update running stats correctly, then fuse activation + division + bias
            y = torch.nn.functional.linear(x, W, b_lin)
            # Update BN running stats and normalize
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

            # B is W^T : (K, N)
            Wt = W.t().contiguous()

            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
            fused_linear_bn_swish_kernel[grid](
                x, Wt, b_lin, scale, shift,
                out,
                M, N, K,
                x.stride(0), x.stride(1),
                Wt.stride(0), Wt.stride(1),
                out.stride(0), out.stride(1),
                INV_DIV=1.0 / self.divide_value,
            )
            return out