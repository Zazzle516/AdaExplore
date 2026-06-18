import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_bias_swish_kernel_bf16(
    X_ptr, W_ptr, Lbias_ptr,
    Scale_ptr, Shift_ptr,
    Bias_ptr,
    Out_ptr,
    M, N, K,
    INV_DIV: tl.constexpr,
    stride_xm, stride_xk,
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        mask_k = offs_k < k_remain
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w, out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    lb = tl.load(Lbias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += lb[None, :]

    scale = tl.load(Scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(Shift_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * scale[None, :] + shift[None, :]

    b = tl.load(Bias_ptr).to(tl.float32)
    acc = acc + b

    acc = acc * INV_DIV
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
        self._cached_W_bf16 = None
        self._cached_Lbias = None
        self._cached_scale = None
        self._cached_shift = None

    def _build_cache(self, device):
        W = self.matmul.weight.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        Lbias = self.matmul.bias.detach().to(device=device, dtype=torch.float32).contiguous()
        running_mean = self.bn.running_mean.to(device=device, dtype=torch.float32)
        running_var = self.bn.running_var.to(device=device, dtype=torch.float32)
        bn_w = self.bn.weight.to(device=device, dtype=torch.float32)
        bn_b = self.bn.bias.to(device=device, dtype=torch.float32)
        invstd = torch.rsqrt(running_var + self.bn_eps)
        scale = (bn_w * invstd).contiguous()
        shift = (bn_b - running_mean * scale).contiguous()
        self._cached_W_bf16 = W
        self._cached_Lbias = Lbias
        self._cached_scale = scale
        self._cached_shift = shift

    def forward(self, x):
        x = x.cuda().contiguous()

        if self.training:
            y = self.matmul(x)
            y = self.bn(y)
            y = y + self.bias
            y = y / self.divide_value
            y = y * torch.sigmoid(y)
            return y

        M, K = x.shape
        N = self.out_features

        if (self._cached_W_bf16 is None) or (self._cached_W_bf16.device != x.device):
            self._build_cache(x.device)

        x_bf16 = x.to(torch.bfloat16)

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        inv_div = 1.0 / self.divide_value

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

        fused_linear_bn_bias_swish_kernel_bf16[grid](
            x_bf16, self._cached_W_bf16, self._cached_Lbias,
            self._cached_scale, self._cached_shift,
            self.bias,
            out,
            M, N, K,
            inv_div,
            x_bf16.stride(0), x_bf16.stride(1),
            self._cached_W_bf16.stride(0), self._cached_W_bf16.stride(1),
            out.stride(0), out.stride(1),
            GROUP_M=8,
        )
        return out