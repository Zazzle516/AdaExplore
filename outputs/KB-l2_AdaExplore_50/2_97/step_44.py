import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_gemm_splitk_kernel(
    A_ptr, W_ptr, Blin_ptr,
    Scale_ptr, BiasTerm_ptr,
    Out_ptr,
    M, N, K,
    INV_DIV: tl.constexpr,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)
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
    offs_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    K_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_sk * K_per_split
    k_end = tl.minimum(k_start + K_per_split, K)
    num_iters = tl.cdiv(k_end - k_start, BLOCK_K)

    for k in range(0, num_iters):
        k_offset = k_start + k * BLOCK_K
        k_remaining = K - k_offset
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_end), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_end) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    if SPLIT_K == 1:
        # apply full epilogue
        blin = tl.load(Blin_ptr + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)), mask=mask_n, other=0.0)
        acc += blin[None, :]
        scale = tl.load(Scale_ptr + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)), mask=mask_n, other=0.0)
        bias_term = tl.load(BiasTerm_ptr + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)), mask=mask_n, other=0.0)
        acc = acc * scale[None, :] + bias_term[None, :]
        acc = acc * INV_DIV
        out = acc * tl.sigmoid(acc)
        out_ptrs = Out_ptr + offs_m[:, None] * stride_om + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_on
        tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])
    else:
        out_ptrs = Out_ptr + offs_m[:, None] * stride_om + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_on
        tl.atomic_add(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def epilogue_kernel(
    Y_ptr, Blin_ptr, Scale_ptr, BiasTerm_ptr,
    M, N,
    INV_DIV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    y = tl.load(ptrs, mask=mask, other=0.0)
    blin = tl.load(Blin_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(Scale_ptr + offs_n, mask=mask_n, other=0.0)
    bias_term = tl.load(BiasTerm_ptr + offs_n, mask=mask_n, other=0.0)
    y = y + blin[None, :]
    y = y * scale[None, :] + bias_term[None, :]
    y = y * INV_DIV
    out = y * tl.sigmoid(y)
    tl.store(ptrs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.divide_value = float(divide_value)

        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._eval_cache = None
        self._eval_cache_key = None

    def _compute_bn_stats(self, y):
        mean = y.mean(dim=0)
        var = y.var(dim=0, unbiased=False)
        with torch.no_grad():
            self.bn.running_mean.mul_(1 - self.bn_momentum).add_(mean.detach(), alpha=self.bn_momentum)
            M = y.shape[0]
            unbiased_var = var.detach() * (M / max(M - 1, 1))
            self.bn.running_var.mul_(1 - self.bn_momentum).add_(unbiased_var, alpha=self.bn_momentum)
        return mean, var

    def forward(self, x):
        x = x.cuda().contiguous()
        M, K = x.shape
        N = self.out_features

        if self.training:
            y = torch.nn.functional.linear(x, self.matmul.weight, self.matmul.bias)
            mean, var = self._compute_bn_stats(y)
            invstd = torch.rsqrt(var + self.bn_eps)
            scale = self.bn.weight * invstd
            shift = self.bn.bias - mean * scale
            extra = self.bias.expand(N).contiguous() if self.bias.numel() != N else self.bias
            inv_div = 1.0 / self.divide_value
            out = (y * scale + shift + extra) * inv_div
            out = out * torch.sigmoid(out)
            return out
        else:
            w = self.matmul.weight  # [N, K]
            cache_key = (w._version, self.bn.running_mean._version, self.bn.running_var._version,
                         self.bn.weight._version, self.bn.bias._version, self.bias._version, N)
            if (self._eval_cache is None) or (self._eval_cache_key != cache_key) or (self._eval_cache[0].device != w.device):
                invstd = torch.rsqrt(self.bn.running_var + self.bn_eps)
                scale = (self.bn.weight * invstd).contiguous()
                shift = self.bn.bias - self.bn.running_mean * scale
                if self.bias.numel() == N:
                    extra = self.bias
                else:
                    extra = self.bias.expand(N)
                bias_term = (shift + extra).contiguous()
                self._eval_cache = (scale, bias_term)
                self._eval_cache_key = cache_key
            scale, bias_term = self._eval_cache

            blin = self.matmul.bias.contiguous()
            inv_div = 1.0 / self.divide_value

            # First, run GEMM with possibly split-K. For SPLIT_K>1 we accumulate into zero-init output.
            # We'll run two attempts via autotuner; need output appropriately initialized.
            # Strategy: zero-initialize output, kernel does atomic_add when SPLIT_K>1 and writes full epilogue when SPLIT_K=1.
            # To handle SPLIT_K>1, we need a separate epilogue. So we do it in two phases conditioned on which config wins.
            # Simpler: always zero-init, and use a separate epilogue kernel that handles bias/scale/swish, while GEMM kernel
            # writes raw matmul accumulator (atomic_add for split-K, plain store for SPLIT_K=1).
            # To keep things uniform, we'll always zero-init and always atomic-add, then run epilogue.
            # But atomic_add path for SPLIT_K=1 hurts. Use the existing kernel that fuses epilogue when SPLIT_K=1.

            out = torch.empty((M, N), device=x.device, dtype=x.dtype)

            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
                META['SPLIT_K'],
            )
            fused_gemm_splitk_kernel[grid](
                x, w, blin,
                scale, bias_term,
                out,
                M, N, K,
                inv_div,
                x.stride(0), x.stride(1),
                w.stride(1), w.stride(0),
                out.stride(0), out.stride(1),
            )
            return out