import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_swish_kernel(
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

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(a, w, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # offsets for output (real, not modded)
    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m_out < M
    mask_n = offs_n_out < N

    blin = tl.load(Blin_ptr + offs_n_out, mask=mask_n, other=0.0)
    scale = tl.load(Scale_ptr + offs_n_out, mask=mask_n, other=0.0)
    bias_term = tl.load(BiasTerm_ptr + offs_n_out, mask=mask_n, other=0.0)

    acc = acc + blin[None, :]
    acc = acc * scale[None, :] + bias_term[None, :]
    acc = acc * INV_DIV
    out = acc * tl.sigmoid(acc)

    out_ptrs = Out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


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
            w = self.matmul.weight
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

            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            inv_div = 1.0 / self.divide_value

            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
            fused_linear_bn_swish_kernel[grid](
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