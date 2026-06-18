import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_swish_kernel(
    A_ptr, W_ptr,
    Scale_ptr, CombBias_ptr,
    Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
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
    w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
            acc += tl.dot(a, w)
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
            acc += tl.dot(a, w)
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk

    # apply fused affine: acc * scale + combined_bias
    # where scale and combined_bias already include INV_DIV folded in
    scale = tl.load(Scale_ptr + offs_n, mask=mask_n, other=0.0)
    cbias = tl.load(CombBias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * scale[None, :] + cbias[None, :]

    # swish: x * sigmoid(x)
    out = acc * tl.sigmoid(acc)

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
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
        # y: [M, N]
        mean = y.mean(dim=0)
        var = y.var(dim=0, unbiased=False)
        with torch.no_grad():
            self.bn.running_mean.mul_(1 - self.bn_momentum).add_(mean.detach(), alpha=self.bn_momentum)
            # unbiased var for running stats
            M = y.shape[0]
            unbiased_var = var.detach() * (M / max(M - 1, 1))
            self.bn.running_var.mul_(1 - self.bn_momentum).add_(unbiased_var, alpha=self.bn_momentum)
        return mean, var

    def forward(self, x):
        x = x.cuda().contiguous()
        M, K = x.shape
        N = self.out_features

        if self.training:
            # need to compute BN stats from actual matmul output
            y = torch.nn.functional.linear(x, self.matmul.weight, self.matmul.bias)
            mean, var = self._compute_bn_stats(y)
            invstd = torch.rsqrt(var + self.bn_eps)
            scale = self.bn.weight * invstd
            shift = self.bn.bias - mean * scale
            # bias broadcast to N
            extra = self.bias.expand(N).contiguous() if self.bias.numel() != N else self.bias
            inv_div = 1.0 / self.divide_value
            out = (y * scale + shift + extra) * inv_div
            out = out * torch.sigmoid(out)
            return out
        else:
            # eval: use running stats, fully fused kernel
            w = self.matmul.weight  # [N, K]
            cache_key = (w._version, self.bn.running_mean._version, self.bn.running_var._version,
                         self.bn.weight._version, self.bn.bias._version, self.bias._version,
                         self.matmul.bias._version, N)
            if (self._eval_cache is None) or (self._eval_cache_key != cache_key) or (self._eval_cache[0].device != w.device):
                inv_div = 1.0 / self.divide_value
                invstd = torch.rsqrt(self.bn.running_var + self.bn_eps)
                scale = self.bn.weight * invstd  # [N]
                shift = self.bn.bias - self.bn.running_mean * scale  # [N]
                if self.bias.numel() == N:
                    extra = self.bias
                else:
                    extra = self.bias.expand(N)
                # full epilogue (pre-swish): ((matmul + blin) * scale + shift + extra) * inv_div
                #                          = matmul * (scale*inv_div) + (blin*scale + shift + extra) * inv_div
                blin = self.matmul.bias
                scale_f = (scale * inv_div).contiguous()
                comb_bias = ((blin * scale + shift + extra) * inv_div).contiguous()
                self._eval_cache = (scale_f, comb_bias)
                self._eval_cache_key = cache_key
            scale_f, comb_bias = self._eval_cache

            out = torch.empty((M, N), device=x.device, dtype=x.dtype)

            even_k = (K % 128 == 0) and (K % 64 == 0)
            # Pass W as [N,K] but tell the kernel it's [K,N] by swapping strides:
            # stride_wk = w.stride(1) (=1), stride_wn = w.stride(0) (=K)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
            fused_linear_bn_swish_kernel[grid](
                x, w,
                scale_f, comb_bias,
                out,
                M, N, K,
                x.stride(0), x.stride(1),
                w.stride(1), w.stride(0),
                out.stride(0), out.stride(1),
                EVEN_K=even_k,
            )
            return out