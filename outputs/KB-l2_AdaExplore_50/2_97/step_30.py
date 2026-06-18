import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_swish_kernel(
    A_ptr, W_ptr, Blin_ptr,
    Scale_ptr, Shift_ptr,
    Bias_ptr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # add linear bias
    blin = tl.load(Blin_ptr + offs_n, mask=mask_n, other=0.0)
    acc += blin[None, :]

    # apply BN affine: scale * x + shift
    scale = tl.load(Scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(Shift_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * scale[None, :] + shift[None, :]

    # add extra bias (scalar broadcast or per-feature)
    extra = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + extra[None, :]

    # divide
    acc = acc * INV_DIV

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
        self._wt_cache = None
        self._wt_version = -1

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
            invstd = torch.rsqrt(self.bn.running_var + self.bn_eps)
            scale = (self.bn.weight * invstd).contiguous()
            shift = (self.bn.bias - self.bn.running_mean * scale).contiguous()
            if self.bias.numel() == N:
                extra = self.bias.contiguous()
            else:
                extra = self.bias.expand(N).contiguous()

            W = self.matmul.weight  # [N, K]
            if (self._wt_cache is None) or (self._wt_version != W._version) or (self._wt_cache.device != W.device):
                self._wt_cache = W.t().contiguous().to(W.device)
                self._wt_version = W._version
            W_t = self._wt_cache  # [K, N] contiguous
            blin = self.matmul.bias.contiguous()  # [N]

            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            inv_div = 1.0 / self.divide_value

            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
            fused_linear_bn_swish_kernel[grid](
                x, W_t, blin,
                scale, shift,
                extra,
                out,
                M, N, K,
                inv_div,
                x.stride(0), x.stride(1),
                W_t.stride(0), W_t.stride(1),
                out.stride(0), out.stride(1),
            )
            return out