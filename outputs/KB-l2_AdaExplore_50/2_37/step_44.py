import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K', 'CPG'])
@triton.jit
def fused_gemm_swish_bias_gn_kernel(
    X_ptr, W_ptr, Bl_ptr, Bp_ptr, Gamma_ptr, Beta_ptr, Y_ptr,
    M, N, K, G, CPG,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_local = tl.arange(0, BLOCK_N)
    offs_n = pid_g * BLOCK_N + offs_n_local
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None], other=0.0)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # linear bias
    bl = tl.load(Bl_ptr + offs_n)
    acc = acc + bl[None, :]

    # swish
    acc = acc * tl.sigmoid(acc)

    # bias param
    bp = tl.load(Bp_ptr + offs_n)
    acc = acc + bp[None, :]

    # GroupNorm: per-row reduction across BLOCK_N (=CPG)
    inv_cpg = 1.0 / CPG
    mean = tl.sum(acc, axis=1) * inv_cpg  # (BLOCK_M,)
    xc = acc - mean[:, None]
    var = tl.sum(xc * xc, axis=1) * inv_cpg
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(Gamma_ptr + offs_n)
    b = tl.load(Beta_ptr + offs_n)

    y = xc * rstd[:, None] * g[None, :] + b[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask_m[:, None])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.cpg = out_features // num_groups

        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        M, K = x.shape
        N = self.out_features
        G = self.num_groups
        CPG = self.cpg

        W = self.matmul.weight
        bl = self.matmul.bias
        bp = self.bias

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), G)

        fused_gemm_swish_bias_gn_kernel[grid](
            x, W, bl, bp,
            self.group_norm.weight, self.group_norm.bias,
            y,
            M, N, K, G, CPG,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            y.stride(0), y.stride(1),
            float(self.group_norm.eps),
            BLOCK_N=CPG,
        )
        return y