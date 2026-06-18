import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------
# Fused GEMM + bias + scale -> y, with column-major layout output
# y has shape (N, M) but we store it as (M, N) (transposed) so that
# BN reduction along N is contiguous.
# ---------------------------------------------------------------

GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_yn, stride_ym,  # Note: Y is (M, N) layout (transposed). stride_yn first.
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias (per output column n) and scale (per output column n)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    # Store transposed: y has shape (N, M). Output index is (n, m).
    # Y_ptr indexed with stride_yn for n axis, stride_ym for m axis.
    # We have (BLOCK_M, BLOCK_N) = (m, n). Transpose to (n, m).
    y_ptrs = Y_ptr + offs_n[:, None] * stride_yn + offs_m[None, :] * stride_ym
    y_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(y_ptrs, tl.trans(acc), mask=y_mask)


# ---------------------------------------------------------------
# BN kernel: input is (N, M) col-of-features layout (each row is one channel,
# all M batch values contiguous). One program per channel.
# Two-pass Welford-style: pass 1 mean, pass 2 variance, then normalize+affine,
# stores back to (M, N) row-major output.
# ---------------------------------------------------------------

@triton.jit
def bn_kernel(
    Yt_ptr,        # (N, M) input (transposed gemm output)
    Out_ptr,       # (M, N) output (final result, row-major as original)
    gamma_ptr, beta_ptr,
    M, N,
    eps,
    BLOCK_M: tl.constexpr,
):
    n = tl.program_id(0)  # channel index

    row_ptr = Yt_ptr + n * M

    # Pass 1: mean
    sum_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < M
        x = tl.load(row_ptr + offs, mask=mask, other=0.0)
        sum_acc += x
    mean = tl.sum(sum_acc, axis=0) / M

    # Pass 2: variance
    var_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < M
        x = tl.load(row_ptr + offs, mask=mask, other=0.0)
        d = tl.where(mask, x - mean, 0.0)
        var_acc += d * d
    var = tl.sum(var_acc, axis=0) / M

    rstd = 1.0 / tl.sqrt(var + eps)
    gamma = tl.load(gamma_ptr + n)
    beta = tl.load(beta_ptr + n)

    scale = gamma * rstd
    shift = beta - mean * scale

    # Pass 3: normalize and write transposed -> (M, N)
    for m_start in range(0, M, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < M
        x = tl.load(row_ptr + offs, mask=mask, other=0.0)
        y = x * scale + shift
        # Output (M, N), row-major, so index = m*N + n
        out_offs = offs * N + n
        tl.store(Out_ptr + out_offs, y, mask=mask)


# ---------------------------------------------------------------
# Eval-mode BN: use running mean/var. Fold into scale/shift on Yt layout
# and write to (M, N) output.
# ---------------------------------------------------------------

@triton.jit
def bn_eval_kernel(
    Yt_ptr, Out_ptr,
    scale_ptr, shift_ptr,  # precomputed per channel
    M, N,
    BLOCK_M: tl.constexpr,
):
    n = tl.program_id(0)
    row_ptr = Yt_ptr + n * M
    scale = tl.load(scale_ptr + n)
    shift = tl.load(shift_ptr + n)
    for m_start in range(0, M, BLOCK_M):
        offs = m_start + tl.arange(0, BLOCK_M)
        mask = offs < M
        x = tl.load(row_ptr + offs, mask=mask, other=0.0)
        y = x * scale + shift
        out_offs = offs * N + n
        tl.store(Out_ptr + out_offs, y, mask=mask)


# ---------------------------------------------------------------
# Update running stats (training mode), CPU-side helper using the
# already-computed batch mean/var. We compute them on GPU in a separate
# small kernel since BN kernel doesn't return them.
# Simplest: recompute mean/var via torch for running stats update only.
# ---------------------------------------------------------------


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        x = x.contiguous().cuda()
        M = x.shape[0]
        K = x.shape[1]
        N = self.out_features

        W = self.gemm.weight  # (N, K)
        bias = self.gemm.bias  # (N,)
        scale = self.scale  # (N,)

        # B = W.T -> (K, N); we need to make it accessible with strides.
        Wt = W.t().contiguous()  # (K, N)

        # Output of GEMM stored as (N, M) (transposed for BN locality)
        Yt = torch.empty((N, M), device=x.device, dtype=torch.float32)

        grid_gemm = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

        gemm_scale_kernel[grid_gemm](
            x, Wt, bias, scale, Yt,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            Yt.stride(0), Yt.stride(1),
        )

        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        BLOCK_M = 1024
        if M <= 256:
            BLOCK_M = 256
        elif M <= 512:
            BLOCK_M = 512
        elif M <= 1024:
            BLOCK_M = 1024
        else:
            BLOCK_M = 1024

        if self.training:
            # Need batch mean/var to update running stats.
            # Compute on GPU using torch on Yt (rows are channels).
            with torch.no_grad():
                batch_mean = Yt.mean(dim=1)
                batch_var = Yt.var(dim=1, unbiased=False)
                self.bn.running_mean.mul_(1 - self.momentum).add_(batch_mean, alpha=self.momentum)
                # PyTorch uses unbiased variance for running_var
                unbiased_var = Yt.var(dim=1, unbiased=True)
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)
                self.bn.num_batches_tracked.add_(1)

            bn_kernel[(N,)](
                Yt, out,
                self.bn.weight, self.bn.bias,
                M, N,
                self.eps,
                BLOCK_M=BLOCK_M,
                num_warps=4,
            )
        else:
            rstd = 1.0 / torch.sqrt(self.bn.running_var + self.eps)
            scale_eff = self.bn.weight * rstd
            shift_eff = self.bn.bias - self.bn.running_mean * scale_eff
            bn_eval_kernel[(N,)](
                Yt, out,
                scale_eff.contiguous(), shift_eff.contiguous(),
                M, N,
                BLOCK_M=BLOCK_M,
                num_warps=4,
            )

        return out