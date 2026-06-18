import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bn_gelu_relu_kernel(
    A, B, C,
    scale_ptr, shift_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # BN affine: y = x * scale + shift  (scale and shift folded with weight/bias)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scale = tl.load(scale_ptr + offs_cn, mask=offs_cn < N, other=1.0)
    shift = tl.load(shift_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    y = acc * scale[None, :] + shift[None, :]

    # GELU (erf form) then ReLU
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    out = tl.maximum(gelu, 0.0)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def _get_eval_affine(self):
        # In eval mode use running stats; in training, compute on-the-fly (fallback).
        bn = self.batch_norm
        eps = bn.eps
        rm = bn.running_mean
        rv = bn.running_var
        w = bn.weight
        b = bn.bias
        invstd = torch.rsqrt(rv + eps)
        scale = w * invstd
        shift = b - rm * scale
        return scale, shift

    def forward(self, x):
        x = x.cuda().contiguous()
        if self.training:
            # Fallback to standard path during training
            y = self.gemm(x)
            y = self.batch_norm(y)
            y = torch.nn.functional.gelu(y)
            y = torch.relu(y)
            return y

        M, K = x.shape
        N = self.out_features
        W = self.gemm.weight  # (N, K)
        bias = self.gemm.bias  # (N,)

        scale, shift = self._get_eval_affine()
        # fused_shift = bias * scale + shift
        fused_shift = bias * scale + shift
        fused_scale = scale.contiguous()
        fused_shift = fused_shift.contiguous()

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # B = W.T, shape (K, N)
        Wt = W.t()

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_bn_gelu_relu_kernel[grid](
            x, Wt, out,
            fused_scale, fused_shift,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            out.stride(0), out.stride(1),
        )
        return out