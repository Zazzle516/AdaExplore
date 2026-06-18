import torch
import torch.nn as nn
import triton
import triton.language as tl

AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_gemm_bn_gelu_relu_kernel(
    A_ptr, B_ptr, scale_ptr, shift_ptr, C_ptr,
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply scale & shift (BN folded with bias) per output column
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_cn < N
    scale = tl.load(scale_ptr + offs_cn, mask=n_mask, other=0.0)
    shift = tl.load(shift_ptr + offs_cn, mask=n_mask, other=0.0)

    y = acc * scale[None, :] + shift[None, :]

    # GELU (erf form): 0.5 * y * (1 + erf(y / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    # ReLU after GELU
    out = tl.maximum(gelu, 0.0)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def _get_fold(self):
        # Use running stats for BN at eval, but training also has running stats.
        # Reference uses BN in training mode by default. We must NOT change semantics.
        # However, BN at training time computes from batch stats. To stay correct,
        # we'll only fold at eval. At training time, fall back to functional BN.
        return None

    def forward(self, x):
        x = x.contiguous().cuda()
        if not self.training:
            # Fold BN into linear: y = (Wx + b - mean)/sqrt(var+eps) * gamma + beta
            # = (gamma/sqrt(var+eps)) * (Wx + b) + (beta - mean*gamma/sqrt(var+eps))
            bn = self.batch_norm
            with torch.no_grad():
                inv_std = torch.rsqrt(bn.running_var + bn.eps)
                scale = bn.weight * inv_std  # [N]
                shift = bn.bias - bn.running_mean * scale  # [N]
                # combine with linear bias
                bias_eff = self.gemm.bias * scale + shift
                scale_full = scale
                shift_full = bias_eff

            M = x.shape[0]
            K = x.shape[1]
            N = self.out_features
            W = self.gemm.weight  # [N, K]
            B_t = getattr(self, '_B_t_cache', None)
            if B_t is None or B_t.device != W.device or B_t.dtype != W.dtype:
                B_t = W.t().contiguous()
                self._B_t_cache = B_t
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)

            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
            fused_gemm_bn_gelu_relu_kernel[grid](
                x, B_t, scale_full.contiguous(), shift_full.contiguous(), out,
                M, N, K,
                x.stride(0), x.stride(1),
                B_t.stride(0), B_t.stride(1),
                out.stride(0), out.stride(1),
            )
            return out
        else:
            # Training: use standard ops to preserve BN stats updates
            x = self.gemm(x)
            x = self.batch_norm(x)
            x = torch.nn.functional.gelu(x)
            x = torch.relu(x)
            return x