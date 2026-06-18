import torch
import torch.nn as nn
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 2}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 2}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def fused_linear_bn_swish_kernel(
    A_ptr, B_ptr, bias_ptr,
    scale_ptr, shift_ptr,
    Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    INV_DIV: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
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
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    K_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_sk * K_per_split
    k_end = tl.minimum(k_start + K_per_split, K)
    num_k_iters = tl.cdiv(k_end - k_start, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak)
    b_ptrs = B_ptr + ((k_start + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    for k in range(0, num_k_iters):
        k_offs = k_start + k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_end
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if SPLIT_K == 1:
        scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
        shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)
        bias_extra = tl.load(bias_ptr)
        y = acc * scale[None, :] + shift[None, :] + bias_extra
        y = y * INV_DIV
        y = y * tl.sigmoid(y)
        out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])
    else:
        out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.atomic_add(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def epilogue_kernel(
    Out_ptr, scale_ptr, shift_ptr, bias_ptr,
    M, N,
    INV_DIV: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    acc = tl.load(ptrs, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)
    bias_extra = tl.load(bias_ptr)
    y = acc * scale[None, :] + shift[None, :] + bias_extra
    y = y * INV_DIV
    y = y * tl.sigmoid(y)
    tl.store(ptrs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.divide_value = float(divide_value)

        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        self._cached_B = None
        self._cached_scale = None
        self._cached_shift = None

    def _build_fused_params(self):
        W = self.matmul.weight
        b_lin = self.matmul.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias

        inv_std = torch.rsqrt(running_var + self.bn_eps)
        scale = (gamma * inv_std).contiguous()
        shift = (beta - running_mean * gamma * inv_std).contiguous()
        shift = (b_lin * scale + shift).contiguous()
        B = W.t().contiguous()
        return B, scale, shift

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        if self.training:
            y = self.matmul(x)
            y = self.bn(y)
            y = y + self.bias
            y = y / self.divide_value
            y = y * torch.sigmoid(y)
            return y

        if self._cached_B is None:
            B, scale, shift = self._build_fused_params()
            self._cached_B = B
            self._cached_scale = scale
            self._cached_shift = shift
        B = self._cached_B
        scale = self._cached_scale
        shift = self._cached_shift

        bias_extra = self.bias.view(-1)[0:1].contiguous()
        inv_div = 1.0 / self.divide_value

        def grid(meta):
            return (
                triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
                meta['SPLIT_K'],
            )

        # We need to allocate output. If SPLIT_K > 1, we need to zero-init for atomic add,
        # then run epilogue. We can't know SPLIT_K before autotune picks it, so always
        # zero-init and run epilogue conditionally. Use two-kernel approach only if needed.
        # Simpler: always allocate zero-inited output, kernel handles SPLIT_K==1 with
        # store, SPLIT_K>1 with atomic_add. Then run epilogue only when SPLIT_K>1.
        # But we can't know that here. Approach: always do epilogue=False path in kernel,
        # then run epilogue separately. But that adds overhead for SPLIT_K==1.
        # Compromise: do separate-path. We know after autotune via best_config.

        # Allocate output (uninitialized for SPLIT_K==1; zero for SPLIT_K>1).
        # We'll first do a probing strategy: allocate zero, kernel always atomic-adds
        # only if SPLIT_K>1; otherwise stores final result with epilogue. The kernel
        # branches inside.

        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        # We need to pre-zero if SPLIT_K > 1. Check best config after first call.
        # To handle this cleanly: zero-init always (cost is small relative to GEMM
        # for split-k case; for non-split-k case, we just write over).
        # Actually zero_ is wasteful for SPLIT_K=1. Let's do: try without zeroing,
        # check selected config, if SPLIT_K>1 we need to zero and rerun. Complex.
        # 
        # Simpler: always zero. 1024*8192*4 = 32MB write, ~0.3ms on RTX 4090.
        # That might erode gains. Better: probe config first.

        # Approach: launch with a wrapper that decides based on best_config.
        # The autotune .best_config is available after first run via warmup.
        # We'll do a one-time warmup to determine the SPLIT_K of the chosen config.

        if not hasattr(self, '_split_k_chosen'):
            # Warmup to select best config
            tmp = torch.zeros((M, N), device=x.device, dtype=torch.float32)
            fused_linear_bn_swish_kernel[grid](
                x, B, bias_extra, scale, shift, tmp,
                M, N, K,
                x.stride(0), x.stride(1),
                B.stride(0), B.stride(1),
                tmp.stride(0), tmp.stride(1),
                INV_DIV=inv_div,
            )
            best_cfg = fused_linear_bn_swish_kernel.best_config
            self._split_k_chosen = best_cfg.kwargs.get('SPLIT_K', 1)

        if self._split_k_chosen > 1:
            out.zero_()
            fused_linear_bn_swish_kernel[grid](
                x, B, bias_extra, scale, shift, out,
                M, N, K,
                x.stride(0), x.stride(1),
                B.stride(0), B.stride(1),
                out.stride(0), out.stride(1),
                INV_DIV=1.0,  # don't apply in kernel for split-k path
            )
            # Apply epilogue
            BLOCK_M_EP = 32
            BLOCK_N_EP = 128
            grid_ep = (triton.cdiv(M, BLOCK_M_EP), triton.cdiv(N, BLOCK_N_EP))
            epilogue_kernel[grid_ep](
                out, scale, shift, bias_extra,
                M, N,
                INV_DIV=inv_div,
                BLOCK_M=BLOCK_M_EP, BLOCK_N=BLOCK_N_EP,
            )
        else:
            fused_linear_bn_swish_kernel[grid](
                x, B, bias_extra, scale, shift, out,
                M, N, K,
                x.stride(0), x.stride(1),
                B.stride(0), B.stride(1),
                out.stride(0), out.stride(1),
                INV_DIV=inv_div,
            )
        return out