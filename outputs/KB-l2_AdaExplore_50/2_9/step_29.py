import torch
import torch.nn as nn
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "SPLIT_K": 1}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 1}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "SPLIT_K": 1}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 1}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 1}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "SPLIT_K": 1}, num_warps=4, num_stages=4),
    # Split-K variants
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "SPLIT_K": 2}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 2}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "SPLIT_K": 4}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 2}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 2}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def linear_sub_mul_relu_splitk_kernel(
    x_ptr, w_ptr, b_eff_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    MUL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

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
    offs_k = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

    x_ptrs = x_ptr + offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_bn[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    k_iters = tl.cdiv(K, BLOCK_K * SPLIT_K)
    for k in range(0, k_iters):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += BLOCK_K * SPLIT_K * stride_xk
        w_ptrs += BLOCK_K * SPLIT_K * stride_wk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = mask_m[:, None] & mask_n[None, :]

    if SPLIT_K == 1:
        b_eff = tl.load(b_eff_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc * MUL + b_eff[None, :]
        acc = tl.where(acc > 0.0, acc, 0.0)
        tl.store(out_ptrs, acc, mask=mask_out)
    else:
        tl.atomic_add(out_ptrs, acc, mask=mask_out)


@triton.jit
def epilogue_kernel(
    out_ptr, b_eff_ptr,
    M, N,
    stride_om, stride_on,
    MUL: tl.constexpr,
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
    ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    acc = tl.load(ptrs, mask=mask, other=0.0)
    b_eff = tl.load(b_eff_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * MUL + b_eff[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    tl.store(ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous().cuda()
        b = self.linear.bias.contiguous().cuda()

        b_eff = (b - self.subtract_value) * self.multiply_value

        M, K = x.shape
        N = W.shape[0]

        # We'll need to know SPLIT_K at launch to decide whether to zero-init.
        # Use a callable grid; allocate zero-initialized output to be safe for split-K atomics.
        # For SPLIT_K=1, the kernel stores directly (overwriting), so zeros are fine too.
        out = torch.zeros((M, N), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
            meta["SPLIT_K"],
        )

        linear_sub_mul_relu_splitk_kernel[grid](
            x, W, b_eff, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            out.stride(0), out.stride(1),
            MUL=self.multiply_value,
        )

        # If SPLIT_K > 1, we need to apply the epilogue (bias add + relu) after atomic accum.
        best_cfg = linear_sub_mul_relu_splitk_kernel.best_config
        split_k = best_cfg.kwargs.get("SPLIT_K", 1)
        if split_k > 1:
            BLOCK_M = 64
            BLOCK_N = 128
            grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            epilogue_kernel[grid2](
                out, b_eff,
                M, N,
                out.stride(0), out.stride(1),
                MUL=self.multiply_value,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        return out