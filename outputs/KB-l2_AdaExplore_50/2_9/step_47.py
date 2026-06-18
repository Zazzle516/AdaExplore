import torch
import torch.nn as nn
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K', 'SPLIT_K'])
@triton.jit
def fused_linear_splitk_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    SUB: tl.constexpr,
    MUL: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
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
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_step = BLOCK_K * SPLIT_K
    for k in range(pid_k * BLOCK_K, K, k_step):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w)
        x_ptrs += k_step * stride_xk
        w_ptrs += k_step * stride_wk

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m_out < M
    mask_n = offs_n_out < N

    if SPLIT_K == 1:
        b = tl.load(b_ptr + offs_n_out, mask=mask_n, other=0.0)
        acc += b[None, :]
        acc = (acc - SUB) * MUL
        acc = tl.maximum(acc, 0.0)
        out_ptrs = out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])
    else:
        out_ptrs = out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
        tl.atomic_add(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def epilogue_kernel(
    acc_ptr, b_ptr, out_ptr,
    M, N,
    SUB: tl.constexpr,
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
    ptrs = acc_ptr + offs_m[:, None] * N + offs_n[None, :]
    v = tl.load(ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    v = v + b[None, :]
    v = (v - SUB) * MUL
    v = tl.maximum(v, 0.0)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, v, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)
        self.in_features = in_features
        self.out_features = out_features

        # Pre-transpose weight to (K, N) layout for contiguous B-operand loads.
        with torch.no_grad():
            wt = self.linear.weight.detach().t().contiguous()  # (K, N)
        self.register_buffer("weight_t", wt.cuda(), persistent=False)
        self.register_buffer("bias_buf", self.linear.bias.detach().contiguous().cuda(), persistent=False)

        self.SPLIT_K = 1

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        M, K = x.shape
        N = self.out_features
        w = self.weight_t
        if not w.is_cuda:
            w = w.cuda()
            self.weight_t = w
        b = self.bias_buf
        if not b.is_cuda:
            b = b.cuda()
            self.bias_buf = b

        SPLIT_K = self.SPLIT_K

        if SPLIT_K == 1:
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
                1,
            )
            fused_linear_splitk_kernel[grid](
                x, w, b, out,
                M, N, K,
                x.stride(0), x.stride(1),
                w.stride(0), w.stride(1),
                out.stride(0), out.stride(1),
                SUB=self.subtract_value,
                MUL=self.multiply_value,
                SPLIT_K=1,
            )
            return out
        else:
            acc = torch.zeros((M, N), device=x.device, dtype=torch.float32)
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
                SPLIT_K,
            )
            fused_linear_splitk_kernel[grid](
                x, w, b, acc,
                M, N, K,
                x.stride(0), x.stride(1),
                w.stride(0), w.stride(1),
                acc.stride(0), acc.stride(1),
                SUB=self.subtract_value,
                MUL=self.multiply_value,
                SPLIT_K=SPLIT_K,
            )
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            BLOCK_M = 64
            BLOCK_N = 128
            grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            epilogue_kernel[grid2](
                acc, b, out,
                M, N,
                SUB=self.subtract_value,
                MUL=self.multiply_value,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            return out