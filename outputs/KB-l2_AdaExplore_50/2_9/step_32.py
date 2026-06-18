import torch
import torch.nn as nn
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def linear_sub_mul_relu_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    SUB_VAL: tl.constexpr,
    MUL_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_MN: tl.constexpr,
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

    if EVEN_MN:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    else:
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    offs_n_store = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_store = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    if EVEN_MN:
        bias = tl.load(bias_ptr + offs_n_store)
        acc = acc + bias[None, :]
        acc = (acc - SUB_VAL) * MUL_VAL
        acc = tl.maximum(acc, 0.0)
        c_ptrs = C_ptr + offs_m_store[:, None] * stride_cm + offs_n_store[None, :] * stride_cn
        tl.store(c_ptrs, acc)
    else:
        mask_n = offs_n_store < N
        mask_m = offs_m_store < M
        bias = tl.load(bias_ptr + offs_n_store, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]
        acc = (acc - SUB_VAL) * MUL_VAL
        acc = tl.maximum(acc, 0.0)
        c_ptrs = C_ptr + offs_m_store[:, None] * stride_cm + offs_n_store[None, :] * stride_cn
        c_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)
        self.in_features = in_features
        self.out_features = out_features
        # Pre-transpose weight so B loads are contiguous along N (stride_bn = 1)
        with torch.no_grad():
            wt = self.linear.weight.t().contiguous()  # shape (K, N)
        self.register_buffer('weight_t', wt)

    def forward(self, x):
        x = x.contiguous()
        if x.device.type != 'cuda':
            x = x.cuda()
        # Ensure weight_t matches the (possibly updated) linear.weight
        weight_t = self.weight_t
        bias = self.linear.bias
        M, K = x.shape
        N = self.out_features

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # weight_t is (K, N) contiguous: stride_bk = N, stride_bn = 1
        stride_bk = weight_t.stride(0)
        stride_bn = weight_t.stride(1)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

        even_mn = (M % 128 == 0) and (N % 128 == 0)
        even_k = (K % 128 == 0)

        linear_sub_mul_relu_kernel[grid](
            x, weight_t, bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            stride_bk, stride_bn,
            out.stride(0), out.stride(1),
            SUB_VAL=self.subtract_value,
            MUL_VAL=self.multiply_value,
            EVEN_MN=even_mn,
            EVEN_K=even_k,
        )
        return out