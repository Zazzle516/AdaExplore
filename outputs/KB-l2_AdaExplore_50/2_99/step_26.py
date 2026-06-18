import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_gelu_kernel(
    A_ptr, B_ptr, C_ptr, Bias_ptr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias_ptr + offs_n)
    acc = acc + bias[None, :]

    # GELU (tanh approximation) via sigmoid
    x = acc
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    gelu = 0.5 * x * (1.0 + t)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, gelu, mask=c_mask)


@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr = X_ptr + pid * stride_xm
    y_ptr = Y_ptr + pid * stride_ym

    # Pass 1: compute max and sum via online algorithm
    m = -float('inf')
    s = 0.0
    for start in tl.static_range(0, BLOCK_N, TILE):
        offs = start + tl.arange(0, TILE)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        new_m = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m

    # Pass 2: write normalized output
    inv_s = 1.0 / s
    for start in tl.static_range(0, BLOCK_N, TILE):
        offs = start + tl.arange(0, TILE)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
        out = tl.exp(x - m) * inv_s
        tl.store(y_ptr + offs, out, mask=mask)


def matmul_bias_gelu(x, wT, b):
    M, K = x.shape
    N = wT.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_bias_gelu_kernel[grid](
        x, wT, out, b,
        M, N, K,
        x.stride(0), x.stride(1),
        wT.stride(0), wT.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def softmax_rowwise(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N >= 4096:
        TILE = 2048
        num_warps = 8
    elif BLOCK_N >= 1024:
        TILE = min(BLOCK_N, 1024)
        num_warps = 4
    else:
        TILE = BLOCK_N
        num_warps = 2
    softmax_kernel[(M,)](
        x, out,
        M, N,
        x.stride(0), out.stride(0),
        BLOCK_N=BLOCK_N,
        TILE=TILE,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Pre-transpose weight so the GEMM has stride_bk=1 contiguous along K's inner dim
        wT = self.linear.weight.detach().t().contiguous().cuda()
        self.register_buffer('wT', wT)
        self.register_buffer('bias_buf', self.linear.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        y = matmul_bias_gelu(x, self.wT, self.bias_buf)
        y = softmax_rowwise(y)
        return y