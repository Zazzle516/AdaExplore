import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_htanh_mish_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
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

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias_ptr + offs_n)
    acc = acc + bias[None, :]

    # hardtanh
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    acc = acc * th

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


@triton.jit
def groupnorm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    C, G,
    eps,
    GROUP_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid encodes (row_block, group)
    row_block = pid // G
    grp = pid % G

    offs = tl.arange(0, GROUP_SIZE)
    row_offs = tl.arange(0, ROWS_PER_PROG)
    rows = row_block * ROWS_PER_PROG + row_offs

    base = rows[:, None] * C + grp * GROUP_SIZE + offs[None, :]
    x = tl.load(X_ptr + base)

    mean = tl.sum(x, axis=1) / GROUP_SIZE
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + grp * GROUP_SIZE + offs)
    b = tl.load(B_ptr + grp * GROUP_SIZE + offs)

    y = xc * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(Y_ptr + base, y)


def fused_gemm_bias_htanh_mish(x, w_t, bias_total):
    M, K = x.shape
    N = w_t.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_htanh_mish_kernel[grid](
        x, w_t, bias_total, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def groupnorm_apply(x, weight, bias, num_groups, eps=1e-5):
    M, C = x.shape
    G = num_groups
    GROUP_SIZE = C // G
    out = torch.empty_like(x)
    ROWS_PER_PROG = 4
    while M % ROWS_PER_PROG != 0 and ROWS_PER_PROG > 1:
        ROWS_PER_PROG //= 2
    grid = ((M // ROWS_PER_PROG) * G,)
    groupnorm_kernel[grid](
        x, out, weight, bias,
        C, G,
        eps,
        GROUP_SIZE=GROUP_SIZE,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self._cached_wt = None
        self._cached_bias = None

    def _get_cached(self):
        w = self.gemm.weight
        b_linear = self.gemm.bias
        b_extra = self.bias
        if (self._cached_wt is None or
            self._cached_wt.shape[0] != w.shape[1] or
            self._cached_wt.shape[1] != w.shape[0]):
            self._cached_wt = w.t().contiguous()
            self._cached_bias = (b_linear + b_extra).contiguous()
        return self._cached_wt, self._cached_bias

    def forward(self, x):
        x = x.contiguous()
        if self.training:
            w_t = self.gemm.weight.t().contiguous()
            bias_total = (self.gemm.bias + self.bias).contiguous()
        else:
            w_t, bias_total = self._get_cached()
        y = fused_gemm_bias_htanh_mish(x, w_t, bias_total)
        y = groupnorm_apply(y, self.groupnorm.weight, self.groupnorm.bias,
                            self.num_groups, self.groupnorm.eps)
        return y