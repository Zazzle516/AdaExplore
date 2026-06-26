import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoConfig


@triton.jit
def fused_ln_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * N
    out_ptr += row * N

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sumsq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x
        _sumsq += x * x
    mean = tl.sum(_sum, axis=0) / N
    mean_sq = tl.sum(_sumsq, axis=0) / N
    var = mean_sq - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y, mask=mask)


def fused_layernorm(x, weight, bias, eps):
    orig_shape = x.shape
    N = orig_shape[-1]
    x_flat = x.reshape(-1, N).contiguous()
    M = x_flat.shape[0]
    out = torch.empty_like(x_flat)
    BLOCK_SIZE = 1024 if N > 1024 else triton.next_power_of_2(N)
    fused_ln_kernel[(M,)](x_flat, weight, bias, out, M, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(orig_shape)


@triton.jit
def gelu_new_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
    e2x = tl.exp(2.0 * inner)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offs, y, mask=mask)


def fused_gelu_new(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_new_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# Triton matmul: C = A @ B + bias
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
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
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        offs_bias = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias = tl.load(bias_ptr + offs_bias, mask=offs_bias < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def triton_matmul(a, b, bias=None):
    # a: (M, K), b: (K, N) -> c: (M, N)
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    has_bias = bias is not None
    bias_ptr = bias if has_bias else a  # dummy
    matmul_bias_kernel[grid](
        a, b, c, bias_ptr,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=has_bias,
    )
    return c


class TritonLayerNorm(nn.Module):
    def __init__(self, ln):
        super().__init__()
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps

    def forward(self, x):
        return fused_layernorm(x, self.weight, self.bias, self.eps)


class TritonGELU(nn.Module):
    def forward(self, x):
        return fused_gelu_new(x)


class TritonConv1D(nn.Module):
    """Replacement for transformers.pytorch_utils.Conv1D.
    Conv1D does: x @ weight + bias where weight is (in_features, out_features)."""
    def __init__(self, conv1d):
        super().__init__()
        self.weight = conv1d.weight  # (nx, nf)
        self.bias = conv1d.bias      # (nf,)
        self.nf = conv1d.nf if hasattr(conv1d, 'nf') else conv1d.weight.shape[-1]

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1])
        out = triton_matmul(x2, self.weight, self.bias)
        return out.reshape(*orig_shape[:-1], self.nf)


class TritonLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.weight = lin.weight  # (out, in)
        self.bias = lin.bias
        self.out_features = lin.out_features

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1])
        # weight is (out, in); for matmul we need (in, out)
        w_t = self.weight.t().contiguous() if not hasattr(self, '_wt') else self._wt
        out = triton_matmul(x2, w_t, self.bias)
        return out.reshape(*orig_shape[:-1], self.out_features)


def replace_modules(module):
    for name, child in module.named_children():
        cls_name = type(child).__name__
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TritonLayerNorm(child))
        elif cls_name in ("NewGELUActivation", "GELUActivation", "FastGELUActivation", "QuickGELUActivation"):
            setattr(module, name, TritonGELU())
        elif cls_name == "Conv1D":
            setattr(module, name, TritonConv1D(child))
        elif isinstance(child, nn.Linear):
            setattr(module, name, TritonLinear(child))
        else:
            replace_modules(child)


class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        replace_modules(self.model)

    def forward(self, x):
        return self.model(x).logits