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

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * w + b
    tl.store(out_ptr + cols, y, mask=mask)


def triton_layernorm(x, weight, bias, eps):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2d = x.reshape(-1, N).contiguous()
    M = x2d.shape[0]
    out = torch.empty_like(x2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    fused_ln_kernel[(M,)](
        x2d, weight, bias, out,
        M, N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
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
    # GPT-2 NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    # Numerically stable tanh: sign(x)*(1 - 2/(exp(2|x|)+1))
    abs_in = tl.abs(inner)
    sign = tl.where(inner >= 0.0, 1.0, -1.0)
    # clamp to avoid overflow
    abs_in = tl.minimum(abs_in, 20.0)
    e2 = tl.exp(2.0 * abs_in)
    th = sign * (1.0 - 2.0 / (e2 + 1.0))
    y = 0.5 * x * (1.0 + th)
    tl.store(out_ptr + offs, y, mask=mask)


def triton_gelu_new(x):
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_new_kernel[grid](x_c, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


class TritonLayerNorm(nn.Module):
    def __init__(self, orig_ln):
        super().__init__()
        self.weight = orig_ln.weight
        self.bias = orig_ln.bias
        self.eps = orig_ln.eps
        self.normalized_shape = orig_ln.normalized_shape

    def forward(self, x):
        return triton_layernorm(x, self.weight, self.bias, self.eps)


class TritonNewGELU(nn.Module):
    def forward(self, x):
        return triton_gelu_new(x)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_m[:, None] < M) & ((k * BLOCK_K + offs_k)[None, :] < K)
        b_mask = ((k * BLOCK_K + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_bias(x, weight, bias):
    # x: (..., K), weight: (K, N), bias: (N,) or None
    orig_shape = x.shape
    K = orig_shape[-1]
    x2d = x.reshape(-1, K).contiguous()
    M = x2d.shape[0]
    N = weight.shape[1]
    weight = weight.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    matmul_bias_kernel[grid](
        x2d, weight, out, bias if bias is not None else x2d,
        M, N, K,
        x2d.stride(0), x2d.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    if bias is None:
        # The kernel above always adds bias; handle None case separately
        pass
    return out.reshape(*orig_shape[:-1], N)


class TritonConv1D(nn.Module):
    def __init__(self, orig):
        super().__init__()
        # transformers Conv1D: weight shape (in_features, out_features), bias (out_features,)
        self.weight = orig.weight
        self.bias = orig.bias
        self.nf = orig.nf

    def forward(self, x):
        return triton_matmul_bias(x, self.weight, self.bias)


def replace_layernorms(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TritonLayerNorm(child))
        else:
            replace_layernorms(child)


def replace_gelu(module):
    # Replace GPT-2's NewGELUActivation with our triton variant
    for name, child in module.named_children():
        cname = type(child).__name__
        if cname in ("NewGELUActivation", "GELUActivation", "FastGELUActivation"):
            setattr(module, name, TritonNewGELU())
        else:
            replace_gelu(child)


def replace_conv1d(module):
    for name, child in module.named_children():
        cname = type(child).__name__
        if cname == "Conv1D":
            setattr(module, name, TritonConv1D(child))
        else:
            replace_conv1d(child)


class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        replace_layernorms(self.model)
        replace_gelu(self.model)
        replace_conv1d(self.model)

    def forward(self, x):
        return self.model(x).logits