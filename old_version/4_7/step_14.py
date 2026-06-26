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
    # Numerically stable tanh via sigmoid: tanh(z) = 2*sigmoid(2z) - 1
    two_inner = 2.0 * inner
    # clamp to avoid overflow
    two_inner = tl.where(two_inner > 40.0, 40.0, two_inner)
    two_inner = tl.where(two_inner < -40.0, -40.0, two_inner)
    sig = 1.0 / (1.0 + tl.exp(-two_inner))
    th = 2.0 * sig - 1.0
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


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    bias_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remain), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def triton_linear(x, weight, bias):
    # x: (..., K), weight: (N, K), bias: (N,) or None
    orig_shape = x.shape
    K = orig_shape[-1]
    N = weight.shape[0]
    x2d = x.reshape(-1, K).contiguous()
    M = x2d.shape[0]
    w = weight.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        x2d, w, out,
        M, N, K,
        x2d.stride(0), x2d.stride(1),
        w.stride(1), w.stride(0),  # weight is (N,K), we use it as K x N => transpose strides
        out.stride(0), out.stride(1),
        bias is not None,
        bias if bias is not None else x2d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=4, num_stages=3,
    )
    return out.reshape(*orig_shape[:-1], N)


class TritonLinear(nn.Module):
    def __init__(self, orig_linear):
        super().__init__()
        self.weight = orig_linear.weight
        self.bias = orig_linear.bias

    def forward(self, x):
        return triton_linear(x, self.weight, self.bias)


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


class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        replace_layernorms(self.model)
        replace_gelu(self.model)
        # Replace lm_head (heavy Linear op) with custom Triton matmul
        if hasattr(self.model, "lm_head") and isinstance(self.model.lm_head, nn.Linear):
            self.model.lm_head = TritonLinear(self.model.lm_head)

    def forward(self, x):
        return self.model(x).logits