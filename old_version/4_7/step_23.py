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

    # Compute mean and variance
    mean = 0.0
    var = 0.0
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
    # GPT2 gelu_new: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    # tanh via exp
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


def replace_layernorms(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TritonLayerNorm(child))
        else:
            replace_layernorms(child)


def replace_gelu(module):
    # Replace activation_function in MLP blocks with our triton GELU
    for name, child in module.named_children():
        cls_name = type(child).__name__
        if cls_name in ("NewGELUActivation", "GELUActivation", "FastGELUActivation", "QuickGELUActivation"):
            setattr(module, name, TritonGELU())
        else:
            replace_gelu(child)


class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        replace_layernorms(self.model)
        replace_gelu(self.model)

    def forward(self, x):
        return self.model(x).logits