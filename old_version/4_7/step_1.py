import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoConfig


@triton.jit
def fused_gelu_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # GPT-2 GELU (tanh approximation as used in HF)
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    # tanh via exp
    e2 = tl.exp(2.0 * inner)
    t = (e2 - 1.0) / (e2 + 1.0)
    y = 0.5 * x * (1.0 + t)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_gelu(x):
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_gelu_kernel[grid](x_c, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out.view_as(x)


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    out_row = out_ptr + row * N

    # compute mean and var
    _sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    _sumsq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.where(mask, a, 0.0)
        _sumsq += tl.where(mask, a * a, 0.0)
    mean = tl.sum(_sum, axis=0) / N
    var = tl.sum(_sumsq, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (a - mean) * rstd * w + b
        tl.store(out_row + cols, y, mask=mask)


def triton_layernorm(x, weight, bias, eps):
    x_c = x.contiguous()
    orig_shape = x_c.shape
    N = orig_shape[-1]
    M = x_c.numel() // N
    x_flat = x_c.view(M, N)
    out = torch.empty_like(x_flat)
    BLOCK_SIZE = 1024 if N > 512 else 512
    grid = (M,)
    layernorm_kernel[grid](
        x_flat, weight.contiguous(), bias.contiguous(), out,
        M, N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=4,
    )
    return out.view(orig_shape)


class TritonGELU(nn.Module):
    def forward(self, x):
        return triton_gelu(x)


class TritonLayerNorm(nn.Module):
    def __init__(self, ln):
        super().__init__()
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps
        self.normalized_shape = ln.normalized_shape

    def forward(self, x):
        return triton_layernorm(x, self.weight, self.bias, self.eps)


def replace_modules(model):
    # Replace GELU activations and LayerNorms in the GPT-2 model
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in ("NewGELUActivation", "GELUActivation", "FastGELUActivation", "QuickGELUActivation"):
                setattr(module, child_name, TritonGELU())
            elif isinstance(child, nn.LayerNorm):
                setattr(module, child_name, TritonLayerNorm(child))
    return model


class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self.model = replace_modules(self.model)

    def forward(self, x):
        return self.model(x).logits