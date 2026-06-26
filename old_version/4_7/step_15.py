import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoConfig


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * N
    out_ptr += row * N

    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xm * rstd * w + b
    tl.store(out_ptr + offs, y, mask=mask)


def triton_layernorm(x, weight, bias, eps):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    M = x2.shape[0]
    out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(N)
    layernorm_kernel[(M,)](x2, weight, bias, out, N, eps, BLOCK=BLOCK, num_warps=4)
    return out.view(orig_shape)


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    # tanh via sigmoid: tanh(z) = 2*sigmoid(2z) - 1
    t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    y = 0.5 * x * (1.0 + t)
    tl.store(out_ptr + offs, y, mask=mask)


def triton_gelu(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    gelu_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4)
    return out


class TritonLayerNorm(nn.Module):
    def __init__(self, orig_ln: nn.LayerNorm):
        super().__init__()
        self.weight = orig_ln.weight
        self.bias = orig_ln.bias
        self.eps = orig_ln.eps
        self.normalized_shape = orig_ln.normalized_shape

    def forward(self, x):
        return triton_layernorm(x, self.weight, self.bias, self.eps)


class TritonGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return triton_gelu(x)


def replace_modules(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TritonLayerNorm(child))
        elif child.__class__.__name__ == "NewGELUActivation":
            setattr(module, name, TritonGELU())
        else:
            replace_modules(child)


class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        replace_modules(self.model)

    def forward(self, x):
        return self.model(x).logits