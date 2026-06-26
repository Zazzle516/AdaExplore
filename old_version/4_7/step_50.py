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
    # GPT-2 uses tanh approximation gelu_new
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    # tanh via exp
    e2 = tl.exp(2.0 * inner)
    t = (e2 - 1.0) / (e2 + 1.0)
    y = 0.5 * x * (1.0 + t)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_gelu_kernel[grid](x_c, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


class TritonGELU(nn.Module):
    def forward(self, x):
        return triton_gelu(x)


@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride_x, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride_x
    out_ptr += row * stride_x
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


def triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    out = torch.empty_like(x2)
    M = x2.shape[0]
    # next power of 2 for BLOCK_SIZE
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    layer_norm_kernel[(M,)](
        x2, weight, bias, out,
        x2.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return out.view(orig_shape)


class TritonLayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps
        self.normalized_shape = ln.normalized_shape

    def forward(self, x):
        return triton_layer_norm(x, self.weight, self.bias, self.eps)


def replace_modules(model):
    # Replace GELU activations and LayerNorms in GPT-2 blocks
    for name, module in model.named_modules():
        # GPT-2 MLP uses NewGELUActivation
        for child_name, child in list(module.named_children()):
            cls_name = child.__class__.__name__
            if cls_name in ("NewGELUActivation", "GELUActivation", "FastGELUActivation"):
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