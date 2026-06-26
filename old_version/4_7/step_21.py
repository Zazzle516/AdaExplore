import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoConfig


@triton.jit
def matmul_add_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    acc = acc + bias[None, :]
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_add(a, b, bias):
    # a: (M, K), b: (K, N), bias: (N,)
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_add_kernel[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=4, num_stages=3,
    )
    return c


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


class TritonConv1D(nn.Module):
    def __init__(self, orig):
        super().__init__()
        # HF Conv1D: forward = x @ weight + bias, weight shape (nx, nf)
        self.weight = orig.weight
        self.bias = orig.bias
        self.nf = orig.nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x2 = x.contiguous().view(-1, x.size(-1))
        w = self.weight
        if not w.is_contiguous():
            w = w.contiguous()
        out = triton_matmul_add(x2, w, self.bias)
        return out.view(size_out)


def replace_modules(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TritonLayerNorm(child))
        elif child.__class__.__name__ == "NewGELUActivation":
            setattr(module, name, TritonGELU())
        elif child.__class__.__name__ == "Conv1D":
            setattr(module, name, TritonConv1D(child))
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