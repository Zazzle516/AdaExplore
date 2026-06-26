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
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
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
def matmul_kernel(
    A, B, C, Bias,
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

    offs_am_raw = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn_raw = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_am = offs_am_raw % M
    offs_bn = offs_bn_raw % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        offs_bias = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_vals = tl.load(Bias + offs_bias, mask=offs_bias < N, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


def triton_matmul(a, b, bias=None):
    # a: (M, K), b: (K, N), bias: (N,) or None
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        a, b, c, bias if bias is not None else a,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=8, num_stages=4,
    )
    return c


class TritonConv1D(nn.Module):
    """Replacement for transformers.pytorch_utils.Conv1D.
    Computes: x @ weight + bias, weight shape (nx, nf), bias shape (nf,).
    """
    def __init__(self, conv1d):
        super().__init__()
        self.weight = conv1d.weight  # (nx, nf)
        self.bias = conv1d.bias      # (nf,)
        self.nf = conv1d.nf if hasattr(conv1d, 'nf') else conv1d.weight.shape[1]

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x_flat = x.reshape(-1, x.size(-1))
        out = triton_matmul(x_flat, self.weight, self.bias)
        return out.view(size_out)


class TritonLinear(nn.Module):
    """Replacement for nn.Linear (used as lm_head). No bias for lm_head typically."""
    def __init__(self, linear):
        super().__init__()
        self.weight = linear.weight  # (out, in)
        self.bias = linear.bias      # (out,) or None
        self.out_features = linear.out_features
        self.in_features = linear.in_features

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, self.in_features).contiguous()
        # weight is (out, in), we need to multiply x @ W.T
        # Use torch.mm for lm_head to avoid issues with tied weights / large N (vocab)
        if self.bias is not None:
            out = torch.addmm(self.bias, x_flat, self.weight.t())
        else:
            out = torch.mm(x_flat, self.weight.t())
        return out.view(size_out)


def replace_modules(model):
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:
        Conv1D = None

    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in ("NewGELUActivation", "GELUActivation", "FastGELUActivation", "QuickGELUActivation"):
                setattr(module, child_name, TritonGELU())
            elif Conv1D is not None and isinstance(child, Conv1D):
                setattr(module, child_name, TritonConv1D(child))
            elif isinstance(child, nn.Linear):
                setattr(module, child_name, TritonLinear(child))
    return model


class TritonGELU(nn.Module):
    def forward(self, x):
        return triton_gelu(x)


class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self.model = self.model.cuda()
        self.model = replace_modules(self.model)

    def forward(self, x):
        return self.model(x).logits