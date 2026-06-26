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
def matmul_bias_kernel(
    A_ptr, B_ptr, C_ptr, Bias_ptr,
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_am < M
    mask_n = offs_bn < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_bn, mask=mask_n, other=0.0)
        acc += bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a, b, bias=None):
    # a: [M, K], b: [K, N], bias: [N] or None
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    has_bias = bias is not None
    bias_ptr = bias if has_bias else a  # dummy
    matmul_bias_kernel[grid](
        a, b, c, bias_ptr,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=4, num_stages=3,
    )
    return c


class TritonConv1D(nn.Module):
    """Replacement for transformers.pytorch_utils.Conv1D.
    Forward: y = x @ weight + bias  where weight: [nx, nf]
    """
    def __init__(self, conv1d):
        super().__init__()
        self.weight = conv1d.weight  # [nx, nf]
        self.bias = conv1d.bias      # [nf]
        self.nf = conv1d.nf if hasattr(conv1d, 'nf') else conv1d.weight.shape[1]

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x2d = x.reshape(-1, x.size(-1))
        out = triton_matmul(x2d, self.weight, self.bias)
        return out.view(*size_out)


class TritonLinear(nn.Module):
    """Replacement for nn.Linear: y = x @ W^T + b"""
    def __init__(self, linear):
        super().__init__()
        self.weight = linear.weight  # [out, in]
        self.bias = linear.bias
        self.out_features = linear.out_features
        # Pre-transpose weight for matmul efficiency? Keep as-is and transpose lazily
        self._wt = None

    def _get_wt(self):
        if self._wt is None or self._wt.data_ptr() != self.weight.data_ptr() + 0:
            self._wt = self.weight.t().contiguous()
        return self._wt

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_features,)
        x2d = x.reshape(-1, x.size(-1))
        wt = self.weight.t().contiguous()
        out = triton_matmul(x2d, wt, self.bias)
        return out.view(*size_out)


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
    # Replace GELU activations, LayerNorms, Conv1D and Linear in the GPT-2 model
    try:
        from transformers.pytorch_utils import Conv1D as HFConv1D
    except Exception:
        HFConv1D = None
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            cls_name = type(child).__name__
            if cls_name in ("NewGELUActivation", "GELUActivation", "FastGELUActivation", "QuickGELUActivation"):
                setattr(module, child_name, TritonGELU())
            elif isinstance(child, nn.LayerNorm):
                setattr(module, child_name, TritonLayerNorm(child))
            elif HFConv1D is not None and isinstance(child, HFConv1D):
                setattr(module, child_name, TritonConv1D(child))
            elif isinstance(child, nn.Linear):
                setattr(module, child_name, TritonLinear(child))
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