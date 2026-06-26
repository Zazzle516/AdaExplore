import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.pytorch_utils import Conv1D


# -----------------------------
# GEMM with optional bias + GELU epilogue
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K', 'TRANS_B', 'ACT'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr, Bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    TRANS_B: tl.constexpr,   # if True, B is [N,K] (we transpose on the fly)
    ACT: tl.constexpr,       # 0 = none, 1 = GELU (tanh approx)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    if TRANS_B:
        # B is [N, K], stride_bk = stride along N (row), stride_bn = stride along K (col)
        # We treat B[k, n] as B_orig[n, k]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bk + offs_k[:, None] * stride_bn)
    else:
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_remaining), other=0.0)
        if TRANS_B:
            b = tl.load(b_ptrs, mask=mask_n[None, :] & (offs_k[:, None] < k_remaining), other=0.0)
        else:
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        if TRANS_B:
            b_ptrs += BLOCK_K * stride_bn
        else:
            b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    if ACT == 1:
        # GELU tanh approximation
        c = 0.7978845608028654  # sqrt(2/pi)
        x = acc
        inner = c * (x + 0.044715 * x * x * x)
        # tanh via exp
        e2 = tl.exp(2.0 * inner)
        t = (e2 - 1.0) / (e2 + 1.0)
        acc = 0.5 * x * (1.0 + t)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_gemm(a: torch.Tensor, b: torch.Tensor, bias=None, trans_b=False, act=0):
    """
    a: [M, K]
    b: [K, N] if not trans_b else [N, K]
    Returns [M, N]
    """
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    if trans_b:
        N, K2 = b.shape
    else:
        K2, N = b.shape
    assert K == K2

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    has_bias = bias is not None
    bias_ptr = bias.contiguous() if has_bias else a

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    gemm_kernel[grid](
        a, b, out, bias_ptr,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=has_bias,
        TRANS_B=trans_b,
        ACT=act,
    )
    return out


# -----------------------------
# Modules
# -----------------------------
class TritonConv1D(nn.Module):
    """Replacement for HuggingFace Conv1D. weight: [in, out], bias: [out]."""
    def __init__(self, orig: Conv1D, fuse_gelu=False):
        super().__init__()
        self.nf = orig.nf
        self.weight = orig.weight  # [in, out]
        self.bias = orig.bias      # [out]
        self.fuse_gelu = fuse_gelu

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x2d = x.reshape(-1, x.size(-1))
        act = 1 if self.fuse_gelu else 0
        out = triton_gemm(x2d, self.weight, self.bias, trans_b=False, act=act)
        return out.view(*size_out)


class TritonLinear(nn.Module):
    """Replacement for nn.Linear (e.g. lm_head). weight: [out, in], bias optional.
       Reads live weight tensor each forward (supports tied embeddings)."""
    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.out_features = orig.out_features
        self.in_features = orig.in_features
        self.weight = orig.weight  # [out, in]
        self.bias = orig.bias

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_features,)
        x2d = x.reshape(-1, x.size(-1)).contiguous()
        # weight is [out, in] = [N, K]; use trans_b
        out = triton_gemm(x2d, self.weight, self.bias, trans_b=True, act=0)
        return out.view(*size_out)


def _replace_modules(model):
    # Replace MLP c_fc with fused GELU version
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
    for name, module in model.named_children():
        if isinstance(module, GPT2MLP):
            # c_fc -> conv1d with GELU; act -> Identity; c_proj -> conv1d
            module.c_fc = TritonConv1D(module.c_fc, fuse_gelu=True)
            module.act = nn.Identity()
            module.c_proj = TritonConv1D(module.c_proj, fuse_gelu=False)
            # Also recurse into the rest of MLP (already handled)
            _replace_modules(module)
        elif isinstance(module, Conv1D):
            setattr(model, name, TritonConv1D(module, fuse_gelu=False))
        elif isinstance(module, nn.Linear):
            setattr(model, name, TritonLinear(module))
        else:
            _replace_modules(module)


class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self.model = self.model.cuda()
        _replace_modules(self.model)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        return self.model(x).logits