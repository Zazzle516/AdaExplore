import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.pytorch_utils import Conv1D


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K', 'HAS_BIAS'],
)
@triton.jit
def gemm_bias_kernel(
    A, B, C, Bias,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        k_mask2 = offs_k[:, None] < (K - k)
        b = tl.load(b_ptrs, mask=k_mask2 & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_gemm_bias(a, b, bias=None):
    # a: [M, K], b: [K, N], bias: [N] or None
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    has_bias = bias is not None
    if has_bias:
        bias_c = bias.contiguous()
    else:
        bias_c = torch.empty(1, device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    gemm_bias_kernel[grid](
        a, b, c, bias_c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=has_bias,
    )
    return c


class TritonConv1D(nn.Module):
    """Replacement for HF Conv1D: y = x @ weight + bias, weight shape (nx, nf)."""
    def __init__(self, orig: Conv1D):
        super().__init__()
        self.nf = orig.nf
        self.weight = nn.Parameter(orig.weight.detach().clone())
        self.bias = nn.Parameter(orig.bias.detach().clone()) if orig.bias is not None else None

    def forward(self, x):
        size_out = x.shape[:-1] + (self.nf,)
        x2d = x.reshape(-1, x.shape[-1])
        out = triton_gemm_bias(x2d, self.weight, self.bias)
        return out.view(size_out)


class TritonLMHead(nn.Module):
    """Linear without bias, for tied lm_head. Weight is taken from wte at forward time."""
    def __init__(self, wte: nn.Embedding):
        super().__init__()
        self.wte = wte  # reference, not copy

    def forward(self, x):
        # weight: [vocab, hidden], we need x @ weight.T -> b = weight.T => [hidden, vocab]
        w = self.wte.weight  # [vocab, hidden]
        size_out = x.shape[:-1] + (w.shape[0],)
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        wt = w.t().contiguous()
        out = triton_gemm_bias(x2d, wt, None)
        return out.view(size_out)


def _replace_conv1d(module):
    for name, child in list(module.named_children()):
        if isinstance(child, Conv1D):
            setattr(module, name, TritonConv1D(child))
        else:
            _replace_conv1d(child)


class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        # Replace all Conv1D modules in transformer blocks
        _replace_conv1d(self.model.transformer)
        # Replace lm_head with triton-backed linear that uses tied weight
        self.model.lm_head = TritonLMHead(self.model.transformer.wte)

    def forward(self, x):
        return self.model(x).logits