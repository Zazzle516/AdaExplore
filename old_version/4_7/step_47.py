import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.pytorch_utils import Conv1D


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_gemm_bias(a, b, bias=None, stride_b0=None, stride_b1=None, b_shape=None):
    """
    a: [M, K] contiguous
    b: tensor whose strides give a [K, N] view (could be a transposed view)
    """
    assert a.is_cuda and b.is_cuda
    if not a.is_contiguous():
        a = a.contiguous()
    M, K = a.shape
    if b_shape is None:
        K2, N = b.shape
    else:
        K2, N = b_shape
    assert K == K2
    if stride_b0 is None:
        stride_b0 = b.stride(0)
        stride_b1 = b.stride(1)

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    has_bias = bias is not None
    if has_bias:
        bias_c = bias.contiguous() if not bias.is_contiguous() else bias
    else:
        bias_c = a  # dummy

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_kernel[grid](
        a, b, c, bias_c,
        M, N, K,
        a.stride(0), a.stride(1),
        stride_b0, stride_b1,
        c.stride(0), c.stride(1),
        HAS_BIAS=has_bias,
    )
    return c


class TritonConv1D(nn.Module):
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
    """LM head with tied weight from wte. weight is [vocab, hidden] - we use it as B with transposed strides."""
    def __init__(self, wte: nn.Embedding):
        super().__init__()
        self.wte = wte

    def forward(self, x):
        w = self.wte.weight  # [vocab, hidden] contiguous
        vocab, hidden = w.shape
        size_out = x.shape[:-1] + (vocab,)
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        # We want x2d @ w.T => B is [hidden, vocab] view; use w with swapped strides
        # w has strides (hidden, 1); for w.T view: shape (hidden, vocab), strides (1, hidden)
        out = triton_gemm_bias(
            x2d, w, None,
            stride_b0=w.stride(1), stride_b1=w.stride(0),
            b_shape=(hidden, vocab),
        )
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
        _replace_conv1d(self.model.transformer)
        self.model.lm_head = TritonLMHead(self.model.transformer.wte)

    def forward(self, x):
        return self.model(x).logits