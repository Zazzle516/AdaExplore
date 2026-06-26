import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.pytorch_utils import Conv1D
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K', 'HAS_BIAS', 'ACT'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,  # 0 = none, 1 = gelu_tanh
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

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :].to(tl.float32)

    if ACT == 1:
        # GELU tanh approx
        c = 0.7978845608028654  # sqrt(2/pi)
        x = acc
        inner = c * (x + 0.044715 * x * x * x)
        # tanh via exp
        e2 = tl.exp(2.0 * inner)
        t = (e2 - 1.0) / (e2 + 1.0)
        acc = 0.5 * x * (1.0 + t)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def triton_gemm(a: torch.Tensor, b: torch.Tensor, bias=None, act: int = 0):
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous()
        bias_ptr = bias
    else:
        bias_ptr = a  # dummy

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    gemm_kernel[grid](
        a, b, out, bias_ptr,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=has_bias,
        ACT=act,
    )
    return out


class TritonConv1D(nn.Module):
    """Replacement for HuggingFace Conv1D. weight: [in, out], bias: [out]."""
    def __init__(self, conv1d: Conv1D, fuse_gelu: bool = False):
        super().__init__()
        self.nf = conv1d.nf
        self.weight = conv1d.weight  # [in, out]
        self.bias = conv1d.bias      # [out]
        self.fuse_gelu = fuse_gelu

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x2d = x.reshape(-1, x.size(-1))
        act = 1 if self.fuse_gelu else 0
        out = triton_gemm(x2d, self.weight, self.bias, act=act)
        return out.view(size_out)


class IdentityAct(nn.Module):
    def forward(self, x):
        return x


def replace_in_block(block):
    """Replace Conv1D modules in a transformer block. Fuse GELU into mlp.c_fc."""
    # attn
    if hasattr(block, 'attn'):
        attn = block.attn
        if hasattr(attn, 'c_attn') and isinstance(attn.c_attn, Conv1D):
            attn.c_attn = TritonConv1D(attn.c_attn, fuse_gelu=False)
        if hasattr(attn, 'c_proj') and isinstance(attn.c_proj, Conv1D):
            attn.c_proj = TritonConv1D(attn.c_proj, fuse_gelu=False)
    # mlp
    if hasattr(block, 'mlp'):
        mlp = block.mlp
        if hasattr(mlp, 'c_fc') and isinstance(mlp.c_fc, Conv1D):
            mlp.c_fc = TritonConv1D(mlp.c_fc, fuse_gelu=True)
            # replace activation with identity since fused
            if hasattr(mlp, 'act'):
                mlp.act = IdentityAct()
        if hasattr(mlp, 'c_proj') and isinstance(mlp.c_proj, Conv1D):
            mlp.c_proj = TritonConv1D(mlp.c_proj, fuse_gelu=False)


class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self.model = self.model.cuda()

        # Replace only Conv1D inside transformer blocks; keep lm_head as nn.Linear
        # to preserve weight tying with wte.
        try:
            blocks = self.model.transformer.h
            for blk in blocks:
                replace_in_block(blk)
        except AttributeError:
            pass

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        return self.model(x).logits