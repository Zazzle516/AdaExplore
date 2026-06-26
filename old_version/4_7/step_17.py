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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K', 'FUSE_GELU'],
)
@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    FUSE_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remain)
        w_mask = (offs_k[:, None] < k_remain) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :].to(tl.float32)

    if FUSE_GELU:
        # GELU 'new' approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        c = 0.7978845608028654
        x3 = acc * acc * acc
        inner = c * (acc + 0.044715 * x3)
        # tanh via exp
        e2 = tl.exp(2.0 * inner)
        t = (e2 - 1.0) / (e2 + 1.0)
        acc = 0.5 * acc * (1.0 + t)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, fuse_gelu: bool = False):
    # x: (..., K), weight: (K, N), bias: (N,)
    orig_shape = x.shape
    K = weight.shape[0]
    N = weight.shape[1]
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    w = weight.contiguous()
    b = bias.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    conv1d_kernel[grid](
        x2, w, b, out,
        M, N, K,
        x2.stride(0), x2.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        FUSE_GELU=fuse_gelu,
    )
    return out.reshape(*orig_shape[:-1], N)


class FusedConv1D(nn.Module):
    def __init__(self, orig: Conv1D, fuse_gelu: bool = False):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.nf = orig.nf
        self.fuse_gelu = fuse_gelu

    def forward(self, x):
        return triton_conv1d(x, self.weight, self.bias, self.fuse_gelu)


class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)

        # Replace all Conv1D layers in transformer blocks with custom Triton kernel
        for block in self.model.transformer.h:
            # Attention projections
            block.attn.c_attn = FusedConv1D(block.attn.c_attn, fuse_gelu=False)
            block.attn.c_proj = FusedConv1D(block.attn.c_proj, fuse_gelu=False)
            # MLP: fuse GELU into c_fc, keep c_proj plain
            block.mlp.c_fc = FusedConv1D(block.mlp.c_fc, fuse_gelu=True)
            block.mlp.c_proj = FusedConv1D(block.mlp.c_proj, fuse_gelu=False)
            # Replace activation with identity since it's fused
            block.mlp.act = nn.Identity()

    def forward(self, x):
        return self.model(x).logits