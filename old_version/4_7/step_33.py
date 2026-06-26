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
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K', 'ACT'],
)
@triton.jit
def matmul_kernel(
    A, B, C, bias_ptr,
    M, N, K,
    sA0, sA1,
    sB0, sB1,
    sC0, sC1,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,  # 0 none, 1 gelu_new
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    a_ptrs = A + (offs_m[:, None] * sA0 + offs_k[None, :] * sA1)
    b_ptrs = B + (offs_k[:, None] * sB0 + offs_n[None, :] * sB1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * sA1
        b_ptrs += BLOCK_K * sB0

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

    if ACT == 1:
        # GELU new (tanh approx as used by GPT-2)
        c = 0.7978845608028654  # sqrt(2/pi)
        x = acc
        inner = c * (x + 0.044715 * x * x * x)
        # numerically stable tanh via sigmoid: tanh(z) = 2*sigmoid(2z) - 1
        t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        acc = 0.5 * x * (1.0 + t)

    c_ptrs = C + offs_m[:, None] * sC0 + offs_n[None, :] * sC1
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(a, b, bias=None, act=0):
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    has_bias = bias is not None
    bias_ptr = bias if has_bias else a
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, out, bias_ptr,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=has_bias,
        ACT=act,
    )
    return out


class TritonLinear(nn.Module):
    def __init__(self, B, bias, act=0):
        super().__init__()
        self.weight_b = nn.Parameter(B.contiguous().to(torch.float32), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.contiguous().to(torch.float32), requires_grad=False)
        else:
            self.bias = None
        self.act = act

    def forward(self, x):
        orig_shape = x.shape
        in_features = self.weight_b.shape[0]
        out_features = self.weight_b.shape[1]
        x2 = x.reshape(-1, in_features)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if x2.dtype != torch.float32:
            x2 = x2.to(torch.float32)
        out = triton_matmul(x2, self.weight_b, self.bias, self.act)
        new_shape = orig_shape[:-1] + (out_features,)
        return out.reshape(new_shape)


def replace_linears(module, parent_name=""):
    for name, child in list(module.named_children()):
        full = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, Conv1D):
            B = child.weight.data
            bias = child.bias.data if child.bias is not None else None
            # Detect MLP fc layer to fuse GELU; in GPT-2 it's named 'c_fc'
            act = 1 if name == "c_fc" else 0
            new = TritonLinear(B, bias, act=act)
            setattr(module, name, new)
        elif isinstance(child, nn.Linear):
            B = child.weight.data.t().contiguous()
            bias = child.bias.data if child.bias is not None else None
            new = TritonLinear(B, bias, act=0)
            setattr(module, name, new)
        else:
            replace_linears(child, full)


# Patch GPT-2 MLP to skip the activation since we fused it
def patch_mlp(module):
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
    for child in module.modules():
        if isinstance(child, GPT2MLP):
            child.act = nn.Identity()


class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self.model = self.model.cuda()
        replace_linears(self.model)
        patch_mlp(self.model)

    def forward(self, x):
        return self.model(x).logits