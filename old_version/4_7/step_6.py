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
    ],
    key=['M', 'N', 'K', 'HAS_BIAS'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr, BIAS_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tl.static_assert(BLOCK_K > 0)
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

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if HAS_BIAS:
        bias = tl.load(BIAS_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


def triton_gemm(a: torch.Tensor, b: torch.Tensor, bias=None, trans_b: bool = False):
    # a: [M, K]
    # if trans_b False: b is [K, N] -> [M, N]
    # if trans_b True: b is [N, K] -> [M, N]
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    if trans_b:
        N, K2 = b.shape
        # interpret b as [K, N] with strides (1, K) i.e., swap
        stride_bk = b.stride(1)  # =1
        stride_bn = b.stride(0)  # =K
    else:
        K2, N = b.shape
        stride_bk = b.stride(0)
        stride_bn = b.stride(1)
    assert K == K2, f"K mismatch: {K} vs {K2}"
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
        stride_bk, stride_bn,
        out.stride(0), out.stride(1),
        HAS_BIAS=has_bias,
    )
    return out


class TritonConv1D(nn.Module):
    """Replacement for HuggingFace Conv1D: y = x @ W + b, W shape [in, out]."""
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight  # [in, out]
        self.bias = bias
        self.nf = weight.shape[1]

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x_2d = x.reshape(-1, x.size(-1))
        out = triton_gemm(x_2d, self.weight, self.bias)
        return out.view(*size_out)


class TritonLinear(nn.Module):
    """Replacement for nn.Linear: y = x @ W^T + b. Uses weight directly via trans_b."""
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight  # [out, in]
        self.bias = bias
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_features,)
        x_2d = x.reshape(-1, x.size(-1))
        out = triton_gemm(x_2d, self.weight, self.bias, trans_b=True)
        return out.view(*size_out)


def _replace_modules(model):
    for name, module in model.named_children():
        if isinstance(module, Conv1D):
            new_mod = TritonConv1D(module.weight, module.bias)
            setattr(model, name, new_mod)
        elif isinstance(module, nn.Linear):
            new_mod = TritonLinear(module.weight, module.bias)
            setattr(model, name, new_mod)
        else:
            _replace_modules(module)


class ModelNew(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)
        self.model = self.model.cuda()
        _replace_modules(self.model)

    def forward(self, x):
        return self.model(x).logits