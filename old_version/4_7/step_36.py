import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2Block


# ---------------- LayerNorm (Welford) ----------------
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * N
    y_ptr += row * N

    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(y_ptr + offs, y, mask=mask)


def triton_layernorm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    M = x2.shape[0]
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK >= 2048:
        num_warps = 8
    if BLOCK >= 4096:
        num_warps = 16
    layernorm_kernel[(M,)](x2, weight, bias, y, N, eps, BLOCK=BLOCK, num_warps=num_warps)
    return y.reshape(orig_shape)


# ---------------- GELU (tanh approx) ----------------
@triton.jit
def gelu_tanh_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GPT2 uses NewGELUActivation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    # tanh via exp
    e2 = tl.exp(2.0 * inner)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = 0.5 * x * (1.0 + th)
    tl.store(y_ptr + offs, y, mask=mask)


def triton_gelu(x):
    x_c = x.contiguous()
    y = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    gelu_tanh_kernel[grid](x_c, y, n, BLOCK=BLOCK, num_warps=4)
    return y.reshape(x.shape)


# ---------------- Matmul with optional transpose for B ----------------
# Computes C = A @ B (or A @ B.T if TRANS_B) + optional bias (broadcast on last dim)
@triton.autotune(
    configs=[
        triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'TRANS_B'],
)
@triton.jit
def matmul_kernel(
    A, B, C, Bias,
    M, N, K,
    sa0, sa1,
    sb0, sb1,
    sc0, sc1,
    TRANS_B: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + offs_m[:, None] * sa0 + offs_k[None, :] * sa1
    if TRANS_B:
        # B is [N, K], B.T element [k,n] = B[n, k]
        b_ptrs = B + offs_n[None, :] * sb0 + offs_k[:, None] * sb1
    else:
        b_ptrs = B + offs_k[:, None] * sb0 + offs_n[None, :] * sb1

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k0 in range(0, K, BK):
        k_idx = k0 + offs_k
        mask_k = k_idx < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        if TRANS_B:
            b = tl.load(b_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        else:
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * sa1
        if TRANS_B:
            b_ptrs += BK * sb1
        else:
            b_ptrs += BK * sb0

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c_ptrs = C + offs_m[:, None] * sc0 + offs_n[None, :] * sc1
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(a, b, bias=None, trans_b=False):
    # a: [M,K]; b: [K,N] if not trans_b else [N,K]
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    if trans_b:
        N = b.shape[0]
        assert b.shape[1] == K
    else:
        assert b.shape[0] == K
        N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    has_bias = bias is not None
    bias_t = bias if has_bias else a  # dummy
    grid = lambda META: (triton.cdiv(M, META['BM']), triton.cdiv(N, META['BN']))
    matmul_kernel[grid](
        a, b, c, bias_t,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        TRANS_B=trans_b,
        HAS_BIAS=has_bias,
    )
    return c


def linear_via_triton(x, weight, bias):
    # x: [..., in], weight: [out, in], bias: [out]
    orig_shape = x.shape
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    x2 = x.reshape(-1, in_features)
    y = triton_matmul(x2, weight, bias=bias, trans_b=True)
    return y.reshape(*orig_shape[:-1], out_features)


def conv1d_via_triton(x, weight, bias):
    # GPT2 Conv1D: y = x @ weight + bias; weight: [in, out]
    orig_shape = x.shape
    in_features = weight.shape[0]
    out_features = weight.shape[1]
    x2 = x.reshape(-1, in_features)
    y = triton_matmul(x2, weight, bias=bias, trans_b=False)
    return y.reshape(*orig_shape[:-1], out_features)


# ---------------- Custom MLP ----------------
class FastGPT2MLP(nn.Module):
    def __init__(self, orig: GPT2MLP):
        super().__init__()
        self.c_fc_w = orig.c_fc.weight
        self.c_fc_b = orig.c_fc.bias
        self.c_proj_w = orig.c_proj.weight
        self.c_proj_b = orig.c_proj.bias

    def forward(self, x):
        h = conv1d_via_triton(x, self.c_fc_w, self.c_fc_b)
        h = triton_gelu(h)
        h = conv1d_via_triton(h, self.c_proj_w, self.c_proj_b)
        return h


# ---------------- Custom Attention (uses triton matmul for QKV/proj, sdpa for attn) ----------------
class FastGPT2Attention(nn.Module):
    def __init__(self, orig: GPT2Attention):
        super().__init__()
        self.embed_dim = orig.embed_dim
        self.num_heads = orig.num_heads
        self.head_dim = orig.head_dim
        self.split_size = orig.split_size
        self.c_attn_w = orig.c_attn.weight
        self.c_attn_b = orig.c_attn.bias
        self.c_proj_w = orig.c_proj.weight
        self.c_proj_b = orig.c_proj.bias
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, hidden_states, layer_past=None, attention_mask=None,
                head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False, **kwargs):
        B, T, C = hidden_states.shape
        qkv = conv1d_via_triton(hidden_states, self.c_attn_w, self.c_attn_b)
        q, k, v = qkv.split(self.split_size, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # causal scaled dot product
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = conv1d_via_triton(attn_out, self.c_proj_w, self.c_proj_b)
        outputs = (out, None)
        if output_attentions:
            outputs = outputs + (None,)
        return outputs


# ---------------- Custom Block ----------------
class FastGPT2Block(nn.Module):
    def __init__(self, orig: GPT2Block):
        super().__init__()
        self.ln_1_w = orig.ln_1.weight
        self.ln_1_b = orig.ln_1.bias
        self.ln_1_eps = orig.ln_1.eps
        self.ln_2_w = orig.ln_2.weight
        self.ln_2_b = orig.ln_2.bias
        self.ln_2_eps = orig.ln_2.eps
        self.attn = FastGPT2Attention(orig.attn)
        self.mlp = FastGPT2MLP(orig.mlp)

    def forward(self, hidden_states, layer_past=None, attention_mask=None,
                head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False, **kwargs):
        residual = hidden_states
        h = triton_layernorm(hidden_states, self.ln_1_w, self.ln_1_b, self.ln_1_eps)
        attn_outputs = self.attn(h, output_attentions=output_attentions)
        h = attn_outputs[0]
        hidden_states = residual + h

        residual = hidden_states
        h = triton_layernorm(hidden_states, self.ln_2_w, self.ln_2_b, self.ln_2_eps)
        h = self.mlp(h)
        hidden_states = residual + h
        outputs = (hidden_states,)
        return outputs


class ModelNew(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        self.model = self.model.cuda()

        # Replace blocks
        transformer = self.model.transformer
        new_blocks = nn.ModuleList([FastGPT2Block(b) for b in transformer.h])
        transformer.h = new_blocks

        # Save final layernorm params
        self.ln_f_w = transformer.ln_f.weight
        self.ln_f_b = transformer.ln_f.bias
        self.ln_f_eps = transformer.ln_f.eps
        self.transformer = transformer
        # tied lm_head -> use wte.weight
        self.wte_weight = transformer.wte.weight

    def forward(self, x):
        x = x.cuda()
        transformer = self.transformer
        B, T = x.shape
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        inputs_embeds = transformer.wte(x)
        position_embeds = transformer.wpe(position_ids)
        hidden = inputs_embeds + position_embeds

        for block in transformer.h:
            outputs = block(hidden)
            hidden = outputs[0]

        hidden = triton_layernorm(hidden, self.ln_f_w, self.ln_f_b, self.ln_f_eps)
        # lm head: tied with wte; weight shape [vocab, hidden]
        logits = linear_via_triton(hidden, self.wte_weight, None)
        return logits