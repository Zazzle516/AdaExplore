import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lse_mish_kernel(
    in_ptr, out_ptr,
    M, N,
    SCALE2: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = in_ptr + pid * N

    max_val = -float('inf')
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        v = tl.load(row_ptr + offs, mask=mask, other=-float('inf'))
        v = v * SCALE2
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        v = tl.where(mask, v, -float('inf'))
        m_blk = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, m_blk)

    sum_exp = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        v = tl.load(row_ptr + offs, mask=mask, other=-float('inf'))
        v = v * SCALE2
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    lse = max_val + tl.log(sum_exp)
    sp = tl.log(1.0 + tl.exp(lse))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    mish_lse = lse * tanh_sp
    out_val = lse * mish_lse
    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.matmul = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight
        b = self.matmul.bias
        M, K = x.shape
        N = W.shape[0]

        # Use cuBLAS via addmm for the GEMM (very fast on 4090)
        partial = torch.addmm(b, x, W.t())

        scale2 = self.scale_factor * 2.0

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        BLOCK_N = 2048
        lse_mish_kernel[(M,)](
            partial, out, M, N,
            SCALE2=scale2,
            CLAMP_MIN=self.clamp_min,
            CLAMP_MAX=self.clamp_max,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out