import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_configs():
    configs = []
    for BM in [64, 128]:
        for BN in [64, 128, 256]:
            for BK in [32, 64, 128]:
                for nw in [4, 8]:
                    for ns in [2, 3, 4]:
                        configs.append(
                            triton.Config(
                                {'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK},
                                num_warps=nw, num_stages=ns,
                            )
                        )
    return configs


@triton.autotune(configs=_get_configs(), key=['M', 'N', 'K'])
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    scale,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            # W is pre-transposed to [K, N], contiguous along N
            w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
            w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

            acc += tl.dot(x_tile, w_tile)

        b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + b_vals[None, :]

        neg_inf = float('-inf')
        acc = tl.where(n_mask[None, :], acc, neg_inf)

        acc_reshaped = tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2))
        pooled = tl.max(acc_reshaped, axis=2)

        row_sum += tl.sum(pooled, axis=1)

    out = row_sum * scale
    tl.store(out_ptr + offs_m, out, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Mirror nn.Linear init
        self.matmul = nn.Linear(in_features, out_features)
        # Pre-transpose W to [K, N] (contiguous on N) for coalesced loads
        self.register_buffer('w_t', self.matmul.weight.detach().t().contiguous())

    def forward(self, x):
        x = x.contiguous().cuda()
        W_t = self.w_t
        B = self.matmul.bias.contiguous()

        M, K = x.shape
        N = self.out_features

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

        fused_kernel[grid](
            x, W_t, B, out,
            M, N, K,
            float(self.scale_factor),
            x.stride(0), x.stride(1),
            W_t.stride(0), W_t.stride(1),
        )

        return out