import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_gelu_partial_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    PartMax_ptr, PartSum_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_pm, stride_pn,  # for PartMax / PartSum: [M, num_n_tiles]
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

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n)
    acc += bias[None, :]

    # GELU (tanh approximation)
    x = acc
    k0 = 0.7978845608028654
    k1 = 0.044715
    inner = k0 * (x + k1 * x * x * x)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * x * (1.0 + tanh_val)

    # Store gelu result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, gelu)

    # Compute partial max and partial sum-of-exp(x - partial_max) along N for this tile
    part_max = tl.max(gelu, axis=1)  # [BLOCK_M]
    part_exp = tl.exp(gelu - part_max[:, None])
    part_sum = tl.sum(part_exp, axis=1)  # [BLOCK_M]

    pm_ptrs = PartMax_ptr + offs_m * stride_pm + pid_n * stride_pn
    ps_ptrs = PartSum_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(pm_ptrs, part_max)
    tl.store(ps_ptrs, part_sum)


@triton.jit
def softmax_finalize_kernel(
    in_ptr, out_ptr,
    PartMax_ptr, PartSum_ptr,
    M, N, NUM_TILES,
    stride_m, stride_n,
    stride_pm, stride_pn,
    BLOCK_N_TILE: tl.constexpr,
    NUM_TILES_C: tl.constexpr,
):
    row = tl.program_id(0)
    # Load all partial max/sum for this row
    tile_idx = tl.arange(0, NUM_TILES_C)
    tile_mask = tile_idx < NUM_TILES
    pm = tl.load(PartMax_ptr + row * stride_pm + tile_idx * stride_pn,
                 mask=tile_mask, other=-float('inf'))
    ps = tl.load(PartSum_ptr + row * stride_pm + tile_idx * stride_pn,
                 mask=tile_mask, other=0.0)

    global_max = tl.max(pm, axis=0)
    # adjust each partial sum
    adj = ps * tl.exp(pm - global_max)
    adj = tl.where(tile_mask, adj, 0.0)
    global_sum = tl.sum(adj, axis=0)
    inv_sum = 1.0 / global_sum

    # Now stream over N in chunks, normalize and write output
    pid_n_blocks = tl.cdiv(N, BLOCK_N_TILE)
    for i in range(0, pid_n_blocks):
        offs = i * BLOCK_N_TILE + tl.arange(0, BLOCK_N_TILE)
        mask = offs < N
        x = tl.load(in_ptr + row * stride_m + offs * stride_n, mask=mask, other=0.0)
        y = tl.exp(x - global_max) * inv_sum
        tl.store(out_ptr + row * stride_m + offs * stride_n, y, mask=mask)


def next_pow2(x):
    return 1 << (x - 1).bit_length()


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        # Pre-transpose weight: store W^T shape (in_features, out_features) so that
        # B[k, n] is contiguous along n for the GEMM.
        with torch.no_grad():
            wt = self.linear.weight.detach().t().contiguous()  # (in, out)
        self.register_buffer('weight_t', wt.cuda(), persistent=False)

    def forward(self, x):
        x = x.contiguous().cuda()
        W_t = self.weight_t  # (K, N) contiguous
        b = self.linear.bias.contiguous().cuda()
        M, K = x.shape
        N = W_t.shape[1]

        out_gelu = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Estimate worst-case num_n_tiles: use smallest BLOCK_N (128) -> ceil(N/128)
        max_num_n_tiles = (N + 127) // 128
        # Actual num tiles depends on chosen config; allocate big enough.
        part_max = torch.empty((M, max_num_n_tiles), device=x.device, dtype=torch.float32)
        part_sum = torch.empty((M, max_num_n_tiles), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_gelu_partial_kernel[grid](
            x, W_t, b, out_gelu,
            part_max, part_sum,
            M, N, K,
            x.stride(0), x.stride(1),
            W_t.stride(0), W_t.stride(1),
            out_gelu.stride(0), out_gelu.stride(1),
            part_max.stride(0), part_max.stride(1),
        )

        # Determine actual num_n_tiles used by autotuned config
        best_cfg = gemm_gelu_partial_kernel.best_config
        BLOCK_N_used = best_cfg.kwargs['BLOCK_N']
        num_n_tiles = (N + BLOCK_N_used - 1) // BLOCK_N_used
        num_tiles_c = next_pow2(num_n_tiles)
        if num_tiles_c < 1:
            num_tiles_c = 1

        out = torch.empty_like(out_gelu)
        BLOCK_N_TILE = 1024
        softmax_finalize_kernel[(M,)](
            out_gelu, out,
            part_max, part_sum,
            M, N, num_n_tiles,
            out_gelu.stride(0), out_gelu.stride(1),
            part_max.stride(0), part_max.stride(1),
            BLOCK_N_TILE=BLOCK_N_TILE,
            NUM_TILES_C=num_tiles_c,
            num_warps=8, num_stages=2,
        )
        return out