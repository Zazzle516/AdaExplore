import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_gn_min_bias_kernel(
    x_ptr, w_ptr, b_ptr, gn_w_ptr, gn_b_ptr, bias_ptr, out_ptr,
    M, N, K,
    num_groups, group_size, eps,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # one program per row (M)
    row = tl.program_id(0)
    if row >= M:
        return

    # Compute GEMM output for the entire row: y[n] = sum_k x[row,k] * w[n,k] + b[n]
    # N must equal BLOCK_N (we process all N at once per row)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        # x: [BLOCK_K]
        x_vals = tl.load(x_ptr + row * K + offs_k, mask=k_mask, other=0.0)
        # w: [BLOCK_N, BLOCK_K] - w is stored as [N, K]
        w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
        w_mask = k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Add bias
    bias_vals = tl.load(b_ptr + offs_n)
    y = acc + bias_vals  # [BLOCK_N]

    # GroupNorm: groups along N dimension. group_size = N / num_groups.
    # For each group, compute mean and var, then normalize.
    # group index for each element: offs_n // group_size
    group_idx = offs_n // group_size

    # Compute per-group mean and variance.
    # We'll compute for each group: sum and sum of squares.
    # Use a loop over groups (num_groups is constexpr-ish... actually it's runtime).
    # Better approach: use segmented reduction via mask per group.
    # Since num_groups can be large (512), we do it differently.
    # Approach: reshape conceptually as [num_groups, group_size], compute mean/var per group.
    # We compute normalized values into y_norm.

    # Use a loop over groups. This is OK since num_groups is moderate.
    # Actually simpler: compute mean per element by checking group membership.
    # That's O(N * num_groups). For N=8192 num_groups=512, that's 4M ops per row - too much.

    # Better: reshape y into [num_groups, group_size] view via index math.
    # We compute group sums by using tl.sum with mask.
    # But we'd loop num_groups times — 512 iterations per row, 1024 rows = 512K iters.
    # Each iter has a sum over BLOCK_N. Too slow.

    # Best: use the fact that BLOCK_N = num_groups * group_size.
    # Reshape: y_2d[g, i] = y[g*group_size + i]
    # We can compute group means by doing y reshaped — but Triton can use tl.reshape.
    y_2d = tl.reshape(y, (num_groups, group_size))
    mean = tl.sum(y_2d, axis=1) / group_size  # [num_groups]
    diff = y_2d - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / group_size  # [num_groups]
    rstd = 1.0 / tl.sqrt(var + eps)
    y_norm_2d = diff * rstd[:, None]
    y_norm = tl.reshape(y_norm_2d, (BLOCK_N,))

    # Apply affine
    gn_w = tl.load(gn_w_ptr + offs_n)
    gn_b = tl.load(gn_b_ptr + offs_n)
    y_norm = y_norm * gn_w + gn_b

    # Min reduction across N
    min_val = tl.min(y_norm, axis=0)  # scalar

    # Add bias (bias is shape [1, N, 1, 1] = N elements). Output is [M, 1, N, 1] after broadcasting.
    # x shape: [M, in], gemm -> [M, out], gn -> [M, out], min over dim=1 keepdim -> [M, 1]
    # Then x + bias where bias is [1, out, 1, 1] => broadcast to [M, out, out, 1]? Let me re-check.
    # x after min: [M, 1] (2D). bias is [1, out, 1, 1] (4D).
    # Broadcasting [M, 1] + [1, out, 1, 1] => result shape: [1, out, M, 1]? 
    # Actually broadcasting aligns from the right: [M,1] vs [1,out,1,1]
    # [M,1] becomes [1,1,M,1], then broadcast with [1,out,1,1] => [1,out,M,1]
    # So output shape is [1, out, M, 1], where out[0,n,m,0] = min_val[m] + bias[n]

    # We write min_val[row] + bias[n] for n in [0, N)
    out_offs = offs_n * M + row  # output layout: [N, M] then reshape to [1, N, M, 1]
    bias_n = tl.load(bias_ptr + offs_n)
    out_vals = min_val + bias_n
    tl.store(out_ptr + out_offs, out_vals)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_size = out_features // num_groups
        self.eps = 1e-5

        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.cuda().contiguous()
        M, K = x.shape
        N = self.out_features

        w = self.gemm.weight.contiguous()  # [N, K]
        b = self.gemm.bias.contiguous()    # [N]
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        bias = self.bias.contiguous().view(-1)  # [N]

        # Output layout: we'll store as [N, M] flat, then reshape to [1, N, M, 1]
        out_flat = torch.empty((N, M), device=x.device, dtype=x.dtype)

        BLOCK_N = N  # must equal out_features (power of 2 = 8192)
        BLOCK_K = 64

        grid = (M,)
        gemm_gn_min_bias_kernel[grid](
            x, w, b, gn_w, gn_b, bias, out_flat,
            M, N, K,
            self.num_groups, self.group_size, self.eps,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=16,
            num_stages=2,
        )

        # Reshape to [1, N, M, 1] to match torch broadcasting
        return out_flat.view(1, N, M, 1)