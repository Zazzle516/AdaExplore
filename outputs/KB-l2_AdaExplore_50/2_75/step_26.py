import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_min_bias_kernel(
    y_ptr, gn_w_ptr, gn_b_ptr, bias_ptr, out_ptr,
    M, N,
    eps,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row.
    row = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    # Load full row of GEMM output [N]
    y = tl.load(y_ptr + row * N + offs_n).to(tl.float32)

    # GroupNorm: reshape to [NUM_GROUPS, GROUP_SIZE]
    y_2d = tl.reshape(y, (NUM_GROUPS, GROUP_SIZE))
    mean = tl.sum(y_2d, axis=1) / GROUP_SIZE  # [NUM_GROUPS]
    diff = y_2d - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)
    y_norm_2d = diff * rstd[:, None]
    y_norm = tl.reshape(y_norm_2d, (BLOCK_N,))

    # Affine
    gn_w = tl.load(gn_w_ptr + offs_n)
    gn_b = tl.load(gn_b_ptr + offs_n)
    y_norm = y_norm * gn_w + gn_b

    # Min over N
    min_val = tl.min(y_norm, axis=0)

    # Output layout [N, M] -> view as [1, N, M, 1]
    bias_n = tl.load(bias_ptr + offs_n)
    out_vals = min_val + bias_n
    tl.store(out_ptr + offs_n * M + row, out_vals)


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

        # Use cuBLAS for the heavy GEMM
        y = F.linear(x, self.gemm.weight, self.gemm.bias).contiguous()  # [M, N]

        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        bias = self.bias.contiguous().view(-1)  # [N]

        out_flat = torch.empty((N, M), device=x.device, dtype=x.dtype)

        BLOCK_N = N  # 8192
        grid = (M,)
        gn_min_bias_kernel[grid](
            y, gn_w, gn_b, bias, out_flat,
            M, N,
            self.eps,
            NUM_GROUPS=self.num_groups,
            GROUP_SIZE=self.group_size,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=2,
        )

        return out_flat.view(1, N, M, 1)