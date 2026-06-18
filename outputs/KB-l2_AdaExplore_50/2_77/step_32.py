import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 4096}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_S': 8192}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_S': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_S': 16384}, num_warps=16, num_stages=2),
    ],
    key=['S'],
)
@triton.jit
def _gap_affine_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    bias_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * S + c * S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for s_off in range(0, S, BLOCK_S):
        offs = s_off + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x

    total = tl.sum(acc, axis=0)
    inv_s = 1.0 / S.to(tl.float32)
    mean = total * inv_s

    scale = tl.load(scale_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    out = mean * scale + bias
    tl.store(out_ptr + n * C + c, out)


@triton.jit
def _gap_partial_kernel(
    x_ptr,
    partial_ptr,
    N, C, S, NSPLIT,
    BLOCK_S: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_split = tl.program_id(1)
    n = pid_nc // C
    c = pid_nc % C

    base = n * C * S + c * S
    # divide S into NSPLIT roughly equal chunks
    chunk = (S + NSPLIT - 1) // NSPLIT
    s_start = pid_split * chunk
    s_end = tl.minimum(s_start + chunk, S)

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for s_off in range(s_start, s_end, BLOCK_S):
        offs = s_off + tl.arange(0, BLOCK_S)
        mask = offs < s_end
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x

    total = tl.sum(acc, axis=0)
    tl.store(partial_ptr + pid_nc * NSPLIT + pid_split, total)


@triton.jit
def _gap_finalize_kernel(
    partial_ptr,
    out_ptr,
    scale_ptr,
    bias_ptr,
    N, C, S, NSPLIT,
    BLOCK_NS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    offs = tl.arange(0, BLOCK_NS)
    mask = offs < NSPLIT
    p = tl.load(partial_ptr + pid * NSPLIT + offs, mask=mask, other=0.0)
    total = tl.sum(p, axis=0)
    inv_s = 1.0 / S.to(tl.float32)
    mean = total * inv_s

    scale = tl.load(scale_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    out = mean * scale + bias
    tl.store(out_ptr + n * C + c, out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.eps = eps
        self._cached_A = None
        self._cached_B = None
        self._cached_version = None

    def _get_AB(self):
        rm = self.batch_norm.running_mean
        rv = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        version = (rm._version, rv._version, gamma._version, beta._version)
        if self._cached_version != version or self._cached_A is None:
            inv = torch.rsqrt(rv + self.eps)
            A = (self.scale_factor * gamma * inv).contiguous()
            B = (beta - rm * gamma * inv).contiguous()
            self._cached_A = A
            self._cached_B = B
            self._cached_version = version
        return self._cached_A, self._cached_B

    def forward(self, x):
        x = self.conv_transpose(x)

        if self.training:
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x

        A, B = self._get_AB()

        N, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(N, C, S)
        out = torch.empty((N, C), device=x.device, dtype=x.dtype)

        NSPLIT = 4
        if N * C * NSPLIT >= 4096 and S >= 8192:
            partial = torch.empty((N * C, NSPLIT), device=x.device, dtype=torch.float32)
            grid1 = (N * C, NSPLIT)
            _gap_partial_kernel[grid1](
                x_flat, partial,
                N, C, S, NSPLIT,
                BLOCK_S=4096, num_warps=8, num_stages=3,
            )
            grid2 = (N * C,)
            _gap_finalize_kernel[grid2](
                partial, out, A, B,
                N, C, S, NSPLIT,
                BLOCK_NS=4, num_warps=1, num_stages=2,
            )
        else:
            grid = (N * C,)
            _gap_affine_kernel[grid](
                x_flat, out, A, B,
                N, C, S,
            )
        return out.view(N, C, 1, 1, 1)