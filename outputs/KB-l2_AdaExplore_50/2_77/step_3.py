import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gap_affine_kernel(
    x_ptr,         # [N, C, S]
    out_ptr,       # [N, C]
    scale_ptr,     # [C]
    bias_ptr,      # [C]
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
        acc += tl.where(mask, x, 0.0)

    total = tl.sum(acc, axis=0)
    mean = total / S.to(tl.float32)

    scale = tl.load(scale_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    out = mean * scale + bias
    tl.store(out_ptr + n * C + c, out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.eps = eps

    def forward(self, x):
        # Run the (heavy) transposed conv via PyTorch's optimized cuDNN path.
        x = self.conv_transpose(x)

        if self.training:
            # Fall back to reference path during training so BN updates stats correctly.
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x

        # Eval: fuse scale * BN(affine) * GAP into one kernel.
        # y = (x*s - mean) / sqrt(var+eps) * gamma + beta
        #   = x * (s*gamma/sqrt(var+eps)) + (beta - mean*gamma/sqrt(var+eps))
        # GAP commutes with affine: GAP(x*A + B) = mean(x)*A + B
        rm = self.batch_norm.running_mean
        rv = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        inv = torch.rsqrt(rv + self.eps)
        A = self.scale_factor * gamma * inv  # [C]
        B = beta - rm * gamma * inv          # [C]

        N, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(N, C, S)
        out = torch.empty((N, C), device=x.device, dtype=x.dtype)

        # choose BLOCK_S
        if S >= 4096:
            BLOCK_S = 1024
            num_warps = 8
        elif S >= 1024:
            BLOCK_S = 512
            num_warps = 4
        elif S >= 256:
            BLOCK_S = 256
            num_warps = 4
        else:
            BLOCK_S = 128
            num_warps = 2

        grid = (N * C,)
        _gap_affine_kernel[grid](
            x_flat, out, A.contiguous(), B.contiguous(),
            N, C, S,
            BLOCK_S=BLOCK_S, num_warps=num_warps,
        )
        return out.view(N, C, 1, 1, 1)