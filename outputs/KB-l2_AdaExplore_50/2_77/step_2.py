import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Strategy:
# At eval-time (and we use eval-mode BN), the entire pipeline collapses to:
#   y = conv_transpose3d(x, W, b) * scale
#   y_norm = (y - running_mean) * (gamma / sqrt(running_var + eps)) + beta
#         = y * a + c     where  a = scale * gamma / sqrt(running_var + eps)
#                                c = beta - running_mean * gamma / sqrt(running_var + eps)
#   out[n, oc] = mean over (D_out, H_out, W_out) of y_norm[n, oc, ...]
#
# Since mean is linear:
#   out[n, oc] = a[oc] * mean(conv_transpose_output[n, oc, ...]) + c[oc]
#
# We need to compute S[n, oc] = sum over output spatial positions of conv_transpose
# value, then divide by D_out*H_out*W_out, then apply affine.
#
# Sum over output of convT = sum_{od,oh,ow} sum_{ic,kd,kh,kw} x[n,ic,od-kd,oh-kh,ow-kw] * W[ic,oc,kd,kh,kw]
# Swap sums:
#   = sum_{ic,kd,kh,kw} W[ic,oc,kd,kh,kw] * sum_{od,oh,ow valid} x[n,ic,od-kd,oh-kh,ow-kw]
#   = sum_{ic} sum_{kd,kh,kw} W[ic,oc,kd,kh,kw] * sum_{id,ih,iw in input} x[n,ic,id,ih,iw]
#     (every input position contributes to all kernel positions since padding=0, stride=1)
#   = sum_{ic} (sum_{id,ih,iw} x[n,ic,id,ih,iw]) * (sum_{kd,kh,kw} W[ic,oc,kd,kh,kw])
#
# Wait — but that's a graph-level algebraic shortcut. The safety contract says
# "Every operator in the reference forward must execute at runtime on the actual
# input tensor. Do not collapse a heavy op into a downstream reduction at init
# time (for example, precomputing a column/row sum of a weight so that forward
# runs a matvec instead of the full operator)."
#
# So we cannot precompute weight_sum at init. But we CAN compute it at runtime
# on the actual weight tensor. The contract bars init-time precomputation; a
# runtime compute on the weight is allowed but doesn't help because the weight
# doesn't change between calls. Still — we must execute conv_transpose on the
# actual input.
#
# Re-reading: "Every operator in the reference forward must execute at runtime
# on the actual input tensor." — we must actually do the convT.
#
# OK, so we must run the conv_transpose. Let's just make the convT fast and
# fuse scale+BN+avgpool aggressively.
#
# Best plan: use torch's cuDNN convT (already optimized), then fuse the rest:
# scale * BN_affine reduces to elementwise affine, then global avg pool.
# Fuse: out[n,oc] = a[oc] * mean(convT_out[n,oc,...]) + c[oc]
# We can do reduction first, then affine — saves a full pass over data.


@triton.jit
def _sum_pool_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = n * C * S + c * S
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s_start in range(0, S, BLOCK):
        offs = s_start + tl.arange(0, BLOCK)
        mask = offs < S
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += vals
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, total / S)


@triton.jit
def _affine_kernel(
    mean_ptr, a_ptr, c_ptr, out_ptr,
    N, C,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N * C
    c_idx = offs % C
    m = tl.load(mean_ptr + offs, mask=mask, other=0.0)
    a = tl.load(a_ptr + c_idx, mask=mask, other=0.0)
    cc = tl.load(c_ptr + c_idx, mask=mask, other=0.0)
    out = m * a + cc
    tl.store(out_ptr + offs, out, mask=mask)


def triton_fused_avgpool_affine(x: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    N, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    mean = torch.empty((N, C), device=x.device, dtype=x.dtype)
    BLOCK = 1024
    _sum_pool_kernel[(N * C,)](x_c, mean, N, C, S, BLOCK=BLOCK, num_warps=4)

    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    BLOCK2 = 256
    grid2 = ((N * C + BLOCK2 - 1) // BLOCK2,)
    _affine_kernel[grid2](mean, a, c, out, N, C, BLOCK=BLOCK2, num_warps=4)
    return out.view(N, C, 1, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.eps = eps

    def forward(self, x):
        # Execute conv_transpose on actual input (required by safety contract).
        y = self.conv_transpose(x)
        # Scale * BN(eval) folds to: y_norm = y * a + c, where
        #   a = scale * gamma / sqrt(running_var + eps)
        #   c = beta - running_mean * (gamma / sqrt(running_var + eps))
        # But note BN sees (y * scale), so:
        #   y_norm = ((y*scale) - mean) * gamma/sqrt(var+eps) + beta
        #          = y * (scale * gamma / sqrt(var+eps)) + (beta - mean*gamma/sqrt(var+eps))
        if self.training:
            # Fall back to standard path during training (BN stats update).
            y = y * self.scale_factor
            y = self.batch_norm(y)
            return self.global_avg_pool(y)

        bn = self.batch_norm
        inv = torch.rsqrt(bn.running_var + bn.eps)
        a = (self.scale_factor * bn.weight * inv).contiguous()
        c = (bn.bias - bn.running_mean * bn.weight * inv).contiguous()
        return triton_fused_avgpool_affine(y, a, c)