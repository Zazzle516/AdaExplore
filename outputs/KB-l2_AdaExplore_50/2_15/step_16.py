import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bn_mean_sub_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    shift_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    c = pid % C
    row_offset = pid * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    # Pass 1: compute sum for mean
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        vals_bn = vals * scale + shift
        acc += tl.sum(vals_bn, axis=0)
    mean = acc / S

    # Pass 2: write bn(x) - mean
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        vals_bn = vals * scale + shift
        tl.store(out_ptr + row_offset + offs, vals_bn - mean, mask=mask)


def fused_bn_mean_sub(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    if S <= 2048:
        BLOCK_S = 1024
        num_warps = 4
        num_stages = 2
    elif S <= 8192:
        BLOCK_S = 4096
        num_warps = 8
        num_stages = 2
    else:
        BLOCK_S = 8192
        num_warps = 8
        num_stages = 2
    _bn_mean_sub_kernel[grid](x, out, scale, shift, N, C, S,
                              BLOCK_S=BLOCK_S, num_warps=num_warps, num_stages=num_stages)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        # Try channels_last_3d for the conv weights/inputs
        try:
            self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)
            self._use_cl3d = True
        except Exception:
            self._use_cl3d = False

    def forward(self, x):
        if self._use_cl3d:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)

        # Fold BN affine using running stats (eval) or batch stats (train).
        bn = self.batch_norm
        if not self.training:
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps
            inv = torch.rsqrt(var + eps)
            scale = bn.weight * inv if bn.weight is not None else inv
            shift = (bn.bias - mean * scale) if bn.bias is not None else (-mean * scale)
            x = x.contiguous()
            return fused_bn_mean_sub(x, scale.contiguous(), shift.contiguous())
        else:
            x = bn(x)
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
            return x