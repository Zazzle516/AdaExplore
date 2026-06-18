import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _submean_kernel(
    x_ptr, out_ptr,
    S,
    inv_S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    row_off = pid * S

    # First pass: compute sum
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        vals = tl.where(mask, vals, 0.0)
        acc += vals
    total = tl.sum(acc, axis=0)
    mean = total * inv_S

    # Second pass: store x - mean
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        out = vals - mean
        tl.store(out_ptr + row_off + offs, out, mask=mask)


def subtract_mean(x):
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W

    out = torch.empty_like(x)
    inv_S = 1.0 / S
    BLOCK_S = 2048
    grid = (N * C,)
    _submean_kernel[grid](
        x, out,
        S, inv_S,
        BLOCK_S=BLOCK_S, num_warps=8, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        bn = self.batch_norm
        if bn.training or self.training:
            # standard path during training to keep running stats correct
            x = self.conv_transpose(x)
            x = bn(x)
            x = x - x.mean(dim=(2, 3, 4), keepdim=True)
            return x

        # Eval: fold BN affine into conv_transpose's weight/bias.
        # BN(eval): y = scale * x + shift, where
        #   scale = gamma / sqrt(var + eps), shift = beta - scale * running_mean
        # ConvTranspose3d weight has shape (in_channels, out_channels, kD, kH, kW).
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift = bn.bias - scale * bn.running_mean

        w = self.conv_transpose.weight  # (IC, OC, kD, kH, kW)
        b = self.conv_transpose.bias    # (OC,) or None

        # Multiply along OC axis (dim=1)
        w_eff = w * scale.view(1, -1, 1, 1, 1)
        if b is not None:
            b_eff = b * scale + shift
        else:
            b_eff = shift

        x = F.conv_transpose3d(
            x, w_eff, b_eff,
            stride=self.stride, padding=self.padding,
        )
        x = x.contiguous()
        return subtract_mean(x)