import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    BLOCK_IC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, oc)
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    # Sum over all output positions of conv_transpose output for (n, oc)
    # out[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    #   where oh = ih + kh, ow = iw + kw  (stride=1, padding=0)
    # We compute total sum then divide by (OH*OW)

    # Iterate ic blocks, and within each, iterate over input positions
    IH_IW = IH * IW
    acc = tl.zeros((1,), dtype=tl.float32)

    # We compute: sum_{n,oc} = sum_{ic} (sum_{ih,iw} x[n,ic,ih,iw]) * (sum_{kh,kw} w[ic,oc,kh,kw])
    # WAIT - that's only valid if we're summing all output positions over ALL (oh,ow).
    # But each output position (oh,ow) gets contributions from various (ih,iw,kh,kw) with ih+kh=oh, iw+kw=ow.
    # Sum over ALL (oh, ow) of out[n,oc,oh,ow] = sum_{ih,iw,kh,kw,ic} x[n,ic,ih,iw] * w[ic,oc,kh,kw]
    #   = sum_{ic} (sum_{ih,iw} x) * (sum_{kh,kw} w)
    # This IS the algebraic shortcut forbidden by safety contract.
    # We must materialize the full conv_transpose output. Don't take this shortcut.
    pass


# Custom ConvTranspose2d kernel: scatter-add approach won't work well here.
# Instead, use im2col-style gather: for each output position, gather contributions.
# out[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh - kh, ow - kw] * w[ic, oc, kh, kw]
#   (valid when 0 <= oh-kh < IH and 0 <= ow-kw < IW)

@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW, KH, KW,
    HAS_BIAS: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N * OH, ceil(OW / BLOCK_OW), ceil(OC / BLOCK_OC))
    pid_noh = tl.program_id(0)
    pid_ow = tl.program_id(1)
    pid_oc = tl.program_id(2)

    n = pid_noh // OH
    oh = pid_noh % OH

    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    ow_mask = ow_offs < OW
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

    for kh in range(0, KH):
        ih = oh - kh
        ih_valid = (ih >= 0) & (ih < IH)
        for kw in range(0, KW):
            iw = ow_offs - kw  # [BLOCK_OW]
            iw_valid = (iw >= 0) & (iw < IW) & ow_mask
            for ic in range(0, IC):
                # x[n, ic, ih, iw]: shape [BLOCK_OW]
                x_off = ((n * IC + ic) * IH + ih) * IW + iw
                x_mask = ih_valid & iw_valid
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_OW]
                # w[ic, oc, kh, kw]: shape [BLOCK_OC]
                w_off = ((ic * OC + oc_offs) * KH + kh) * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += w_val[:, None] * x_val[None, :]

    if HAS_BIAS:
        b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
        acc += b[:, None]

    # store
    out_off = ((n * OC + oc_offs[:, None]) * OH + oh) * OW + ow_offs[None, :]
    mask = oc_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def mean_pool_kernel(
    in_ptr, out_ptr,
    N, OC, OH, OW,
    BLOCK: tl.constexpr,
):
    # one program per (n, oc), sum over OH*OW
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC
    HW = OH * OW
    base = (n * OC + oc) * HW

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        acc += v
    s = tl.sum(acc, axis=0) / HW
    tl.store(out_ptr + n * OC + oc, s)


@triton.jit
def add_bias_logsumexp_kernel(
    pooled_ptr, bias_ptr, out_ptr,
    N, OC,
    BLOCK_OC: tl.constexpr,
):
    # one program per n; compute logsumexp over OC of (pooled[n,oc] + bias[oc])
    n = tl.program_id(0)
    oc_offs = tl.arange(0, BLOCK_OC)
    mask = oc_offs < OC
    p = tl.load(pooled_ptr + n * OC + oc_offs, mask=mask, other=-float('inf'))
    b = tl.load(bias_ptr + oc_offs, mask=mask, other=0.0)
    v = tl.where(mask, p + b, -float('inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) + m
    # final: sum over (1,1) dims is just lse itself, then * 10
    tl.store(out_ptr + n, lse * 10.0)


def conv_transpose2d_triton(x, w, bias=None):
    N, IC, IH, IW = x.shape
    IC2, OC, KH, KW = w.shape
    assert IC == IC2
    OH = IH + KH - 1
    OW = IW + KW - 1
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OW = 64
    BLOCK_OC = 32
    grid = (N * OH, triton.cdiv(OW, BLOCK_OW), triton.cdiv(OC, BLOCK_OC))
    conv_transpose2d_kernel[grid](
        x, w, bias if bias is not None else x, out,
        N, IC, IH, IW, OC, OH, OW, KH, KW,
        HAS_BIAS=(bias is not None),
        BLOCK_OW=BLOCK_OW, BLOCK_OC=BLOCK_OC,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous().cuda()
        cb = self.conv_transpose.bias
        cb_c = cb.contiguous().cuda() if cb is not None else None

        # Custom conv_transpose2d
        y = conv_transpose2d_triton(x, w, cb_c)

        N, OC, OH, OW = y.shape

        # Mean pool to (N, OC)
        pooled = torch.empty((N, OC), device=y.device, dtype=y.dtype)
        grid = (N * OC,)
        mean_pool_kernel[grid](y, pooled, N, OC, OH, OW, BLOCK=1024, num_warps=4)

        # Add bias + logsumexp + sum + *10
        bias_flat = self.bias.view(-1).contiguous().cuda()
        out = torch.empty((N,), device=y.device, dtype=y.dtype)
        # find next pow2 >= OC
        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2
        add_bias_logsumexp_kernel[(N,)](pooled, bias_flat, out, N, OC, BLOCK_OC=BLOCK_OC, num_warps=4)

        return out.view(N, 1)