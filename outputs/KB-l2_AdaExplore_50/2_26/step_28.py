import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose3d with stride=2, padding=1, output_padding=1, kernel_size=3
# Output spatial dims = 2 * input spatial dims (exactly)
# For each output voxel (od, oh, ow), it gathers from input voxels:
#   id = (od + padding - kd) / stride, must be integer and in range
# With padding=1, stride=2, kernel=3:
#   For each output position, exactly ceil(K/stride) = 2 kernel positions per dim hit
#   So 2*2*2 = 8 input contributions per output voxel
#
# Specifically: id*stride = od + padding - kd
#   => kd = od + padding - id*stride
#   For id valid: kd in [0, K), and (od+padding-kd) % stride == 0
#   With padding=1, stride=2, K=3: kd has same parity as (od+1)
#   So kd in {0,2} if od is odd, kd in {1} if od is even... wait
#   od even: od+1 odd, kd odd => kd=1 (only one)
#   od odd: od+1 even, kd even => kd in {0, 2}
# So per dim, output voxels alternate: 1 contribution or 2 contributions
# Total per output: 1-8 contributions.

@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, add_ptr, extra_bias_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wi, stride_wo, stride_wd, stride_wh, stride_ww,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    IC_BLOCK: tl.constexpr,
):
    # Each program: one (n, oc_block, spatial_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    OSP = OD * OH * OW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < OSP

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load conv bias and extra bias for these output channels
    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    eb = tl.load(extra_bias_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Accumulator [BLOCK_SP, BLOCK_OC]
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Iterate over kernel positions; for stride=2, padding=1, K=3:
    # only positions where (od+padding-kd) is divisible by stride contribute
    # Process all KD*KH*KW kernel positions, mask out invalid

    for kd in tl.static_range(0, KD):
        # id = (od + padding - kd) / stride if divisible
        d_num = od + PADDING - kd
        id_ = d_num // STRIDE
        d_valid = ((d_num % STRIDE) == 0) & (id_ >= 0) & (id_ < ID) & (d_num >= 0)
        for kh in tl.static_range(0, KH):
            h_num = oh + PADDING - kh
            ih_ = h_num // STRIDE
            h_valid = ((h_num % STRIDE) == 0) & (ih_ >= 0) & (ih_ < IH) & (h_num >= 0)
            for kw in tl.static_range(0, KW):
                w_num = ow + PADDING - kw
                iw_ = w_num // STRIDE
                w_valid = ((w_num % STRIDE) == 0) & (iw_ >= 0) & (iw_ < IW) & (w_num >= 0)

                spatial_valid = d_valid & h_valid & w_valid & sp_mask

                # For this kernel position, do GEMM over IC
                # x[n, ic, id, ih, iw] @ w[ic, oc, kd, kh, kw]
                # x shape [BLOCK_SP] (per ic), w shape [BLOCK_OC] (per ic)
                # Iterate over IC in chunks of IC_BLOCK
                for ic_start in range(0, IC, IC_BLOCK):
                    ic_offs = ic_start + tl.arange(0, IC_BLOCK)
                    ic_mask = ic_offs < IC

                    # x_ptrs: [BLOCK_SP, IC_BLOCK]
                    x_ptrs = (x_ptr + pid_n * stride_xn
                              + ic_offs[None, :] * stride_xc
                              + id_[:, None] * stride_xd
                              + ih_[:, None] * stride_xh
                              + iw_[:, None] * stride_xw)
                    x_load_mask = spatial_valid[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                    # w_ptrs: [IC_BLOCK, BLOCK_OC]
                    w_ptrs = (w_ptr
                              + ic_offs[:, None] * stride_wi
                              + oc_offs[None, :] * stride_wo
                              + kd * stride_wd
                              + kh * stride_wh
                              + kw * stride_ww)
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals)

    # Add conv bias (broadcast over spatial)
    acc = acc + cb[None, :]

    # Load add_input [BLOCK_SP, BLOCK_OC]
    add_ptrs = (add_ptr + pid_n * stride_on
                + oc_offs[None, :] * stride_oc
                + od[:, None] * stride_od
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow)
    add_mask = sp_mask[:, None] & oc_mask[None, :]
    a_vals = tl.load(add_ptrs, mask=add_mask, other=0.0)

    v = acc + a_vals + eb[None, :]
    # hardswish(v) = v * clip(v+3, 0, 6) / 6
    hs = v * tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0) * (1.0 / 6.0)
    out = v * hs

    out_ptrs = (out_ptr + pid_n * stride_on
                + oc_offs[None, :] * stride_oc
                + od[:, None] * stride_od
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow)
    tl.store(out_ptrs, out, mask=add_mask)


def fused_convtranspose3d(x, weight, conv_bias, add_input, extra_bias,
                          stride, padding, output_padding):
    """
    x: (N, IC, ID, IH, IW)
    weight: (IC, OC, KD, KH, KW)
    conv_bias: (OC,)
    add_input: (N, OC, OD, OH, OW)
    extra_bias: (OC, 1, 1, 1, 1) -> flattened to (OC,)
    """
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    OD = (ID - 1) * stride - 2 * padding + KD + output_padding
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    x = x.contiguous()
    weight = weight.contiguous()
    add_input = add_input.contiguous()

    BLOCK_OC = 32
    BLOCK_SP = 64
    IC_BLOCK = 32

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OD * OH * OW, BLOCK_SP))

    conv_transpose3d_fused_kernel[grid](
        x, weight, conv_bias, add_input, extra_bias.view(-1), out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        KD=KD, KH=KH, KW=KW,
        STRIDE=stride, PADDING=padding,
        IC_BLOCK=IC_BLOCK,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x, add_input):
        # Note: original forward does NOT add self.bias; only conv bias is added.
        # forward: conv_transpose(x) + add_input, then v * hardswish(v)
        # We pass extra_bias=0 to keep behavior same as reference.
        zero_bias = torch.zeros_like(self.conv_transpose.bias)
        return fused_convtranspose3d(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            add_input, zero_bias,
            self.stride, self.padding, self.output_padding,
        )