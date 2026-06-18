import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, C, DHW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // DHW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = (2.0 * x + b) * x + x
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def scatter_convtranspose3d_kernel(
    inp_ptr,       # [N, IC, ID, IH, IW]
    w_ptr,         # [IC, OC, KD, KH, KW]
    cb_ptr,        # [OC] conv bias
    eb_ptr,        # [OC] epilogue bias
    out_ptr,       # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # Each program handles one (n, id_, ih, iw) input position over all IC,
    # accumulates into BLOCK_OC outputs across kd, kh, kw, ic.
    # Actually let's do: one program per (n, output voxel block) -- gather GEMM
    pass


# Gather-based ConvTranspose3d: one program per (N, OC tile, output spatial position)
@triton.jit
def gather_convt3d_kernel(
    inp_ptr,      # [N, IC, ID, IH, IW]
    w_ptr,        # [IC, OC, KD, KH, KW]
    cb_ptr,       # [OC] conv bias
    eb_ptr,       # [OC] epilogue bias
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC,
    ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OH * OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < ODHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    # accumulator [BLOCK_SP, BLOCK_OC]
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For ConvTranspose3d:
    # out[n, oc, od, oh, ow] = sum over ic, kd, kh, kw of
    #   inp[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*STRIDE - PAD + kd = od  =>  id = (od + PAD - kd) / STRIDE
    # must be integer and in [0, ID)

    inp_base_n = pid_n * IC * ID * IH * IW

    for kd in tl.static_range(0, KD):
        # id_num = od + PAD - kd
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_ok = (id_num % STRIDE == 0) & (id_val >= 0) & (id_val < ID) & sp_mask
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_val = ih_num // STRIDE
            ih_ok = id_ok & (ih_num % STRIDE == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_val = iw_num // STRIDE
                valid = ih_ok & (iw_num % STRIDE == 0) & (iw_val >= 0) & (iw_val < IW)

                # input offset per sp: inp_base_n + ic*ID*IH*IW + id_val*IH*IW + ih_val*IW + iw_val
                in_sp_off = id_val * (IH * IW) + ih_val * IW + iw_val  # [BLOCK_SP]

                for ic in range(0, IC):
                    in_off = inp_base_n + ic * (ID * IH * IW) + in_sp_off
                    x = tl.load(inp_ptr + in_off, mask=valid, other=0.0)  # [BLOCK_SP]

                    # weight: w[ic, oc, kd, kh, kw], shape [IC, OC, KD, KH, KW]
                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x[:, None] * w_val[None, :]

    # add conv bias
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + cb[None, :]

    # epilogue bias [OC]
    eb = tl.load(eb_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    # original_x = acc; result = (2*acc + eb)*acc + acc
    result = (2.0 * acc + eb[None, :]) * acc + acc

    # store: out[n, oc, od, oh, ow]
    out_base_n = pid_n * OC * ODHW
    out_off = out_base_n + oc_offs[None, :] * ODHW + sp_offs[:, None]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, result, mask=store_mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(x)
    n_elements = x.numel()
    N, C, D, H, W = x.shape
    DHW = D * H * W
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, bias_flat, out, n_elements, C, DHW, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_epilogue(x, self.bias)