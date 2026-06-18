import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr,          # [N, ID, IH, IW, IC] NDHWC
    w_ptr,          # [KD, KH, KW, IC, OC]
    out_ptr,        # [N, OD, OH, OW, OC] NDHWC, pre-bias accumulator
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_S: tl.constexpr,   # input spatial tile
    BLOCK_OC: tl.constexpr,  # OC tile
    BLOCK_IC: tl.constexpr,  # IC tile
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    IDHW = ID * IH * IW
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < IDHW

    id_idx = s_offs // (IH * IW)
    rem = s_offs % (IH * IW)
    ih_idx = rem // IW
    iw_idx = rem % IW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load x [BLOCK_S, IC] - sum over IC done via dot
    # We accumulate input * weight for this (pid_n, pid_oc tile, pid_s tile)
    # produce [BLOCK_S, BLOCK_OC] for each (kd,kh,kw), then atomic_add to output

    in_base = ((pid_n * ID + id_idx) * IH + ih_idx) * IW + iw_idx  # [BLOCK_S]
    in_base = in_base * IC

    for kd in tl.static_range(0, KD):
        od = id_idx * STRIDE - PAD + kd
        od_ok = (od >= 0) & (od < OD)
        for kh in tl.static_range(0, KH):
            oh = ih_idx * STRIDE - PAD + kh
            oh_ok = (oh >= 0) & (oh < OH)
            for kw in tl.static_range(0, KW):
                ow = iw_idx * STRIDE - PAD + kw
                ow_ok = (ow >= 0) & (ow < OW)
                valid = od_ok & oh_ok & ow_ok & s_mask

                w_base = ((kd * KH + kh) * KW + kw) * IC * OC

                acc = tl.zeros((BLOCK_S, BLOCK_OC), dtype=tl.float32)
                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    x_ptrs = x_ptr + in_base[:, None] + ic_offs[None, :]
                    x_load_mask = s_mask[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                    w_ptrs = w_ptr + w_base + ic_offs[:, None] * OC + oc_offs[None, :]
                    w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals, allow_tf32=True)

                # scatter atomic add
                out_lin = ((pid_n * OD + od) * OH + oh) * OW + ow  # [BLOCK_S]
                out_ptrs = out_ptr + out_lin[:, None] * OC + oc_offs[None, :]
                store_mask = valid[:, None] & oc_mask[None, :]
                tl.atomic_add(out_ptrs, acc, mask=store_mask)


@triton.jit
def bias_clamp_div_kernel(
    out_ptr, b_ptr, n_elements, OC,
    MIN_VAL, INV_DIV,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(out_ptr + offsets, mask=mask, other=0.0)
    oc_idx = offsets % OC
    b = tl.load(b_ptr + oc_idx, mask=mask, other=0.0)
    x = x + b
    x = tl.where(x < MIN_VAL, MIN_VAL, x)
    x = x * INV_DIV
    tl.store(out_ptr + offsets, x, mask=mask)


def conv_transpose3d_fused(x, weight, bias, stride, padding, min_value, divisor):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape

    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    x_ndhwc = x.permute(0, 2, 3, 4, 1).contiguous()
    # weight [IC, OC, KD, KH, KW] -> [KD, KH, KW, IC, OC]
    w_layout = weight.permute(2, 3, 4, 0, 1).contiguous()

    out_ndhwc = torch.zeros((N, OD, OH, OW, OC), device=x.device, dtype=torch.float32)

    BLOCK_S = 32
    BLOCK_OC = 64
    BLOCK_IC = 32

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(ID * IH * IW, BLOCK_S))

    conv_transpose3d_scatter_kernel[grid](
        x_ndhwc, w_layout, out_ndhwc,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        stride, padding,
        BLOCK_S=BLOCK_S, BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC,
        num_warps=4, num_stages=2,
    )

    n_elements = out_ndhwc.numel()
    BLOCK = 1024
    grid2 = (triton.cdiv(n_elements, BLOCK),)
    bias_clamp_div_kernel[grid2](
        out_ndhwc, bias, n_elements, OC,
        float(min_value), 1.0 / float(divisor),
        BLOCK_SIZE=BLOCK, num_warps=4,
    )

    out = out_ndhwc.permute(0, 4, 1, 2, 3).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous().cuda()
        return conv_transpose3d_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride, self.padding,
            self.min_value, self.divisor,
        )