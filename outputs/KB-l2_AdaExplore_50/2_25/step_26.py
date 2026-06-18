import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H, W, OH, OW,
    IC: tl.constexpr, OC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < (OH * OW)
    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_offs = tl.arange(0, OC)  # [OC]

    # x is NHWC: stride: (H*W*IC, W*IC, IC, 1)
    # for each spatial location, load IC contiguous floats
    HW_IC = H * W * IC
    W_IC = W * IC

    # accumulator [OC, BLOCK_SP]
    acc = tl.zeros((OC, BLOCK_SP), dtype=tl.float32)

    # weight layout: (OC, KH, KW, IC) contiguous: stride (KH*KW*IC, KW*IC, IC, 1)
    KHW_IC = KH * KW * IC
    KW_IC = KW * IC

    ic_range = tl.arange(0, IC)  # [IC]

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # [BLOCK_SP]
            iw = ow + kw  # [BLOCK_SP]
            # x base offset for this (kh,kw) over BLOCK_SP positions
            x_base = pid_n * HW_IC + ih * W_IC + iw * IC  # [BLOCK_SP]
            # load IC channels for each sp: shape [BLOCK_SP, IC]
            x_off = x_base[:, None] + ic_range[None, :]
            x_vals = tl.load(x_ptr + x_off, mask=sp_mask[:, None], other=0.0)  # [BLOCK_SP, IC]

            # weight: [OC, IC] for this (kh,kw)
            w_base = oc_offs[:, None] * KHW_IC + kh * KW_IC + kw * IC + ic_range[None, :]  # [OC, IC]
            w_vals = tl.load(w_ptr + w_base)  # [OC, IC]

            # acc += w_vals @ x_vals.T  -> [OC, BLOCK_SP]
            acc += tl.dot(w_vals, tl.trans(x_vals))

    # bias
    b_val = tl.load(b_ptr + oc_offs)  # [OC]
    acc += b_val[:, None]

    # min across OC
    min_val = tl.min(acc, axis=0)  # [BLOCK_SP]

    # tanh twice
    y = tl.extra.cuda.libdevice.tanh(min_val)
    y = tl.extra.cuda.libdevice.tanh(y)

    out_off = pid_n * (OH * OW) + sp_offs
    tl.store(out_ptr + out_off, y, mask=sp_mask)


def conv_min_tanh_tanh(x_nhwc, weight_ohwi, bias, N, H, W, IC, OC, KH, KW):
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, 1, OH, OW), device=x_nhwc.device, dtype=x_nhwc.dtype)

    BLOCK_SP = 64
    grid = (N, triton.cdiv(OH * OW, BLOCK_SP))

    conv_min_tanh_kernel[grid](
        x_nhwc, weight_ohwi, bias, out,
        N, H, W, OH, OW,
        IC, OC, KH, KW,
        BLOCK_SP=BLOCK_SP,
        num_warps=4, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        # Convert weight from (OC, IC, KH, KW) to (OC, KH, KW, IC)
        w = self.conv.weight.cuda().permute(0, 2, 3, 1).contiguous()
        b = self.conv.bias.cuda().contiguous()

        return conv_min_tanh_tanh(x_nhwc, w, b, N, H, W, IC, OC, KH, KW)