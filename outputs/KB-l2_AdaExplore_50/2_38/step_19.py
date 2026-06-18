import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_fused_kernel(
    inp_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    POOL: tl.constexpr,
    PID: tl.constexpr, PIH: tl.constexpr, PIW: tl.constexpr,  # pre-pool input dims
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, od, oh, ow)
    pid_now = tl.program_id(0)  # n*OD + od
    pid_hw = tl.program_id(1)   # oh*OW + ow
    n = pid_now // OD
    od = pid_now % OD
    oh = pid_hw // OW
    ow = pid_hw % OW

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = bias

    # ConvT: od = id*STRIDE - PAD + kd  =>  id_q = (od+PAD-kd)/STRIDE
    inv_pool3 = 1.0 / (POOL * POOL * POOL)

    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_q = id_num // STRIDE
        id_r = id_num - id_q * STRIDE
        valid_d = (id_r == 0) & (id_q >= 0) & (id_q < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_q = ih_num // STRIDE
            ih_r = ih_num - ih_q * STRIDE
            valid_h = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_q = iw_num // STRIDE
                iw_r = iw_num - iw_q * STRIDE
                valid_w = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                valid = valid_d & valid_h & valid_w

                if valid:
                    # base into pre-pool input: id_q*POOL, ih_q*POOL, iw_q*POOL, with POOL window
                    pid_d_base = id_q * POOL
                    pih_base = ih_q * POOL
                    piw_base = iw_q * POOL
                    for ic in range(0, IC):
                        # average pool 2x2x2 on the fly
                        s = 0.0
                        for dd in tl.static_range(0, POOL):
                            for hh in tl.static_range(0, POOL):
                                for ww in tl.static_range(0, POOL):
                                    in_off = (((n * IC + ic) * PID + (pid_d_base + dd)) * PIH + (pih_base + hh)) * PIW + (piw_base + ww)
                                    s += tl.load(inp_ptr + in_off)
                        x_val = s * inv_pool3
                        # weight (IC, OC, KD, KH, KW)
                        w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                        acc += x_val * w_val

    # clamp
    acc = tl.minimum(tl.maximum(acc, clamp_min), clamp_max)

    # store: output layout (N, OC, OD, OH, OW)
    out_off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


@triton.jit
def softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    C, S,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    row_start = pid * S
    scale = tl.load(scale_ptr + c)

    max_val = -float('inf')
    sum_exp = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        cur_max = tl.max(v, axis=0)
        new_max = tl.maximum(max_val, cur_max)
        e = tl.exp(v - new_max)
        e = tl.where(mask, e, 0.0)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(e, axis=0)
        max_val = new_max

    inv = 1.0 / sum_exp
    scaled_inv = inv * scale

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        e = tl.exp(v - max_val) * scaled_inv
        tl.store(out_ptr + row_start + idx, e, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # Maintain conv_transpose so weights are initialized identically
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding,
                                                 output_padding=output_padding)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        # x: (N, IC, D, H, W) -- pre-pool dims
        N, IC, PID, PIH, PIW = x.shape
        POOL = self.pool_kernel_size
        ID = PID // POOL
        IH = PIH // POOL
        IW = PIW // POOL

        KD = KH = KW = self.kernel_size
        STRIDE = self.stride
        PAD = self.padding
        OP = self.output_padding
        OC = self.out_channels

        OD = (ID - 1) * STRIDE - 2 * PAD + KD + OP
        OH = (IH - 1) * STRIDE - 2 * PAD + KH + OP
        OW = (IW - 1) * STRIDE - 2 * PAD + KW + OP

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        bias = self.conv_transpose.bias.contiguous()      # (OC,)

        # Output buffer (pre-softmax). We write clamped values here.
        conv_out = torch.empty((N, OC, OD, OH, OW), dtype=x.dtype, device=x.device)

        BLOCK_OC = triton.next_power_of_2(OC)
        grid = (N * OD, OH * OW)
        conv_transpose_fused_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            STRIDE, PAD,
            POOL,
            PID, PIH, PIW,
            self.clamp_min, self.clamp_max,
            BLOCK_OC=BLOCK_OC, num_warps=4, num_stages=2,
        )

        # softmax over spatial dims with scale fused
        S = OD * OH * OW
        x_flat = conv_out.view(N * OC, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(-1).contiguous()

        BLOCK = 4096
        num_warps = 8

        softmax_scale_kernel[(N * OC,)](
            x_flat, scale_flat, out,
            OC, S,
            BLOCK=BLOCK, num_warps=num_warps, num_stages=3,
        )
        return out.view(N, OC, OD, OH, OW)