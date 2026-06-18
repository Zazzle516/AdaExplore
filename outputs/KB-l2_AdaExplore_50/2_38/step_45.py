import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_fused_kernel(
    inp_ptr, weight_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    POOL: tl.constexpr,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, od, oh, ow_tile? no - single voxel)
    # actually: one program per (n, oc_tile, output_voxel)
    pid = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    n = tl.program_id(2)

    total_spatial = OD * OH * OW
    if pid >= total_spatial:
        return

    od = pid // (OH * OW)
    rem = pid % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = oc_block_id * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Initialize with bias
    acc = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)

    # for each kernel position, find input voxel that contributes
    # input idx i satisfies: i*SD - PD + kd = od  =>  i = (od + PD - kd) / SD
    for kd in tl.static_range(0, KD):
        num_d = od + PD - kd
        id_in = num_d // SD
        valid_d = (num_d >= 0) & (num_d % SD == 0) & (id_in >= 0) & (id_in < ID)
        for kh in tl.static_range(0, KH):
            num_h = oh + PH - kh
            ih_in = num_h // SH
            valid_h = (num_h >= 0) & (num_h % SH == 0) & (ih_in >= 0) & (ih_in < IH)
            for kw in tl.static_range(0, KW):
                num_w = ow + PW - kw
                iw_in = num_w // SW
                valid_w = (num_w >= 0) & (num_w % SW == 0) & (iw_in >= 0) & (iw_in < IW)
                valid = valid_d & valid_h & valid_w

                if valid:
                    # Now sum over IC: out += sum_ic input[n,ic,id,ih,iw_pool_avg] * weight[ic,oc,kd,kh,kw]
                    # input is pooled with AvgPool3d(2). Pooled shape: (ID,IH,IW). Original shape: (ID*2, IH*2, IW*2).
                    # pooled[n,ic,id_in,ih_in,iw_in] = mean of 2x2x2 block in original
                    # Original input pointer base for this voxel:
                    od_orig = id_in * POOL
                    oh_orig = ih_in * POOL
                    ow_orig = iw_in * POOL
                    orig_D = ID * POOL
                    orig_H = IH * POOL
                    orig_W = IW * POOL

                    for ic in range(0, IC):
                        # Sum the 2x2x2 pool block
                        pool_sum = 0.0
                        for pd in tl.static_range(0, POOL):
                            for ph in tl.static_range(0, POOL):
                                for pw in tl.static_range(0, POOL):
                                    in_off = (((n * IC + ic) * orig_D + (od_orig + pd)) * orig_H + (oh_orig + ph)) * orig_W + (ow_orig + pw)
                                    pool_sum += tl.load(inp_ptr + in_off)
                        inv_pool = 1.0 / (POOL * POOL * POOL)
                        pooled_val = pool_sum * inv_pool

                        w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(weight_ptr + w_off, mask=oc_mask, other=0.0)
                        acc += pooled_val * w_val

    # Clamp
    acc = tl.minimum(tl.maximum(acc, clamp_min), clamp_max)

    out_off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


@triton.jit
def softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    B, C, S,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    row_start = (b * C + c) * S
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

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        e = tl.exp(v - max_val) * inv * scale
        tl.store(out_ptr + row_start + idx, e, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.pool_kernel_size = pool_kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # Standard path: use torch's optimized conv_transpose3d
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)

        b, c, d, h, w = x.shape
        S = d * h * w
        x_flat = x.contiguous().view(b * c, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(-1).contiguous()

        if S >= 4096:
            BLOCK = 4096
            num_warps = 8
        elif S >= 2048:
            BLOCK = 2048
            num_warps = 8
        elif S >= 1024:
            BLOCK = 1024
            num_warps = 8
        else:
            BLOCK = 512
            num_warps = 4

        grid = (b * c,)
        softmax_scale_kernel[grid](
            x_flat, scale_flat, out,
            b, c, S,
            BLOCK=BLOCK, num_warps=num_warps, num_stages=2,
        )
        return out.view(b, c, d, h, w)