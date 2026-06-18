import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 256}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'KH', 'KW', 'POOL', 'PH', 'PW', 'OC'],
)
@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H, W,
    OC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OH, OW,
    POOL: tl.constexpr, PH, PW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    BLOCK_OC: tl.constexpr, BLOCK_P: tl.constexpr,
):
    # program ids: (n, oc_block, p_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    P_total = PH * PW
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)      # [BLOCK_P]

    mask_oc = offs_oc < OC
    mask_p = offs_p < P_total

    # pooled output coords
    ph = offs_p // PW
    pw = offs_p % PW

    # accumulator [BLOCK_OC, BLOCK_P]
    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    # for each position in pool window
    for ih in tl.static_range(POOL):
        for iw in tl.static_range(POOL):
            # conv output coords
            oh = ph * POOL + ih  # [BLOCK_P]
            ow = pw * POOL + iw  # [BLOCK_P]

            # accumulate conv at this position into acc_pos [BLOCK_OC, BLOCK_P]
            acc_pos = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    ih_in = oh + kh  # [BLOCK_P]
                    iw_in = ow + kw  # [BLOCK_P]
                    # load weights [BLOCK_OC, IC]
                    w_offs = (offs_oc[:, None] * stride_wo +
                              tl.arange(0, IC)[None, :] * stride_wi +
                              kh * stride_wkh + kw * stride_wkw)
                    w_mask = mask_oc[:, None]
                    w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_OC, IC]

                    # load x [IC, BLOCK_P]
                    x_offs = (pid_n * stride_xn +
                              tl.arange(0, IC)[:, None] * stride_xc +
                              ih_in[None, :] * stride_xh +
                              iw_in[None, :] * stride_xw)
                    x_mask = mask_p[None, :]
                    x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [IC, BLOCK_P]

                    acc_pos += tl.dot(w_vals, x_vals)

            # add bias and accumulate into acc (avg over POOL*POOL)
            b = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
            acc_pos = acc_pos + b[:, None]
            acc += acc_pos

    # average pool: divide by POOL*POOL
    acc = acc / (POOL * POOL)

    # sigmoid
    acc = tl.sigmoid(acc)

    # mask out invalid positions before reduction
    valid = mask_oc[:, None] & mask_p[None, :]
    acc = tl.where(valid, acc, 0.0)

    # sum over BLOCK_OC and BLOCK_P, then atomic add to out[pid_n]
    partial = tl.sum(tl.sum(acc, axis=1), axis=0)
    tl.atomic_add(out_ptr + pid_n, partial)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.cuda().contiguous()
        b = self.conv.bias.cuda().contiguous()

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(PH * PW, meta['BLOCK_P']),
        )

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            POOL, PH, PW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        )
        return out