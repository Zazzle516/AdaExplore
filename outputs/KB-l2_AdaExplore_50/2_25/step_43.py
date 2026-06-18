import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 256, 'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 256, 'BLOCK_OC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256, 'BLOCK_OC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H, W, OH, OW,
    IC: tl.constexpr, OC: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < (OH * OW)
    oh = sp_offs // OW
    ow = sp_offs % OW

    HW_IC = H * W * IC
    W_IC = W * IC

    KHW_IC = KH * KW * IC
    KW_IC = KW * IC

    ic_range = tl.arange(0, IC)

    # Running min over OC, shape [BLOCK_SP]
    min_val = tl.full((BLOCK_SP,), float('inf'), dtype=tl.float32)

    for oc_tile in tl.static_range(0, OC, BLOCK_OC):
        oc_offs = oc_tile + tl.arange(0, BLOCK_OC)
        acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_base = pid_n * HW_IC + ih * W_IC + iw * IC
                x_off = x_base[:, None] + ic_range[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=sp_mask[:, None], other=0.0)

                w_base = oc_offs[:, None] * KHW_IC + kh * KW_IC + kw * IC + ic_range[None, :]
                w_vals = tl.load(w_ptr + w_base)

                acc += tl.dot(w_vals, tl.trans(x_vals))

        b_val = tl.load(b_ptr + oc_offs)
        acc += b_val[:, None]

        tile_min = tl.min(acc, axis=0)
        min_val = tl.minimum(min_val, tile_min)

    y = tl.extra.cuda.libdevice.tanh(min_val)
    y = tl.extra.cuda.libdevice.tanh(y)

    out_off = pid_n * (OH * OW) + sp_offs
    tl.store(out_ptr + out_off, y, mask=sp_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to OHWI layout and store as a buffer
        w = self.conv.weight.detach().cuda().permute(0, 2, 3, 1).contiguous()
        b = self.conv.bias.detach().cuda().contiguous()
        self.register_buffer('weight_ohwi', w)
        self.register_buffer('bias_buf', b)

    def forward(self, x):
        x = x.cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        # NCHW -> NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda META: (N, triton.cdiv(OH * OW, META['BLOCK_SP']))

        conv_min_tanh_kernel[grid](
            x_nhwc, self.weight_ohwi, self.bias_buf, out,
            N, H, W, OH, OW,
            IC, OC, KH, KW,
        )
        return out