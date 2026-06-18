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

    oc_offs = tl.arange(0, OC)

    HW_IC = H * W * IC
    W_IC = W * IC

    acc = tl.zeros((OC, BLOCK_SP), dtype=tl.float32)

    KHW_IC = KH * KW * IC
    KW_IC = KW * IC

    ic_range = tl.arange(0, IC)

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

    min_val = tl.min(acc, axis=0)

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

        with torch.no_grad():
            w_ohwi = self.conv.weight.detach().permute(0, 2, 3, 1).contiguous()
        self.register_buffer('w_ohwi', w_ohwi.cuda(), persistent=False)
        self.register_buffer('b_cuda', self.conv.bias.detach().cuda().contiguous(), persistent=False)

    def forward(self, x):
        x = x.cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_SP = 128
        grid = (N, triton.cdiv(OH * OW, BLOCK_SP))

        conv_min_tanh_kernel[grid](
            x_nhwc, self.w_ohwi, self.b_cuda, out,
            N, H, W, OH, OW,
            IC, OC, KH, KW,
            BLOCK_SP=BLOCK_SP,
            num_warps=8, num_stages=4,
        )
        return out