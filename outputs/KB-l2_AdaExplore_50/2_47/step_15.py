import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_mish_tanh_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    # x strides (NDHWC)
    sx_n, sx_d, sx_h, sx_w,  # channel stride is 1
    # w strides (OC, KD, KH, KW, IC), inner is 1
    sw_oc, sw_kd, sw_kh, sw_kw,
    # out strides (NDHWC)
    so_n, so_d, so_h, so_w,
    BLOCK_OC: tl.constexpr, BLOCK_OW: tl.constexpr,
):
    pid_oc = tl.program_id(0)
    pid_sp = tl.program_id(1)  # over OD*OH*(OW/BLOCK_OW)
    pid_n = tl.program_id(2)

    num_ow_tiles = (OW + BLOCK_OW - 1) // BLOCK_OW
    ow_tile = pid_sp % num_ow_tiles
    tmp = pid_sp // num_ow_tiles
    oh = tmp % OH
    od = tmp // OH

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ow = ow_tile * BLOCK_OW + tl.arange(0, BLOCK_OW)
    mask_oc = offs_oc < OC
    mask_ow = offs_ow < OW

    offs_ic = tl.arange(0, IC)  # IC is constexpr, contiguous

    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

    # base pointer for this batch
    x_base = x_ptr + pid_n * sx_n

    for kd in tl.static_range(0, KD):
        id_ = od + kd
        for kh in tl.static_range(0, KH):
            ih = oh + kh
            for kw in tl.static_range(0, KW):
                iw = offs_ow + kw  # [BLOCK_OW]

                # Input: [BLOCK_OW, IC]
                x_off = (id_ * sx_d + ih * sx_h +
                         iw[:, None] * sx_w + offs_ic[None, :])
                x_mask = mask_ow[:, None]
                x_vals = tl.load(x_base + x_off, mask=x_mask, other=0.0)  # [BLOCK_OW, IC]

                # Weight: [OC, IC] at (kd, kh, kw)
                w_off = (offs_oc[:, None] * sw_oc +
                         kd * sw_kd + kh * sw_kh + kw * sw_kw +
                         offs_ic[None, :])
                w_mask = mask_oc[:, None]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, IC]

                # acc[BLOCK_OC, BLOCK_OW] += w_vals @ x_vals.T
                acc += tl.dot(w_vals, tl.trans(x_vals), allow_tf32=True)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[:, None]

    # mish + tanh fused
    # softplus stable
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    t1 = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    m = acc * t1
    out_val = 2.0 * tl.sigmoid(2.0 * m) - 1.0

    # Store to NDHWC output: [BLOCK_OC, BLOCK_OW] -> transpose to [BLOCK_OW, BLOCK_OC]
    out_t = tl.trans(out_val)  # [BLOCK_OW, BLOCK_OC]
    out_off = (pid_n * so_n + od * so_d + oh * so_h +
               offs_ow[:, None] * so_w + offs_oc[None, :])
    out_mask = mask_ow[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, out_t, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        assert stride == 1 and padding == 0
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.KD = self.KH = self.KW = kernel_size
        else:
            self.KD, self.KH, self.KW = kernel_size

        # Pre-transpose weight to [OC, KD, KH, KW, IC] (channels-last for input channels)
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KD, KH, KW]
            w_nhwc = w.permute(0, 2, 3, 4, 1).contiguous()  # [OC, KD, KH, KW, IC]
            self.register_buffer('weight_nhwc', w_nhwc)
            self.register_buffer('bias_buf', self.conv.bias.detach().contiguous())

    def forward(self, x):
        # x: [N, IC, D, H, W] -> permute to NDHWC
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.KD, self.KH, self.KW
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()  # [N, D, H, W, IC]
        out_nhwc = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

        sx_n = x_nhwc.stride(0)
        sx_d = x_nhwc.stride(1)
        sx_h = x_nhwc.stride(2)
        sx_w = x_nhwc.stride(3)

        w = self.weight_nhwc
        sw_oc = w.stride(0)
        sw_kd = w.stride(1)
        sw_kh = w.stride(2)
        sw_kw = w.stride(3)

        so_n = out_nhwc.stride(0)
        so_d = out_nhwc.stride(1)
        so_h = out_nhwc.stride(2)
        so_w = out_nhwc.stride(3)

        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_OC']),
            OD * OH * triton.cdiv(OW, meta['BLOCK_OW']),
            N,
        )

        conv3d_mish_tanh_nhwc_kernel[grid](
            x_nhwc, w, self.bias_buf, out_nhwc,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            sx_n, sx_d, sx_h, sx_w,
            sw_oc, sw_kd, sw_kh, sw_kw,
            so_n, so_d, so_h, so_w,
        )

        # Permute back to NCDHW
        out = out_nhwc.permute(0, 4, 1, 2, 3).contiguous()
        return out