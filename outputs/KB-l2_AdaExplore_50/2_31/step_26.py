import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_nhwc_fused_kernel(
    x_ptr,        # NHWC: (N, H, W, IC)
    w_ptr,        # (KH*KW*IC, OC)
    b_ptr,        # (OC,) conv bias
    bias_ptr,     # (OC,) extra bias
    out_ptr,      # NHWC: (N, OH, OW, OC)
    N, H, W, IC,
    OC, OH, OW,
    KH, KW,
    OUT_HW,
    constant_value, scaling_factor,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_oc = tl.program_id(0)
    pid_sp = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_sp = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    offs_ic = tl.arange(0, BLOCK_IC)

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < OUT_HW

    oh = offs_sp // OW
    ow = offs_sp % OW

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Loop over kh, kw, then over ic chunks
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # no padding
            iw = ow + kw
            # input position offset (in NHWC)
            base_x = pid_b * H * W * IC + ih * W * IC + iw * IC  # [BLOCK_SP]
            # weight offset row base = (kh*KW + kw) * IC
            kw_kh_base = (kh * KW + kw) * IC

            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + offs_ic
                mask_ic = ic_idx < IC

                # Load x tile: shape [BLOCK_SP, BLOCK_IC]
                x_off = base_x[:, None] + ic_idx[None, :]
                x_mask = mask_sp[:, None] & mask_ic[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # Load w tile: rows = kw_kh_base + ic_idx, cols = offs_oc
                # weight layout: (KH*KW*IC, OC), row stride = OC
                w_row = kw_kh_base + ic_idx
                w_off = w_row[:, None] * OC + offs_oc[None, :]
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals, allow_tf32=True, out_dtype=tl.float32)

    # epilogue
    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    bias_vals = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[None, :]
    acc = tl.minimum(acc, constant_value)
    acc = acc + bias_vals[None, :]
    acc = acc * scaling_factor

    # store NHWC: (N, OH, OW, OC)
    out_off = (pid_b * OH * OW * OC
               + offs_sp[:, None] * OC
               + offs_oc[None, :])
    out_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask, eviction_policy='evict_first')


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-pack weights to (KH*KW*IC, OC) once.
        # conv weight shape: (OC, IC, KH, KW)
        with torch.no_grad():
            w = self.conv.weight.detach().clone()  # (OC, IC, KH, KW)
            OC, IC, KH, KW = w.shape
            # Permute to (KH, KW, IC, OC) then reshape to (KH*KW*IC, OC)
            w_packed = w.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)
        self.register_buffer('w_packed', w_packed, persistent=False)

    def forward(self, x):
        # x: (N, IC, H, W)
        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW

        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, IC)

        # Refresh w_packed if weights changed (e.g., training). Cheap if same.
        if self.training or self.w_packed.shape != (KH * KW * IC, OC):
            with torch.no_grad():
                w_packed = self.conv.weight.detach().permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)
            w_packed_local = w_packed
        else:
            w_packed_local = self.w_packed

        b = self.conv.bias.contiguous()
        bias = self.bias.contiguous().view(-1)

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            triton.cdiv(OC, META['BLOCK_OC']),
            triton.cdiv(OUT_HW, META['BLOCK_SP']),
            N,
        )

        conv2d_nhwc_fused_kernel[grid](
            x_nhwc, w_packed_local, b, bias, out_nhwc,
            N, H, W, IC,
            OC, OH, OW,
            KH, KW,
            OUT_HW,
            float(self.constant_value), float(self.scaling_factor),
        )

        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out