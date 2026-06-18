import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'IC', 'OH', 'OW'],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    PARITY: tl.constexpr,  # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
):
    """
    Each program handles a tile of output for one of 4 parity classes.
    For ConvTranspose2d with stride=2, kernel=3, pad=1:
      - oh' = oh + PAD - kh, must be divisible by STRIDE=2
      - For each parity (oh%2, ow%2), only certain (kh, kw) pairs are valid.
    
    PAD=1, STRIDE=2: (oh+1-kh) % 2 == 0  =>  (oh - kh) % 2 == 1  =>  kh%2 != oh%2 (for STRIDE=2)
    Wait: (oh + 1 - kh) % 2 == 0 means oh + 1 - kh is even, i.e. (oh - kh) is odd, i.e. oh%2 != kh%2.
    
    For oh%2=0: kh must be odd => kh in {1} (out of 0,1,2)
    For oh%2=1: kh must be even => kh in {0, 2}
    
    So we have 1 or 2 valid kh per parity. Total: parity (0,0) => 1*1=1, (0,1)=>1*2=2, (1,0)=>2*1=2, (1,1)=>2*2=4 kh/kw pairs.
    
    We launch one grid covering only one parity class.
    """
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    # parity class
    par_h: tl.constexpr = PARITY // 2
    par_w: tl.constexpr = PARITY % 2

    # Subset of output positions with this parity
    # OH_par = OH // 2 if par==0 and OH even, else handle ceil
    OH_par = (OH + (1 - par_h)) // 2  # number of oh's with oh%2==par_h
    OW_par = (OW + (1 - par_w)) // 2

    SP_par = OH_par * OW_par

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < SP_par

    # decode sp_offs into (oh_idx, ow_idx) in parity subgrid
    oh_idx = sp_offs // OW_par
    ow_idx = sp_offs % OW_par

    # actual oh, ow
    oh = oh_idx * 2 + par_h
    ow = ow_idx * 2 + par_w

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    x_batch_off = pid_n * (IC * IH * IW)
    ic_offs = tl.arange(0, BLOCK_IC)

    # Iterate over valid (kh, kw) pairs based on parity
    # For STRIDE=2, PAD=1, KH=KW=3:
    #   kh values where (kh % 2) != par_h
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # parity check at compile time
            kh_par_ok: tl.constexpr = (kh % STRIDE) != par_h
            kw_par_ok: tl.constexpr = (kw % STRIDE) != par_w
            if kh_par_ok and kw_par_ok:
                ih_num = oh + PAD - kh
                iw_num = ow + PAD - kw
                ih = ih_num // STRIDE
                iw = iw_num // STRIDE
                valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & sp_mask

                x_addrs = x_batch_off + ic_offs[:, None] * (IH * IW) + ih[None, :] * IW + iw[None, :]
                x_tile = tl.load(x_ptr + x_addrs, mask=valid[None, :], other=0.0)

                w_addrs = ic_offs[None, :] * (OC * KH * KW) + oc_offs[:, None] * (KH * KW) + kh * KW + kw
                w_tile = tl.load(w_ptr + w_addrs, mask=oc_mask[:, None], other=0.0)

                acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * SCALE
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * INV_SCALE

    # Store: out[n, oc, oh, ow]
    out_addrs = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + oh[None, :] * OW + ow[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_addrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = (IH - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (IW - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        fused_bias = (self.conv_transpose.bias + self.bias.view(-1)).contiguous()
        weight = self.conv_transpose.weight.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_IC = max(16, triton.next_power_of_2(IC))

        for parity in range(4):
            par_h = parity // 2
            par_w = parity % 2
            OH_par = (OH + (1 - par_h)) // 2
            OW_par = (OW + (1 - par_w)) // 2
            SP_par = OH_par * OW_par

            grid = lambda meta, sp=SP_par: (
                triton.cdiv(sp, meta['BLOCK_SP']),
                triton.cdiv(OC, meta['BLOCK_OC']),
                N,
            )

            conv_transpose2d_kernel[grid](
                x, weight, fused_bias, out,
                N, IC, IH, IW,
                OC, OH, OW,
                KH, KW,
                self.stride, self.padding,
                float(self.scaling_factor),
                float(1.0 / self.scaling_factor),
                BLOCK_IC=BLOCK_IC,
                PARITY=parity,
            )
        return out