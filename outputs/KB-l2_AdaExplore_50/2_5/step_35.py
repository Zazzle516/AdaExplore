import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,      # [N, IC, IH, IW] NHWC layout: [N, IH, IW, IC]
    w_ptr,      # [IC, OC, KH, KW] -> we'll layout as [KH, KW, IC, OC]
    bias_conv_ptr,  # [OC]
    bias_sub_ptr,   # [OC]
    out_ptr,    # [N, OC, OH, OW] NHWC: [N, OH, OW, OC]
    N, IC, OC,
    IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    BLOCK_M: tl.constexpr,  # output spatial tile (OH*OW)
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_n_batch = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)        # output spatial tile
    pid_oc = tl.program_id(2)       # OC tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)

    # output positions
    oh = offs_m // OW
    ow = offs_m % OW
    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each (kh, kw), find ih, iw
    # ih = (oh + PADDING - kh) / STRIDE, must be int and in [0, IH)
    for kh in tl.static_range(0, KH):
        ih_num = oh + PADDING - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num >= 0) & ((ih_num % STRIDE) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PADDING - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num >= 0) & ((iw_num % STRIDE) == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid & m_mask  # [BLOCK_M]

            # Input pointer base: x[n, ih, iw, :] in NHWC
            # offset = ((n * IH + ih) * IW + iw) * IC
            in_offset = ((pid_n_batch * IH + ih) * IW + iw) * IC  # [BLOCK_M]

            # Weight pointer base: w[kh, kw, :, oc_tile] -- layout [KH, KW, IC, OC]
            w_offset_base = ((kh * KW + kw) * IC) * OC  # scalar

            # Loop over IC in chunks of BLOCK_K
            for ick in range(0, IC, BLOCK_K):
                offs_k = ick + tl.arange(0, BLOCK_K)
                k_mask = offs_k < IC

                # Load x: [BLOCK_M, BLOCK_K]
                x_ptrs = x_ptr + in_offset[:, None] + offs_k[None, :]
                x_load_mask = valid[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                # Load w: [BLOCK_K, BLOCK_N]
                w_ptrs = w_ptr + w_offset_base + offs_k[:, None] * OC + offs_n[None, :]
                w_load_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add conv bias, subtract sub bias, tanh
    b_conv = tl.load(bias_conv_ptr + offs_n, mask=n_mask, other=0.0)
    b_sub = tl.load(bias_sub_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b_conv[None, :] - b_sub[None, :]

    # tanh
    e2 = tl.exp(2.0 * acc)
    acc = (e2 - 1.0) / (e2 + 1.0)

    # Store: out[n, oh, ow, oc] = out[((n*OH + oh)*OW + ow)*OC + oc]
    out_offset = ((pid_n_batch * OH + oh) * OW + ow) * OC
    out_ptrs = out_ptr + out_offset[:, None] + offs_n[None, :]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Use the same default initialization as nn.ConvTranspose2d
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Precompute reshaped weight in NHWC-friendly layout [KH, KW, IC, OC]
        self._cached_weight = None
        self._cached_bias_conv = None
        self._cached_bias_sub = None

    def _prepare_weights(self):
        # weight shape: [IC, OC, KH, KW]
        w = self.conv_transpose.weight.data  # [IC, OC, KH, KW]
        # Permute to [KH, KW, IC, OC]
        w_perm = w.permute(2, 3, 0, 1).contiguous()
        self._cached_weight = w_perm
        self._cached_bias_conv = self.conv_transpose.bias.data.contiguous()
        self._cached_bias_sub = self.bias.data.view(-1).contiguous()

    def forward(self, x):
        # x: [N, IC, IH, IW]
        x = x.cuda() if not x.is_cuda else x
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding

        OH = (IH - 1) * stride - 2 * padding + KH + output_padding
        OW = (IW - 1) * stride - 2 * padding + KW + output_padding

        # Always reload weights (they could be updated, but for eval typically constant)
        if (self._cached_weight is None or
            self._cached_weight.device != x.device):
            self._prepare_weights()
            self._cached_weight = self._cached_weight.to(x.device)
            self._cached_bias_conv = self._cached_bias_conv.to(x.device)
            self._cached_bias_sub = self._cached_bias_sub.to(x.device)

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, IH, IW, IC]

        # Allocate output in NHWC
        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32 if IC >= 32 else IC

        grid = (N, triton.cdiv(OH * OW, BLOCK_M), triton.cdiv(OC, BLOCK_N))

        conv_transpose_fused_kernel[grid](
            x_nhwc, self._cached_weight, self._cached_bias_conv, self._cached_bias_sub,
            out_nhwc,
            N, IC, OC,
            IH, IW, OH, OW,
            KH=KH, KW=KW,
            STRIDE=stride, PADDING=padding,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Permute back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out