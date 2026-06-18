import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convT_pool_kernel(
    x_ptr,           # (N, IC, ID, IH, IW)
    w_ptr,           # (IC, OC, KD, KH, KW)
    b_ptr,           # (OC,)
    out_ptr,         # (N, OC)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,
    scale, inv_count,
    BLOCK_PHW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    bias = tl.load(b_ptr + oc)

    PHW = PH * PW

    acc = 0.0

    for pd in range(0, PD):
        od0 = 2 * pd
        od1 = od0 + 1

        for phw_start in range(0, PHW, BLOCK_PHW):
            phw_offs = phw_start + tl.arange(0, BLOCK_PHW)
            phw_mask = phw_offs < PHW
            ph_offs = phw_offs // PW
            pw_offs = phw_offs % PW

            oh0 = 2 * ph_offs
            oh1 = oh0 + 1
            ow0 = 2 * pw_offs
            ow1 = ow0 + 1

            v000 = tl.zeros((BLOCK_PHW,), dtype=tl.float32)
            v001 = tl.zeros((BLOCK_PHW,), dtype=tl.float32)
            v010 = tl.zeros((BLOCK_PHW,), dtype=tl.float32)
            v011 = tl.zeros((BLOCK_PHW,), dtype=tl.float32)
            v100 = tl.zeros((BLOCK_PHW,), dtype=tl.float32)
            v101 = tl.zeros((BLOCK_PHW,), dtype=tl.float32)
            v110 = tl.zeros((BLOCK_PHW,), dtype=tl.float32)
            v111 = tl.zeros((BLOCK_PHW,), dtype=tl.float32)

            x_n_base = n * IC * ID * IH * IW

            for ic in range(0, IC):
                x_base = x_n_base + ic * ID * IH * IW
                w_base = ic * OC * KD * KH * KW + oc * KD * KH * KW

                for kd in tl.static_range(0, KD):
                    num_d0 = od0 + PAD - kd
                    num_d1 = od1 + PAD - kd
                    id0 = num_d0 // STRIDE
                    id1 = num_d1 // STRIDE
                    valid_d0 = (num_d0 >= 0) & ((num_d0 % STRIDE) == 0) & (id0 >= 0) & (id0 < ID)
                    valid_d1 = (num_d1 >= 0) & ((num_d1 % STRIDE) == 0) & (id1 >= 0) & (id1 < ID)

                    for kh in tl.static_range(0, KH):
                        num_h0 = oh0 + PAD - kh
                        num_h1 = oh1 + PAD - kh
                        ih0 = num_h0 // STRIDE
                        ih1 = num_h1 // STRIDE
                        valid_h0 = (num_h0 >= 0) & ((num_h0 % STRIDE) == 0) & (ih0 >= 0) & (ih0 < IH)
                        valid_h1 = (num_h1 >= 0) & ((num_h1 % STRIDE) == 0) & (ih1 >= 0) & (ih1 < IH)

                        for kw in tl.static_range(0, KW):
                            num_w0 = ow0 + PAD - kw
                            num_w1 = ow1 + PAD - kw
                            iw0 = num_w0 // STRIDE
                            iw1 = num_w1 // STRIDE
                            valid_w0 = (num_w0 >= 0) & ((num_w0 % STRIDE) == 0) & (iw0 >= 0) & (iw0 < IW)
                            valid_w1 = (num_w1 >= 0) & ((num_w1 % STRIDE) == 0) & (iw1 >= 0) & (iw1 < IW)

                            wval = tl.load(w_ptr + w_base + kd * KH * KW + kh * KW + kw)

                            if valid_d0:
                                idx000 = x_base + id0 * IH * IW + (ih0 * IW + iw0)
                                m000 = valid_h0 & valid_w0
                                x000 = tl.load(x_ptr + idx000, mask=m000, other=0.0)
                                v000 += x000 * wval

                                idx001 = x_base + id0 * IH * IW + (ih0 * IW + iw1)
                                m001 = valid_h0 & valid_w1
                                x001 = tl.load(x_ptr + idx001, mask=m001, other=0.0)
                                v001 += x001 * wval

                                idx010 = x_base + id0 * IH * IW + (ih1 * IW + iw0)
                                m010 = valid_h1 & valid_w0
                                x010 = tl.load(x_ptr + idx010, mask=m010, other=0.0)
                                v010 += x010 * wval

                                idx011 = x_base + id0 * IH * IW + (ih1 * IW + iw1)
                                m011 = valid_h1 & valid_w1
                                x011 = tl.load(x_ptr + idx011, mask=m011, other=0.0)
                                v011 += x011 * wval

                            if valid_d1:
                                idx100 = x_base + id1 * IH * IW + (ih0 * IW + iw0)
                                m100 = valid_h0 & valid_w0
                                x100 = tl.load(x_ptr + idx100, mask=m100, other=0.0)
                                v100 += x100 * wval

                                idx101 = x_base + id1 * IH * IW + (ih0 * IW + iw1)
                                m101 = valid_h0 & valid_w1
                                x101 = tl.load(x_ptr + idx101, mask=m101, other=0.0)
                                v101 += x101 * wval

                                idx110 = x_base + id1 * IH * IW + (ih1 * IW + iw0)
                                m110 = valid_h1 & valid_w0
                                x110 = tl.load(x_ptr + idx110, mask=m110, other=0.0)
                                v110 += x110 * wval

                                idx111 = x_base + id1 * IH * IW + (ih1 * IW + iw1)
                                m111 = valid_h1 & valid_w1
                                x111 = tl.load(x_ptr + idx111, mask=m111, other=0.0)
                                v111 += x111 * wval

            v000 = (v000 + bias) * scale
            v001 = (v001 + bias) * scale
            v010 = (v010 + bias) * scale
            v011 = (v011 + bias) * scale
            v100 = (v100 + bias) * scale
            v101 = (v101 + bias) * scale
            v110 = (v110 + bias) * scale
            v111 = (v111 + bias) * scale

            m1 = tl.maximum(tl.maximum(v000, v001), tl.maximum(v010, v011))
            m2 = tl.maximum(tl.maximum(v100, v101), tl.maximum(v110, v111))
            m = tl.maximum(m1, m2)

            m = tl.where(phw_mask, m, 0.0)
            acc += tl.sum(m)

    mean = acc * inv_count
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * OC + oc, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1
        self.maxpool_kernel_size = maxpool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        stride = self.stride
        padding = self.padding

        OD = (ID - 1) * stride - 2 * padding + KD
        OH = (IH - 1) * stride - 2 * padding + KH
        OW = (IW - 1) * stride - 2 * padding + KW

        k = self.maxpool_kernel_size

        if k == 2 and stride == 2 and padding == 1 and KD == 3:
            PD, PH, PW = OD // 2, OH // 2, OW // 2
            pooled_total = PD * PH * PW
            inv_count = 1.0 / float(pooled_total)

            x = x.contiguous()
            w = self.conv_transpose.weight.contiguous()
            b = self.conv_transpose.bias.contiguous()

            out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)

            grid = (N * OC,)

            PHW = PH * PW
            BLOCK_PHW = 256
            if PHW <= 64:
                BLOCK_PHW = 64
            elif PHW <= 128:
                BLOCK_PHW = 128
            elif PHW <= 256:
                BLOCK_PHW = 256
            else:
                BLOCK_PHW = 256

            fused_convT_pool_kernel[grid](
                x, w, b, out,
                N, IC, ID, IH, IW,
                OC, OD, OH, OW,
                PD, PH, PW,
                float(self.scale), inv_count,
                BLOCK_PHW=BLOCK_PHW,
                KD=KD, KH=KH, KW=KW,
                STRIDE=stride, PAD=padding,
                num_warps=4,
                num_stages=2,
            )
            return out
        else:
            x = self.conv_transpose(x)
            x = x * self.scale
            x = self.maxpool(x)
            x = self.global_avg_pool(x)
            x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
            return x