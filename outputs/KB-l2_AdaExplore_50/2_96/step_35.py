import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _convt_maxpool_gap_kernel(
    x_ptr,           # input: (N, IC, ID, IH, IW)
    w_ptr,           # weight: (IC, OC, KD, KH, KW)
    b_ptr,           # bias: (OC,)
    out_ptr,         # output: (N, OC)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,  # ConvT output shape
    PD, PH, PW,      # pooled shape (OD//2, OH//2, OW//2)
    INV_S: tl.constexpr,    # 1/(PD*PH*PW)
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    MK: tl.constexpr,       # maxpool kernel size = 2
    BLOCK_IC: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    # Loop over pooled output positions, compute conv-transpose value at each
    # output position on the fly, max over 2x2x2 window, accumulate sum.
    acc_sum = 0.0

    bias_val = tl.load(b_ptr + oc)

    # Iterate over pooled (PD, PH, PW)
    for pd in range(0, PD):
        for ph in range(0, PH):
            for pw in range(0, PW):
                # 2x2x2 window in conv-transpose output
                max_val = -1.0e30
                for dd in range(0, MK):
                    for dh in range(0, MK):
                        for dw in range(0, MK):
                            od = pd * MK + dd
                            oh = ph * MK + dh
                            ow = pw * MK + dw

                            # Compute one ConvTranspose3d output value at (n, oc, od, oh, ow):
                            # y = sum_{ic, kd, kh, kw} x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
                            # where: id*stride - pad + kd = od  =>  id = (od + pad - kd)/stride if divisible
                            val = bias_val

                            for kd in range(0, KD):
                                num_d = od + PAD - kd
                                id_ = num_d // STRIDE
                                valid_d = (num_d - id_ * STRIDE == 0) & (id_ >= 0) & (id_ < ID)
                                for kh in range(0, KH):
                                    num_h = oh + PAD - kh
                                    ih_ = num_h // STRIDE
                                    valid_h = (num_h - ih_ * STRIDE == 0) & (ih_ >= 0) & (ih_ < IH)
                                    for kw in range(0, KW):
                                        num_w = ow + PAD - kw
                                        iw_ = num_w // STRIDE
                                        valid_w = (num_w - iw_ * STRIDE == 0) & (iw_ >= 0) & (iw_ < IW)
                                        valid = valid_d & valid_h & valid_w

                                        # Vectorize over IC
                                        ic_offs = tl.arange(0, BLOCK_IC)
                                        ic_mask = ic_offs < IC

                                        # x[n, ic, id_, ih_, iw_]
                                        x_off = (((n * IC + ic_offs) * ID + id_) * IH + ih_) * IW + iw_
                                        x_vals = tl.load(x_ptr + x_off, mask=ic_mask & valid, other=0.0)

                                        # w[ic, oc, kd, kh, kw]
                                        w_off = (((ic_offs * OC + oc) * KD + kd) * KH + kh) * KW + kw
                                        w_vals = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)

                                        val += tl.sum(x_vals * w_vals, axis=0)

                            max_val = tl.maximum(max_val, val)

                acc_sum += max_val

    mean = acc_sum * INV_S
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * OC + oc, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Fold scale into weight & bias at init
        with torch.no_grad():
            self.conv_transpose.weight.mul_(scale)
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.mul_(scale)

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()  # (IC, OC, KD, KH, KW)
        bias = self.conv_transpose.bias.contiguous().cuda()       # (OC,)

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        S = self.stride
        P = self.padding

        OD = (ID - 1) * S - 2 * P + KD
        OH = (IH - 1) * S - 2 * P + KH
        OW = (IW - 1) * S - 2 * P + KW

        MK = self.maxpool_kernel_size
        PD = OD // MK
        PH = OH // MK
        PW = OW // MK

        out = torch.empty(N, OC, 1, 1, 1, device=x.device, dtype=x.dtype)

        # BLOCK_IC must be power-of-2 >= IC
        BLOCK_IC = 1
        while BLOCK_IC < IC:
            BLOCK_IC *= 2

        inv_s = 1.0 / (PD * PH * PW)

        grid = (N * OC,)
        _convt_maxpool_gap_kernel[grid](
            x, weight, bias,
            out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            INV_S=inv_s,
            KD=KD, KH=KH, KW=KW,
            STRIDE=S,
            PAD=P,
            MK=MK,
            BLOCK_IC=BLOCK_IC,
            num_warps=2,
        )
        return out