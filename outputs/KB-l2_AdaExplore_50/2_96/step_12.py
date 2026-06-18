import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convt_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    POD, POH, POW,  # pooled dims
    stride: tl.constexpr, padding: tl.constexpr, KS: tl.constexpr,
    MP: tl.constexpr,  # maxpool kernel
    scale: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # one program per (n, oc) - emits one scalar
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    NEG_INF = -1.0e30
    sum_acc = 0.0
    pool_count = POD * POH * POW

    # Iterate over pooled output cells
    for pd in range(0, POD):
        for ph in range(0, POH):
            for pw in range(0, POW):
                # max over MP^3 window of conv_transpose output
                max_val = NEG_INF
                for md in range(0, MP):
                    od = pd * MP + md
                    for mh in range(0, MP):
                        oh = ph * MP + mh
                        for mw in range(0, MP):
                            ow = pw * MP + mw
                            # compute conv_transpose output[n, oc, od, oh, ow]
                            # = sum over ic, kd, kh, kw where:
                            #   id = (od + padding - kd) / stride, must be integer & in range
                            val = 0.0
                            if b_ptr is not None:
                                pass  # bias added at end
                            # iterate kernel positions
                            for kd in range(0, KS):
                                id_num = od + padding - kd
                                id_q = id_num // stride
                                id_r = id_num - id_q * stride
                                d_valid = (id_r == 0) & (id_q >= 0) & (id_q < ID)
                                for kh in range(0, KS):
                                    ih_num = oh + padding - kh
                                    ih_q = ih_num // stride
                                    ih_r = ih_num - ih_q * stride
                                    h_valid = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
                                    for kw in range(0, KS):
                                        iw_num = ow + padding - kw
                                        iw_q = iw_num // stride
                                        iw_r = iw_num - iw_q * stride
                                        w_valid = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                                        valid = d_valid & h_valid & w_valid
                                        # gather over ic
                                        ic_off = tl.arange(0, BLOCK_IC)
                                        ic_mask = (ic_off < IC) & valid
                                        # input offset: ((n*IC+ic)*ID+id_q)*IH*IW + ih_q*IW + iw_q
                                        x_off = (((n * IC) + ic_off) * ID + id_q) * IH * IW + ih_q * IW + iw_q
                                        # weight offset: ((ic*OC+oc)*KS+kd)*KS*KS + kh*KS + kw
                                        w_off = (((ic_off * OC) + oc) * KS + kd) * KS * KS + kh * KS + kw
                                        xv = tl.load(x_ptr + x_off, mask=ic_mask, other=0.0)
                                        wv = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)
                                        val += tl.sum(xv * wv, axis=0)
                            # add bias
                            bias = tl.load(b_ptr + oc)
                            val = val + bias
                            val = val * scale
                            max_val = tl.maximum(max_val, val)
                sum_acc += max_val

    mean = sum_acc / pool_count
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * OC + oc, mean)


@triton.jit
def fused_reduce_kernel(
    x_ptr, out_ptr,
    N, C, total,
    scale: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * total
    acc = 0.0
    for off in range(0, total, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < total
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        acc += tl.sum(v, axis=0)
    mean = acc / total
    mean = mean * scale
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + pid, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = float(scale)
        # Fold scale into conv weight & bias so the multiply disappears from forward.
        with torch.no_grad():
            self.conv_transpose.weight.mul_(self.scale)
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.mul_(self.scale)
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0
        self.clamp_max = 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        N, C, D, H, W = x.shape
        x = x.contiguous()
        total = D * H * W
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        if total <= 256:
            BLOCK = 256
        elif total <= 1024:
            BLOCK = 1024
        elif total <= 4096:
            BLOCK = 4096
        else:
            BLOCK = 4096
        grid = (N * C,)
        fused_reduce_kernel[grid](
            x, out, N, C, total,
            scale=1.0,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=2,
        )
        return out