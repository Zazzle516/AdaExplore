import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gather_convt_pool_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    POD, POH, POW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    scale1, scale2,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, pd, ph, pw, oc_tile)
    pid = tl.program_id(0)
    oc_tile = tl.program_id(1)

    pw = pid % POW
    tmp = pid // POW
    ph = tmp % POH
    tmp = tmp // POH
    pd = tmp % POD
    n = tmp // POD

    oc_offs = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # The 2x2x2 output block corresponds to output coords:
    #  od in {pd*2, pd*2+1}, oh in {ph*2, ph*2+1}, ow in {pw*2, pw*2+1}
    od0 = pd * 2
    oh0 = ph * 2
    ow0 = pw * 2

    # Accumulator: sum over the 8 conv_out elements (we'll add conv_bias*8 once outside)
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # For each of the 8 output positions
    for dz in tl.static_range(0, 2):
        od = od0 + dz
        for dy in tl.static_range(0, 2):
            oh = oh0 + dy
            for dx in tl.static_range(0, 2):
                ow = ow0 + dx

                # For each kernel position, determine valid id, ih, iw
                # id*SD - PD + kd = od  =>  id = (od + PD - kd) / SD, need divisible
                for kd in tl.static_range(0, 3):
                    num_d = od + PD - kd
                    id_ = num_d // SD
                    valid_d = (num_d - id_ * SD == 0) & (id_ >= 0) & (id_ < ID)
                    for kh in tl.static_range(0, 3):
                        num_h = oh + PH - kh
                        ih = num_h // SH
                        valid_h = (num_h - ih * SH == 0) & (ih >= 0) & (ih < IH)
                        for kw in tl.static_range(0, 3):
                            num_w = ow + PW - kw
                            iw = num_w // SW
                            valid_w = (num_w - iw * SW == 0) & (iw >= 0) & (iw < IW)
                            valid = valid_d & valid_h & valid_w

                            if valid:
                                # loop over IC
                                for ic in range(0, IC):
                                    x_off = n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + id_ * (IH * IW) + ih * IW + iw
                                    x_val = tl.load(x_ptr + x_off)
                                    # weight: [IC, OC, KD, KH, KW]
                                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                    acc += x_val * w_val

    # acc now equals sum over 8 positions of (conv_out without bias). Add conv_bias * 8.
    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + cb * 8.0
    # avg pool divides by 8
    acc = acc * (1.0 / 8.0)
    # scale1
    acc = acc * scale1
    # bias add (per OC)
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b
    # scale2
    acc = acc * scale2

    # Store: out shape [N, OC, POD, POH, POW]
    out_off = n * (OC * POD * POH * POW) + oc_offs * (POD * POH * POW) + pd * (POH * POW) + ph * POW + pw
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.conv_bias = nn.Parameter(conv.bias.detach().clone())

        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OC = self.out_channels

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        POD = OD // 2
        POH = OH // 2
        POW = OW // 2

        out = torch.empty((N, OC, POD, POH, POW), device=x.device, dtype=torch.float32)

        BLOCK_OC = 16
        if BLOCK_OC < OC:
            # find next pow2 >= OC
            BLOCK_OC = triton.next_power_of_2(OC)

        num_oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC
        grid = (N * POD * POH * POW, num_oc_tiles)

        bias_flat = self.bias.view(-1).contiguous()

        gather_convt_pool_fused_kernel[grid](
            x, self.weight, self.conv_bias, bias_flat, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            POD, POH, POW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            float(self.scale1.item()), float(self.scale2.item()),
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )
        return out