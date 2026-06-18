import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _get_autotune_configs():
    configs = []
    for bm in [32, 64, 128]:
        for bn in [32, 64, 128]:
            for bk in [16, 32]:
                for nw in [4, 8]:
                    for ns in [2, 3, 4]:
                        configs.append(
                            triton.Config(
                                {'BLOCK_OC': bm, 'BLOCK_S': bn, 'BLOCK_IC': bk},
                                num_warps=nw, num_stages=ns,
                            )
                        )
    return configs


@triton.autotune(configs=_get_autotune_configs(), key=['OC', 'OD', 'OH', 'OW', 'IC'])
@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_S: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_ic = tl.arange(0, BLOCK_IC)

    OHW = OH * OW
    OS = OD * OHW

    od = offs_s // OHW
    rem = offs_s - od * OHW
    oh = rem // OW
    ow = rem - oh * OW

    s_mask = offs_s < OS
    oc_mask = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_S), dtype=tl.float32)

    x_n_base = pid_n * IC * ID * IH * IW
    IDHW = ID * IH * IW
    IHW = IH * IW

    for kd in tl.static_range(KD):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                num_d = od + PAD - kd
                num_h = oh + PAD - kh
                num_w = ow + PAD - kw

                id_ = num_d // STRIDE
                ih_ = num_h // STRIDE
                iw_ = num_w // STRIDE

                valid_d = (num_d % STRIDE == 0) & (id_ >= 0) & (id_ < ID)
                valid_h = (num_h % STRIDE == 0) & (ih_ >= 0) & (ih_ < IH)
                valid_w = (num_w % STRIDE == 0) & (iw_ >= 0) & (iw_ < IW)
                valid = valid_d & valid_h & valid_w & s_mask

                in_spatial = id_ * IHW + ih_ * IW + iw_

                w_kpos_base = ((kd * KH + kh) * KW + kw) * IC * OC

                for ic_start in range(0, IC, BLOCK_IC):
                    cur_ic = ic_start + offs_ic
                    ic_mask = cur_ic < IC

                    in_ptrs = (x_ptr
                               + x_n_base
                               + cur_ic[:, None] * IDHW
                               + in_spatial[None, :])
                    in_vals = tl.load(in_ptrs,
                                      mask=ic_mask[:, None] & valid[None, :],
                                      other=0.0)

                    w_ptrs = (w_ptr
                              + w_kpos_base
                              + cur_ic[:, None] * OC
                              + offs_oc[None, :])
                    w_vals = tl.load(w_ptrs,
                                     mask=ic_mask[:, None] & oc_mask[None, :],
                                     other=0.0)

                    acc += tl.dot(tl.trans(w_vals), in_vals)

    cb = tl.load(conv_bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    eb = tl.load(extra_bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    x = acc + cb[:, None]
    bias_total = eb[:, None]
    res = (2.0 * x + bias_total) * x + x

    out_base = (pid_n * OC * OS
                + offs_oc[:, None] * OS
                + offs_s[None, :])
    out_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr + out_base, res, mask=out_mask)


def conv_transpose3d_fused(x, weight_permuted, conv_bias, extra_bias,
                           stride, padding, output_padding,
                           IC, OC, KD, KH, KW):
    x = x.contiguous()
    N, _, ID, IH, IW = x.shape

    OD = (ID - 1) * stride - 2 * padding + KD + output_padding
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    OS = OD * OH * OW

    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OS, META['BLOCK_S']))

    conv_transpose3d_fused_kernel[grid](
        x, weight_permuted, conv_bias, extra_bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride, padding,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size

        self._cached_w_perm = None
        self._cached_w_version = None

    def _get_weight_perm(self):
        w = self.conv_transpose.weight
        if (self._cached_w_perm is None
                or self._cached_w_version != w._version):
            self._cached_w_perm = w.detach().permute(2, 3, 4, 0, 1).contiguous()
            self._cached_w_version = w._version
        return self._cached_w_perm

    def forward(self, x):
        w_perm = self._get_weight_perm()
        return conv_transpose3d_fused(
            x,
            w_perm,
            self.conv_transpose.bias,
            self.bias.view(-1),
            self.stride, self.padding, self.output_padding,
            self.in_channels, self.out_channels,
            self.kd, self.kh, self.kw,
        )