import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_htanh_mean_tanh_kernel(
    x_ptr,       # [N, IC, H, W]
    w_ptr,       # [OC, IC, 3, 3] -- already prepared as equivalent conv weight
    b_ptr,       # [OC]
    out_ptr,     # [N, OC]
    N, IC, H, W,
    H_out, W_out,  # pooled dims = H/2, W/2
    htanh_min, htanh_max,
    inv_area,
    BLOCK_IC: tl.constexpr,
    BLOCK_SP: tl.constexpr,  # number of pooled output positions per iteration
):
    # one program per (n, oc)
    pid = tl.program_id(0)
    n = pid // tl.num_programs(1)  # not used; use direct mapping
    # use 2D grid actually
    pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'H', 'W'],
)
@triton.jit
def fused_kernel(
    x_ptr,       # [N, IC, H, W]
    w_ptr,       # [OC, IC, 3, 3]
    b_ptr,       # [OC]
    out_ptr,     # [N, OC]
    N, IC: tl.constexpr, OC, H: tl.constexpr, W: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    htanh_min: tl.constexpr, htanh_max: tl.constexpr,
    inv_area,
    BLOCK_SP: tl.constexpr,
):
    n = tl.program_id(0)
    oc = tl.program_id(1)

    total: tl.constexpr = H_out * W_out

    bias = tl.load(b_ptr + oc)

    # Load all 9 weight vectors [IC] for this oc
    ic_offs = tl.arange(0, IC)
    w_base = oc * IC * 9 + ic_offs * 9
    w00 = tl.load(w_ptr + w_base + 0)
    w01 = tl.load(w_ptr + w_base + 1)
    w02 = tl.load(w_ptr + w_base + 2)
    w10 = tl.load(w_ptr + w_base + 3)
    w11 = tl.load(w_ptr + w_base + 4)
    w12 = tl.load(w_ptr + w_base + 5)
    w20 = tl.load(w_ptr + w_base + 6)
    w21 = tl.load(w_ptr + w_base + 7)
    w22 = tl.load(w_ptr + w_base + 8)

    # Stack weights into a [9, IC] tile for tl.dot
    # We'll just do per-input-position dot products via tl.sum.

    x_base = n * IC * H * W + ic_offs * H * W  # [IC]

    sum_acc = tl.zeros((), dtype=tl.float32)

    sp_offs = tl.arange(0, BLOCK_SP)

    for sp_start in range(0, total, BLOCK_SP):
        idx = sp_start + sp_offs
        mask_sp = idx < total
        ho = idx // W_out
        wo = idx % W_out
        h0 = ho * 2
        w0 = wo * 2

        acc00 = tl.zeros((BLOCK_SP,), dtype=tl.float32)
        acc01 = tl.zeros((BLOCK_SP,), dtype=tl.float32)
        acc10 = tl.zeros((BLOCK_SP,), dtype=tl.float32)
        acc11 = tl.zeros((BLOCK_SP,), dtype=tl.float32)

        # Loop over the 4x4 input patch positions (ir, ic_), each maps to
        # contributions for outputs (dy, dx) where kh=ir-dy in [0,2] and kw=ic_-dx in [0,2].
        for ir in tl.static_range(0, 4):
            ih = h0 - 1 + ir  # [BLOCK_SP]
            row_valid = (ih >= 0) & (ih < H)
            for ic_ in tl.static_range(0, 4):
                iw = w0 - 1 + ic_
                col_valid = (iw >= 0) & (iw < W)
                in_mask = row_valid & col_valid & mask_sp  # [BLOCK_SP]
                addr = x_base[None, :] + ih[:, None] * W + iw[:, None]  # [BLOCK_SP, IC]
                x_val = tl.load(x_ptr + addr, mask=in_mask[:, None], other=0.0)

                # Determine which weight applies for each (dy, dx)
                # dy=0: kh=ir; dy=1: kh=ir-1
                # dx=0: kw=ic_; dx=1: kw=ic_-1
                # Only contribute if kh in [0,2] and kw in [0,2]
                # Use static python ifs (no Triton dynamic branching) by selecting w_vec
                for dy in tl.static_range(0, 2):
                    kh = ir - dy
                    for dx in tl.static_range(0, 2):
                        kw = ic_ - dx
                        if (kh >= 0) and (kh <= 2) and (kw >= 0) and (kw <= 2):
                            # Pick the appropriate weight vector
                            if kh == 0 and kw == 0:
                                wv = w00
                            elif kh == 0 and kw == 1:
                                wv = w01
                            elif kh == 0 and kw == 2:
                                wv = w02
                            elif kh == 1 and kw == 0:
                                wv = w10
                            elif kh == 1 and kw == 1:
                                wv = w11
                            elif kh == 1 and kw == 2:
                                wv = w12
                            elif kh == 2 and kw == 0:
                                wv = w20
                            elif kh == 2 and kw == 1:
                                wv = w21
                            else:
                                wv = w22
                            contrib = tl.sum(x_val * wv[None, :], axis=1)
                            if dy == 0 and dx == 0:
                                acc00 = acc00 + contrib
                            elif dy == 0 and dx == 1:
                                acc01 = acc01 + contrib
                            elif dy == 1 and dx == 0:
                                acc10 = acc10 + contrib
                            else:
                                acc11 = acc11 + contrib

        acc00 = acc00 + bias
        acc01 = acc01 + bias
        acc10 = acc10 + bias
        acc11 = acc11 + bias

        m = tl.maximum(tl.maximum(acc00, acc01), tl.maximum(acc10, acc11))
        m = tl.minimum(tl.maximum(m, htanh_min), htanh_max)

        sum_acc += tl.sum(tl.where(mask_sp, m, 0.0), axis=0)

    mean_val = sum_acc * inv_area
    e2 = tl.exp(2.0 * mean_val)
    t = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + n * OC + oc, t)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Keep the conv_transpose as the source of truth for parameters
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

        # Conditions for the fused path: stride=1, padding=1, kernel=3, maxpool 2x2 stride 2
        self.use_fused = (stride == 1 and padding == 1 and kernel_size == 3
                          and maxpool_kernel_size == 2 and maxpool_stride == 2)

    def _get_equiv_conv_weight(self):
        # ConvTranspose2d weight shape: [in_channels, out_channels, kH, kW]
        # Equivalent Conv2d weight (for stride=1, padding=1, k=3):
        # W_conv[oc, ic, kh, kw] = W_convT[ic, oc, kH-1-kh, kW-1-kw]
        w = self.conv_transpose.weight  # [IC, OC, 3, 3]
        w_eq = w.permute(1, 0, 2, 3).contiguous()  # [OC, IC, 3, 3]
        w_eq = torch.flip(w_eq, dims=(2, 3)).contiguous()
        return w_eq

    def forward(self, x):
        if not self.use_fused:
            x = self.conv_transpose(x)
            x = F.max_pool2d(x, self.maxpool_kernel_size, self.maxpool_stride)
            x = F.hardtanh(x, self.hardtanh_min, self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x

        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        H_out = H // 2
        W_out = W // 2

        w_eq = self._get_equiv_conv_weight()
        bias = self.conv_transpose.bias.contiguous()

        out = torch.empty((N, OC), device=x.device, dtype=x.dtype)
        inv_area = 1.0 / float(H_out * W_out)

        grid = (N, OC)
        fused_kernel[grid](
            x, w_eq, bias, out,
            N, IC, OC, H, W,
            H_out, W_out,
            self.hardtanh_min, self.hardtanh_max,
            inv_area,
        )
        return out.view(N, OC, 1, 1)