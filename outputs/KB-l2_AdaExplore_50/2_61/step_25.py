import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# ConvTranspose3d as a direct GEMM-style "gather conv":
#   y[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, od-kd, oh-kh, ow-kw] * w[ic, oc, kd, kh, kw]
# valid only where (od-kd) in [0, D_in), etc.
# Output spatial size: D_out = D_in + K - 1, etc. (stride=1, padding=0)
# We tile over (N, OC tile, output spatial tile).
# ----------------------------------------------------------------------

@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, y_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    HW_out = H_out * W_out
    DHW_out = D_out * HW_out

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    s_mask = s_offs < DHW_out
    oc_mask = oc_offs < OC

    od = s_offs // HW_out
    rem = s_offs % HW_out
    oh = rem // W_out
    ow = rem % W_out

    acc = tl.zeros((BLOCK_OC, BLOCK_S), dtype=tl.float32)

    # x base for this batch
    x_n_base = pid_n * IC * D_in * H_in * W_in
    # weight shape: (IC, OC, K, K, K), stride: OC*K*K*K, K*K*K, K*K, K, 1
    K3 = K * K * K
    K2 = K * K

    for kd in tl.static_range(0, K):
        id_ = od - kd  # input D index
        d_valid = (id_ >= 0) & (id_ < D_in)
        for kh in tl.static_range(0, K):
            ih = oh - kh
            h_valid = (ih >= 0) & (ih < H_in)
            for kw in tl.static_range(0, K):
                iw = ow - kw
                w_valid = (iw >= 0) & (iw < W_in)
                spatial_mask = d_valid & h_valid & w_valid & s_mask

                # input offset for all ic; we'll loop over ic
                # but better: vectorize over ic? Let's do inner ic loop with accumulation
                in_spatial_off = id_ * H_in * W_in + ih * W_in + iw
                # clamp to avoid OOB; mask covers correctness
                in_spatial_off = tl.where(spatial_mask, in_spatial_off, 0)

                kw_base = kd * K2 + kh * K + kw  # within the K^3

                for ic in range(0, IC):
                    # load x[n, ic, id, ih, iw] for each spatial (BLOCK_S,)
                    x_off = x_n_base + ic * D_in * H_in * W_in + in_spatial_off
                    x_vals = tl.load(x_ptr + x_off, mask=spatial_mask, other=0.0)  # (BLOCK_S,)

                    # load w[ic, oc_offs, kd, kh, kw] -> (BLOCK_OC,)
                    w_off = ic * OC * K3 + oc_offs * K3 + kw_base
                    w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # (BLOCK_OC,)

                    acc += w_vals[:, None] * x_vals[None, :]

    # Store
    y_base = pid_n * OC * DHW_out
    y_off = y_base + oc_offs[:, None] * DHW_out + s_offs[None, :]
    store_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(y_ptr + y_off, acc, mask=store_mask)


def conv_transpose3d_triton(x, weight):
    """
    x: (N, IC, D, H, W)
    weight: (IC, OC, K, K, K)  (PyTorch ConvTranspose3d weight layout)
    """
    N, IC, D_in, H_in, W_in = x.shape
    IC_w, OC, K, K2_, K3_ = weight.shape
    assert IC == IC_w
    assert K == K2_ == K3_

    D_out = D_in + K - 1
    H_out = H_in + K - 1
    W_out = W_in + K - 1

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    DHW_out = D_out * H_out * W_out
    BLOCK_OC = 64
    BLOCK_S = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(DHW_out, BLOCK_S))
    conv_transpose3d_kernel[grid](
        x, weight, y,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K=K,
        BLOCK_OC=BLOCK_OC,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )
    return y


# ----------------------------------------------------------------------
# Fused ReLU + GroupNorm kernel
# ----------------------------------------------------------------------

@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, S, G, C_PER_G,
    eps,
    BLOCK_S: tl.constexpr,
    C_PER_G_C: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    group_size = C_PER_G_C * S
    sum_x = 0.0
    sum_x2 = 0.0

    base = n * C * S + g * C_PER_G_C * S

    for c_off in range(0, C_PER_G_C):
        c_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            sum_x += tl.sum(tl.where(mask, x, 0.0))
            sum_x2 += tl.sum(tl.where(mask, x * x, 0.0))

    mean = sum_x / group_size
    var = sum_x2 / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c_off in range(0, C_PER_G_C):
        c_idx = g * C_PER_G_C + c_off
        w = tl.load(weight_ptr + c_idx)
        b = tl.load(bias_ptr + c_idx)
        c_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            y = (x - mean) * rstd * w + b
            tl.store(y_ptr + c_base + offs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_G = C // groups
    x_c = x.contiguous()
    y = torch.empty_like(x_c)

    BLOCK_S = 1024
    grid = (N, groups)
    fused_relu_groupnorm_kernel[grid](
        x_c, y, weight, bias,
        N, C, S, groups, C_PER_G,
        eps,
        BLOCK_S=BLOCK_S,
        C_PER_G_C=C_PER_G,
        num_warps=8,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.eps = 1e-5
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

    def forward(self, x):
        x = x.contiguous()
        # Use custom convtranspose3d kernel
        y = conv_transpose3d_triton(x, self.conv_transpose.weight)
        if self.bias and self.conv_transpose.bias is not None:
            y = y + self.conv_transpose.bias.view(1, -1, 1, 1, 1)
        y = fused_relu_groupnorm(
            y,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return y