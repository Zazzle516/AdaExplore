import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    H_pool, W_pool,
    sub1, sub2,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # program_id: (n, c_out, hw_pool_tile)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    pool_total = H_pool * W_pool
    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < pool_total

    ph = offs // W_pool
    pw = offs % W_pool

    # For each pool output, we need to sum POOL*POOL conv outputs
    inv_pool_area = 1.0 / (POOL * POOL)

    bias = tl.load(b_ptr + pid_c)

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Iterate over pool window
    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            oh = ph * POOL + dh
            ow = pw * POOL + dw
            # Compute conv output at (pid_n, pid_c, oh, ow)
            conv_acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
            for ic in range(C_in):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        ih = oh + kh
                        iw = ow + kw
                        in_mask = mask & (ih < H_in) & (iw < W_in)
                        x_off = pid_n * (C_in * H_in * W_in) + ic * (H_in * W_in) + ih * W_in + iw
                        x_val = tl.load(x_ptr + x_off, mask=in_mask, other=0.0)
                        w_off = pid_c * (C_in * KH * KW) + ic * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off)
                        conv_acc += x_val * w_val
            conv_acc += bias
            conv_acc = conv_acc - sub1
            # tanh
            e2x = tl.exp(2.0 * conv_acc)
            t = (e2x - 1.0) / (e2x + 1.0)
            t = t - sub2
            acc += t

    acc = acc * inv_pool_area

    out_off = pid_n * (C_out * H_pool * W_pool) + pid_c * (H_pool * W_pool) + ph * W_pool + pw
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def conv_tanh_sub_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    sub1, sub2,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    total = H_out * W_out
    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < total

    oh = offs // W_out
    ow = offs % W_out

    bias = tl.load(b_ptr + pid_c)
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for ic in range(C_in):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                ih = oh + kh
                iw = ow + kw
                x_off = pid_n * (C_in * H_in * W_in) + ic * (H_in * W_in) + ih * W_in + iw
                x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0)
                w_off = pid_c * (C_in * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    acc += bias
    acc = acc - sub1
    e2x = tl.exp(2.0 * acc)
    t = (e2x - 1.0) / (e2x + 1.0)
    t = t - sub2

    out_off = pid_n * (C_out * H_out * W_out) + pid_c * (H_out * W_out) + oh * W_out + ow
    tl.store(out_ptr + out_off, t, mask=mask)


@triton.jit
def avgpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    POOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    total = H_out * W_out
    offs = pid_hw * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    oh = offs // W_out
    ow = offs % W_out

    inv = 1.0 / (POOL * POOL)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            ih = oh * POOL + dh
            iw = ow * POOL + dw
            x_off = pid_nc * (H_in * W_in) + ih * W_in + iw
            v = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            acc += v
    acc = acc * inv

    out_off = pid_nc * (H_out * W_out) + oh * W_out + ow
    tl.store(out_ptr + out_off, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = kernel_size_pool
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        KH = KW = self.kernel_size
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1
        POOL = self.kernel_size_pool

        # Step 1: conv + sub1 + tanh + sub2 -> intermediate
        inter = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float32)
        BLOCK_HW = 128
        grid1 = (N, C_out, triton.cdiv(H_out * W_out, BLOCK_HW))
        conv_tanh_sub_kernel[grid1](
            x, w, b, inter,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            self.subtract1_value, self.subtract2_value,
            KH, KW, BLOCK_HW,
            num_warps=4,
        )

        # Step 2: avg pool
        H_p = H_out // POOL
        W_p = W_out // POOL
        out = torch.empty((N, C_out, H_p, W_p), device=x.device, dtype=torch.float32)
        BLOCK = 128
        grid2 = (N * C_out, triton.cdiv(H_p * W_p, BLOCK))
        avgpool2d_kernel[grid2](
            inter, out,
            N, C_out, H_out, W_out, H_p, W_p,
            POOL, BLOCK,
            num_warps=4,
        )
        return out