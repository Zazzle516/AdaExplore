import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,        # input: (N, IC, H_in, W_in), but viewed as (M, K) where M=N*H_out*W_out, K=IC, we'll gather
    w_ptr,        # weight: (IC, OC, KH, KW)
    b_ptr,        # bias: (OC,)
    out_ptr,      # output: (N, OC, H_out, W_out)
    M, N_OC, K_IC,
    N_BATCH, H_IN, W_IN,
    H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    ADD_VAL: tl.constexpr,
    MUL_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M = N_BATCH * H_OUT * W_OUT  (output spatial positions across batch)
    # N_OC = OC
    # K_IC = IC
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # decode m -> (n, h_out, w_out)
    HW = H_OUT * W_OUT
    n_idx = offs_m // HW
    hw_idx = offs_m % HW
    h_out = hw_idx // W_OUT
    w_out = hw_idx % W_OUT

    m_mask = offs_m < M
    n_mask = offs_n < N_OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For ConvTranspose2d with no padding:
    # out[n, oc, h_out, w_out] = sum_{ic, kh, kw} x[n, ic, h_in, w_in] * w[ic, oc, kh, kw]
    # where h_in*stride + kh = h_out  =>  h_in = (h_out - kh)/stride, must be integer and in [0,H_IN)
    # Iterate over kh, kw, ic
    for kh in tl.static_range(0, KH):
        h_in_num = h_out - kh
        h_in = h_in_num // STRIDE
        h_in_valid = (h_in_num >= 0) & (h_in_num % STRIDE == 0) & (h_in < H_IN)
        for kw in tl.static_range(0, KW):
            w_in_num = w_out - kw
            w_in = w_in_num // STRIDE
            w_in_valid = (w_in_num >= 0) & (w_in_num % STRIDE == 0) & (w_in < W_IN)
            spatial_valid = h_in_valid & w_in_valid & m_mask  # [BLOCK_M]

            # x_offset base for these (n, h_in, w_in): n*IC*H_IN*W_IN + h_in*W_IN + w_in
            # then for ic loop, add ic*H_IN*W_IN
            x_base = n_idx * (K_IC * H_IN * W_IN) + h_in * W_IN + w_in  # [BLOCK_M]

            # w_offset base: w[ic, oc, kh, kw]; for ic loop add ic*OC*KH*KW; oc varies in inner
            # w_base for (kh, kw) and varying oc: oc*KH*KW + kh*KW + kw, plus ic*OC*KH*KW
            w_base = offs_n * (KH * KW) + kh * KW + kw  # [BLOCK_N]

            # Loop over IC in blocks of BLOCK_K
            for k0 in range(0, K_IC, BLOCK_K):
                ic = k0 + tl.arange(0, BLOCK_K)
                ic_mask = ic < K_IC

                x_ptrs = x_ptr + x_base[:, None] + ic[None, :] * (H_IN * W_IN)
                x_mask = spatial_valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                w_ptrs = w_ptr + w_base[None, :] + ic[:, None] * (N_OC * KH * KW)
                w_mask = n_mask[None, :] & ic_mask[:, None]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Epilogue: add, min(.,0), gelu, multiply
    acc = acc + ADD_VAL
    acc = tl.minimum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    acc = acc * MUL_VAL

    # Store: out[n, oc, h_out, w_out]
    out_ptrs = out_ptr + n_idx[:, None] * (N_OC * H_OUT * W_OUT) \
        + offs_n[None, :] * (H_OUT * W_OUT) \
        + (h_out * W_OUT + w_out)[:, None]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)
        bias = self.conv_transpose.bias.contiguous()

        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        stride = self.stride

        H_out = (H_in - 1) * stride + KH
        W_out = (W_in - 1) * stride + KW

        out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

        M = N * H_out * W_out
        N_OC = OC
        K_IC = IC

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N_OC, META['BLOCK_N']),
        )

        conv_transpose_fused_kernel[grid](
            x, weight, bias, out,
            M, N_OC, K_IC,
            N, H_in, W_in,
            H_out, W_out,
            KH=KH, KW=KW,
            STRIDE=stride,
            ADD_VAL=self.add_value,
            MUL_VAL=self.multiply_value,
        )
        return out