import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['N_PIX', 'C_out', 'K'],
)
@triton.jit
def conv2d_im2col_gemm_kernel(
    x_ptr,        # NHWC input: (N, H_in, W_in, C_in)
    w_ptr,        # weight: (C_out, KH*KW*C_in)  i.e. (C_out, K)
    b_ptr,        # (C_out,)
    out_ptr,      # NHWC output: (N, H_out, W_out, C_out)
    N, H_in, W_in, C_in,
    H_out, W_out, C_out,
    KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # output pixel tile
    pid_n = tl.program_id(1)  # output channel tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output pixel idx (across N*H_out*W_out)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channel idx

    N_PIX = N * H_out * W_out
    m_mask = offs_m < N_PIX
    n_mask = offs_n < C_out

    # decompose offs_m -> (n_idx, oh, ow)
    nhw = offs_m
    n_idx = nhw // (H_out * W_out)
    rem = nhw % (H_out * W_out)
    oh = rem // W_out
    ow = rem % W_out

    # base offset into NHWC input for each output pixel: ((n*H_in + oh)*W_in + ow) * C_in
    in_base = ((n_idx * H_in + oh) * W_in + ow) * C_in  # [BLOCK_M]

    offs_k = tl.arange(0, K)  # K = KH*KW*C_in
    # decompose offs_k -> (kh, kw, ci)
    kh = offs_k // (KW * C_in)
    kwci = offs_k % (KW * C_in)
    kw = kwci // C_in
    ci = kwci % C_in

    # input gather offset per K: (kh * W_in + kw) * C_in + ci
    k_in_off = (kh * W_in + kw) * C_in + ci  # [K]

    # x indices: [BLOCK_M, K]
    x_idx = in_base[:, None] + k_in_off[None, :]
    x_tile = tl.load(x_ptr + x_idx, mask=m_mask[:, None], other=0.0)  # [BLOCK_M, K]

    # weight: (C_out, K), load tile [BLOCK_N, K]
    w_idx = offs_n[:, None] * K + offs_k[None, :]
    w_tile = tl.load(w_ptr + w_idx, mask=n_mask[:, None], other=0.0)  # [BLOCK_N, K]

    # GEMM: [BLOCK_M, K] x [K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
    acc = tl.dot(x_tile, tl.trans(w_tile))

    # bias
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b_vals[None, :]

    # hardswish then relu: relu(x * clamp(x+3,0,6)/6)
    # for x<=0 => 0; x>=3 => x; else x*(x+3)/6
    hs_mid = acc * (acc + 3.0) * (1.0 / 6.0)
    res = tl.where(acc <= 0.0, 0.0, tl.where(acc >= 3.0, acc, hs_mid))

    # store NHWC: out[n, oh, ow, oc]
    out_idx = offs_m[:, None] * C_out + offs_n[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_idx, res, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to (C_out, KH, KW, C_in) -> flatten to (C_out, K)
        with torch.no_grad():
            w = self.conv.weight.detach()  # (C_out, C_in, KH, KW)
            w_perm = w.permute(0, 2, 3, 1).contiguous()  # (C_out, KH, KW, C_in)
            C_out = w_perm.shape[0]
            K = w_perm.shape[1] * w_perm.shape[2] * w_perm.shape[3]
            self.register_buffer('w_flat', w_perm.view(C_out, K).cuda())
            self.register_buffer('b_flat', self.conv.bias.detach().contiguous().cuda())
        self._KH = self.conv.kernel_size[0]
        self._KW = self.conv.kernel_size[1]
        self._K = K
        self._C_out = C_out

    def forward(self, x):
        x = x.cuda()
        N, C_in, H_in, W_in = x.shape
        KH = self._KH
        KW = self._KW
        K = self._K
        C_out = self._C_out
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1

        # Convert input to NHWC contiguous
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # Output buffer NHWC
        out_nhwc = torch.empty((N, H_out, W_out, C_out), device=x.device, dtype=x.dtype)

        N_PIX = N * H_out * W_out

        grid = lambda META: (
            triton.cdiv(N_PIX, META['BLOCK_M']),
            triton.cdiv(C_out, META['BLOCK_N']),
        )

        conv2d_im2col_gemm_kernel[grid](
            x_nhwc, self.w_flat, self.b_flat, out_nhwc,
            N, H_in, W_in, C_in,
            H_out, W_out, C_out,
            KH, KW, K,
        )

        # back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out