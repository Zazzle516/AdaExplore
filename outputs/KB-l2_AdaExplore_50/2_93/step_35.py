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
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M_par', 'N', 'K_par'],
)
@triton.jit
def conv_transpose_parity_kernel(
    x_ptr,             # input (N_b, IC, H, W)
    w_ptr,             # weight pre-permuted: (STRIDE, STRIDE, IC, KH_par, KW_par, OC) effectively
                       # We'll use layout: per-parity tile of shape (K_par, OC) contiguous
                       # passed as w_par_ptr base + parity offset
    b_ptr,             # bias (OC,)
    out_ptr,           # output (N_b, OC, H_out, W_out)
    # parity indices: which parity this kernel is for
    PH, PW,            # parity offsets (0..STRIDE-1)
    H_par, W_par,      # output spatial extents for this parity
    KH_par, KW_par,    # kernel extents along H/W for this parity
    M_par, N, K_par,
    N_b, IC, H, W, OC, H_out, W_out,
    STRIDE: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M_par
    mask_n = offs_n < N

    # decode m -> (n_idx, hp, wp) where hp in [0, H_par), wp in [0, W_par)
    wp = offs_m % W_par
    tmp = offs_m // W_par
    hp = tmp % H_par
    n_idx = tmp // H_par

    # actual output coords:
    # ho = hp * STRIDE + PH
    # wo = wp * STRIDE + PW
    ho = hp * STRIDE + PH
    wo = wp * STRIDE + PW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each parity, the valid (kh, kw) are those with kh % STRIDE == PH and kw % STRIDE == PW.
    # K_par = IC * KH_par * KW_par
    # Decode k -> (ic, khp, kwp) -> kh = khp * STRIDE + PH, kw = kwp * STRIDE + PW
    # Then hi = (ho - kh) / STRIDE = hp - khp; wi = wp - kwp
    # valid if 0 <= hi < H and 0 <= wi < W

    for k_start in range(0, K_par, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_par

        kwp = offs_k % KW_par
        tmpk = offs_k // KW_par
        khp = tmpk % KH_par
        ic = tmpk // KH_par

        hi = hp[:, None] - khp[None, :]
        wi = wp[:, None] - kwp[None, :]
        valid_h = (hi >= 0) & (hi < H)
        valid_w = (wi >= 0) & (wi < W)
        valid = valid_h & valid_w & mask_m[:, None] & mask_k[None, :]

        x_off = (n_idx[:, None] * (IC * H * W)
                 + ic[None, :] * (H * W)
                 + hi * W + wi)
        x_vals = tl.load(x_ptr + x_off, mask=valid, other=0.0)

        # weight layout: (K_par, OC) contiguous for this parity
        # offset = k * OC + n
        w_off = offs_k[:, None] * N + offs_n[None, :]
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    acc = acc + add_value
    acc = tl.minimum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    acc = acc * multiply_value

    out_off = (n_idx[:, None] * (OC * H_out * W_out)
               + offs_n[None, :] * (H_out * W_out)
               + ho[:, None] * W_out
               + wo[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Pre-permute weights per parity.
        # Original weight: (IC, OC, KH, KW)
        # For parity (ph, pw): khp in [0, ceil((KH-ph)/stride)), kwp in [0, ceil((KW-pw)/stride))
        # weight[ic, oc, ph + khp*stride, pw + kwp*stride]
        # we want layout: (K_par, OC) where K_par = IC * KH_par * KW_par
        # k index: ic*KH_par*KW_par + khp*KW_par + kwp
        S = stride
        KH = kernel_size
        KW = kernel_size
        self.parity_weights = []
        self.parity_meta = []  # list of (ph, pw, KH_par, KW_par, K_par)
        with torch.no_grad():
            W = self.conv_transpose.weight.data  # (IC, OC, KH, KW)
            IC = in_channels
            OC = out_channels
            for ph in range(S):
                for pw in range(S):
                    KH_par = (KH - ph + S - 1) // S
                    KW_par = (KW - pw + S - 1) // S
                    if KH_par <= 0 or KW_par <= 0:
                        self.parity_weights.append(None)
                        self.parity_meta.append((ph, pw, 0, 0, 0))
                        continue
                    # gather
                    # rows kh = ph, ph+S, ph+2S, ...
                    kh_idx = torch.arange(ph, KH, S)
                    kw_idx = torch.arange(pw, KW, S)
                    Wp = W[:, :, kh_idx][:, :, :, kw_idx]  # (IC, OC, KH_par, KW_par)
                    # permute to (IC, KH_par, KW_par, OC) then flatten K_par
                    Wp = Wp.permute(0, 2, 3, 1).contiguous()  # (IC, KH_par, KW_par, OC)
                    K_par = IC * KH_par * KW_par
                    Wp = Wp.view(K_par, OC).contiguous()
                    self.parity_weights.append(Wp)
                    self.parity_meta.append((ph, pw, KH_par, KW_par, K_par))

        # Register as buffers so they move with .to(device)
        for i, wp in enumerate(self.parity_weights):
            if wp is not None:
                self.register_buffer(f"_pw_{i}", wp)

    def forward(self, x):
        x = x.contiguous()
        N_b, IC, H, W = x.shape
        OC = self.out_channels
        S = self.stride
        KH = self.kernel_size
        KW = self.kernel_size
        H_out = (H - 1) * S + KH
        W_out = (W - 1) * S + KW

        out = torch.empty((N_b, OC, H_out, W_out), device=x.device, dtype=x.dtype)
        bias = self.conv_transpose.bias.contiguous()

        for i, (ph, pw, KH_par, KW_par, K_par) in enumerate(self.parity_meta):
            if K_par == 0:
                continue
            # H_par = number of hp such that hp*S + ph < H_out
            # hp in [0, ceil((H_out - ph)/S))
            H_par = (H_out - ph + S - 1) // S
            W_par = (W_out - pw + S - 1) // S
            if H_par <= 0 or W_par <= 0:
                continue
            M_par = N_b * H_par * W_par
            w_par = getattr(self, f"_pw_{i}")

            grid = lambda meta: (
                triton.cdiv(M_par, meta['BLOCK_M']),
                triton.cdiv(OC, meta['BLOCK_N']),
            )
            conv_transpose_parity_kernel[grid](
                x, w_par, bias, out,
                ph, pw,
                H_par, W_par,
                KH_par, KW_par,
                M_par, OC, K_par,
                N_b, IC, H, W, OC, H_out, W_out,
                S, KH, KW,
                self.add_value, self.multiply_value,
            )
        return out