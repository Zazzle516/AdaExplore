import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Gather-style ConvTranspose3d implemented as a GEMM where each output voxel
# corresponds to one row of the GEMM's N axis. For ConvTranspose3d with
# stride=S, padding=P, and kernel size K:
#   out[od, oh, ow] = sum over (kd, kh, kw) and ic of
#       input[id, ih, iw, ic] * weight[ic, oc, kd, kh, kw]
# where id = (od + P - kd) / S, requiring (od + P - kd) % S == 0 and 0<=id<ID
# (and similarly for h, w).
#
# We pre-build, on the host, a per-output-position mapping that lists the valid
# kernel taps. Since stride=2, padding=1, kernel=3, each output position has
# either 1 or 2 valid taps in each spatial dim.
#
# Layout: input and output stored as NDHWC (channels-last innermost) so the
# IC axis is contiguous (GEMM-K) and OC is contiguous (GEMM-M). Weight is
# laid out as [kd*kh*kw, IC, OC] -> contiguous on OC.


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr,                  # [N, ID, IH, IW, IC] contiguous, NDHWC
    w_ptr,                  # [KD*KH*KW, IC, OC] contiguous
    b_ptr,                  # [OC]
    out_ptr,                # [N, OD, OH, OW, OC] contiguous, NDHWC
    # Sizes
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    # Conv params
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    S: tl.constexpr, P: tl.constexpr,
    # Epilogue params (scalars baked in)
    MIN_VAL: tl.constexpr,
    INV_DIV: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,   # OC tile
    BLOCK_N: tl.constexpr,   # spatial (D*H*W) tile per (n)
    BLOCK_K: tl.constexpr,   # IC tile
):
    pid_m = tl.program_id(0)   # OC tile
    pid_n = tl.program_id(1)   # spatial tile within one batch (od*oh*ow)
    pid_b = tl.program_id(2)   # batch

    OHW = OH * OW
    ODHW = OD * OH * OW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)            # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)            # spatial
    mask_m = offs_m < OC
    mask_n = offs_n < ODHW

    # Decode (od, oh, ow) for each spatial idx
    od = offs_n // OHW
    rem = offs_n - od * OHW
    oh = rem // OW
    ow = rem - oh * OW

    # For each output coord, valid taps satisfy (oc + P - k) % S == 0 and
    # 0 <= (oc + P - k) / S < I. With S=2, P=1, K=3 the taps are k in
    # parity matching (od + P) odd/even.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kernel positions
    for kd in tl.static_range(0, KD):
        d_num = od + P - kd
        d_ok = (d_num % S == 0)
        id_ = d_num // S
        d_in = (id_ >= 0) & (id_ < ID) & d_ok
        for kh in tl.static_range(0, KH):
            h_num = oh + P - kh
            h_ok = (h_num % S == 0)
            ih_ = h_num // S
            h_in = (ih_ >= 0) & (ih_ < IH) & h_ok
            for kw in tl.static_range(0, KW):
                w_num = ow + P - kw
                w_ok = (w_num % S == 0)
                iw_ = w_num // S
                w_in = (iw_ >= 0) & (iw_ < IW) & w_ok

                spatial_valid = d_in & h_in & w_in & mask_n

                # Compute base input pointer for each spatial element
                # x_index = ((b*ID + id_)*IH + ih_)*IW + iw_, then * IC
                in_base = ((pid_b * ID + id_) * IH + ih_) * IW + iw_   # [BLOCK_N]
                in_base = in_base * IC

                # Weight base for this kernel tap
                k_lin = (kd * KH + kh) * KW + kw
                w_base = k_lin * IC * OC

                # GEMM over IC for this tap
                for k0 in range(0, IC, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < IC

                    # Load x: shape [BLOCK_N, BLOCK_K]
                    x_ptrs = x_ptr + in_base[:, None] + offs_k[None, :]
                    x_mask = spatial_valid[:, None] & mask_k[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                    # Load w: shape [BLOCK_K, BLOCK_M]
                    w_ptrs = w_ptr + w_base + offs_k[:, None] * OC + offs_m[None, :]
                    w_mask = mask_k[:, None] & mask_m[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                    # acc[BLOCK_M, BLOCK_N] += w^T @ x^T -> use tl.dot
                    # x_vals: [BLOCK_N, BLOCK_K], w_vals: [BLOCK_K, BLOCK_M]
                    # we want acc shape [BLOCK_M, BLOCK_N], so compute
                    # (w_vals.T @ x_vals.T) = (x_vals @ w_vals).T ... easier:
                    # accumulate as [BLOCK_N, BLOCK_M] then transpose at end.
                    acc += tl.dot(w_vals, tl.trans(x_vals))

    # Add bias
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias[:, None]

    # Clamp + divide
    acc = tl.where(acc < MIN_VAL, MIN_VAL, acc)
    acc = acc * INV_DIV

    # Store: out shape [N, OD, OH, OW, OC], so for each spatial pos n and
    # channel m: index = pid_b*ODHW*OC + offs_n*OC + offs_m
    out_base = pid_b * ODHW * OC
    out_ptrs = out_ptr + out_base + offs_n[None, :] * OC + offs_m[:, None]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def _conv_transpose3d_fused(x_ndhwc, w_kihw_ic_oc, bias, N, IC, OC, ID, IH, IW,
                             OD, OH, OW, KD, KH, KW, S, P, min_val, inv_div):
    out = torch.empty((N, OD, OH, OW, OC), device=x_ndhwc.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(OC, BLOCK_M),
        triton.cdiv(OD * OH * OW, BLOCK_N),
        N,
    )

    conv_transpose3d_fused_kernel[grid](
        x_ndhwc, w_kihw_ic_oc, bias, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        S=S, P=P,
        MIN_VAL=float(min_val),
        INV_DIV=float(inv_div),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = float(min_value)
        self.divisor = float(divisor)

        # Use ConvTranspose3d to initialize weights with the same default init
        ct = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding)
        # weight shape: [in_channels, out_channels, kD, kH, kW]
        self.weight = nn.Parameter(ct.weight.detach().clone())
        self.bias = nn.Parameter(ct.bias.detach().clone())

        self.KD = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.KH = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        self.KW = kernel_size if isinstance(kernel_size, int) else kernel_size[2]

        self._w_packed = None
        self._w_version = None

    def _pack_weight(self):
        # weight: [IC, OC, KD, KH, KW] -> [KD*KH*KW, IC, OC]
        w = self.weight
        IC, OC, KD, KH, KW = w.shape
        # permute to [KD, KH, KW, IC, OC]
        w_p = w.permute(2, 3, 4, 0, 1).contiguous().view(KD * KH * KW, IC, OC)
        return w_p

    def forward(self, x):
        x = x.cuda().contiguous()
        if self.weight.device != x.device:
            self.weight.data = self.weight.data.to(x.device)
            self.bias.data = self.bias.data.to(x.device)

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        S = self.stride
        P = self.padding
        KD, KH, KW = self.KD, self.KH, self.KW

        OD = (ID - 1) * S - 2 * P + KD
        OH = (IH - 1) * S - 2 * P + KH
        OW = (IW - 1) * S - 2 * P + KW

        # Convert input to NDHWC
        x_ndhwc = x.permute(0, 2, 3, 4, 1).contiguous()

        # Pack weight (cache)
        cur_ver = self.weight._version
        if self._w_packed is None or self._w_version != cur_ver or self._w_packed.device != x.device:
            self._w_packed = self._pack_weight().to(x.device)
            self._w_version = cur_ver

        bias = self.bias.contiguous()

        out_ndhwc = _conv_transpose3d_fused(
            x_ndhwc, self._w_packed, bias,
            N, IC, OC, ID, IH, IW, OD, OH, OW,
            KD, KH, KW, S, P,
            self.min_value, 1.0 / self.divisor,
        )

        # Convert back to NCDHW
        out = out_ndhwc.permute(0, 4, 1, 2, 3).contiguous()
        return out