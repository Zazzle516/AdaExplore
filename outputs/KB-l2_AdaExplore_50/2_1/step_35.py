import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['N', 'OC', 'OUT_HW', 'IC', 'KH', 'KW'])
@triton.jit
def conv2d_relu_bias_nhwc_kernel(
    x_ptr,        # input NHWC: [N, H, W, IC]
    w_ptr,        # weight in (KH*KW, IC, OC) layout - row-major
    b_ptr,        # conv bias [OC]
    bias_ptr,     # extra bias [OC]
    out_ptr,      # output NHWC: [N, OH, OW, OC]
    N, H, W, IC,
    OC, KH, KW,
    OH, OW,
    OUT_HW,
    KHKW_IC,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_idx = pid_n

    num_sp_tiles = tl.cdiv(OUT_HW, BLOCK_M)
    num_oc_tiles = tl.cdiv(OC, BLOCK_N)

    # GROUP_M swizzle over (sp_tile, oc_tile) for L2 reuse of input tiles
    num_pid_in_group = GROUP_M * num_oc_tiles
    group_id = pid // num_pid_in_group
    first_pid_sp = group_id * GROUP_M
    group_size_sp = min(num_sp_tiles - first_pid_sp, GROUP_M)
    sp_tile = first_pid_sp + ((pid % num_pid_in_group) % group_size_sp)
    oc_tile = (pid % num_pid_in_group) // group_size_sp

    sp_start = sp_tile * BLOCK_M
    oc_start = oc_tile * BLOCK_N

    offs_m = sp_start + tl.arange(0, BLOCK_M)
    offs_n = oc_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    oh = offs_m // OW
    ow = offs_m % OW

    m_mask = offs_m < OUT_HW
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # x base for this n_idx
    x_n_ptr = x_ptr + n_idx * (H * W * IC)

    # Collapsed loop over (kh, kw, ic) -> single reduction
    # Total = KH*KW*IC = KHKW_IC
    for k_outer in range(0, KHKW_IC, BLOCK_K):
        k_iter = k_outer + offs_k  # [BLOCK_K]
        k_mask = k_iter < KHKW_IC

        # Decompose k_iter -> (kh*KW + kw, ic)
        kk = k_iter // IC          # kh*KW + kw, in [0, KH*KW)
        ic = k_iter % IC
        kh = kk // KW
        kw = kk % KW

        # Input offsets: x[n, oh+kh, ow+kw, ic]
        # Need (BLOCK_M, BLOCK_K)
        ih = oh[:, None] + kh[None, :]   # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw[None, :]
        x_addr = ih * (W * IC) + iw * IC + ic[None, :]
        x_vals = tl.load(
            x_n_ptr + x_addr,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Weight: w[kk, ic, oc] => ((kh*KW+kw) * IC + ic) * OC + oc
        w_row = kk * IC + ic   # [BLOCK_K]
        w_addr = w_row[:, None] * OC + offs_n[None, :]
        w_vals = tl.load(
            w_ptr + w_addr,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(x_vals, w_vals)

    # Conv bias
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b_vals[None, :]
    # ReLU
    acc = tl.maximum(acc, 0.0)
    # Extra bias
    eb_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc += eb_vals[None, :]

    # Store NHWC output: out[n, oh, ow, oc]
    out_n_ptr = out_ptr + n_idx * (OH * OW * OC)
    out_addr = oh[:, None] * (OW * OC) + ow[:, None] * OC + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_n_ptr + out_addr, acc, mask=out_mask)


def conv2d_relu_bias_nhwc(x_nhwc, w_khkw_ic_oc, b, bias_flat):
    N, H, W, IC = x_nhwc.shape
    KHKW, IC2, OC = w_khkw_ic_oc.shape
    # KH*KW separately stored at python level
    return None  # placeholder, never used


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to (KH, KW, IC, OC) layout, contiguous.
        self._cached_weight = None
        self._cached_weight_version = None

    def _get_weight_nhwc(self):
        w = self.conv.weight  # [OC, IC, KH, KW]
        ver = w._version
        if self._cached_weight is None or self._cached_weight_version != ver:
            # Permute to (KH, KW, IC, OC)
            w_perm = w.permute(2, 3, 1, 0).contiguous()
            OC, IC, KH, KW = w.shape
            w_perm = w_perm.view(KH * KW, IC, OC).contiguous()
            self._cached_weight = w_perm
            self._cached_weight_version = ver
        return self._cached_weight

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()  # NCHW
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        # Convert input NCHW -> NHWC (channels_last memory format is faster than permute+contiguous)
        x_nhwc = x.to(memory_format=torch.channels_last)
        # Reinterpret as NHWC layout for kernel indexing
        x_nhwc = x_nhwc.permute(0, 2, 3, 1)
        if not x_nhwc.is_contiguous():
            x_nhwc = x_nhwc.contiguous()

        w_perm = self._get_weight_nhwc()  # (KH*KW, IC, OC)
        b = self.conv.bias.contiguous()
        bias_flat = self.bias.contiguous().view(-1)

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        OUT_HW = OH * OW

        KHKW_IC = KH * KW * IC

        grid = lambda meta: (
            triton.cdiv(OUT_HW, meta["BLOCK_M"]) * triton.cdiv(OC, meta["BLOCK_N"]),
            N,
        )

        conv2d_relu_bias_nhwc_kernel[grid](
            x_nhwc, w_perm, b, bias_flat, out_nhwc,
            N, H, W, IC,
            OC, KH, KW,
            OH, OW,
            OUT_HW,
            KHKW_IC,
        )

        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out