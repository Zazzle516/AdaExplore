import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'IC_C', 'POOL'],
)
@triton.jit
def fused_conv_softmax_maxpool_kernel(
    x_ptr,          # input: (N, IC, ID, IH, IW)
    w_ptr,          # weight_t: (K_FLAT, OC) = (KD*KH*KW*IC, OC)
    b_ptr,          # bias: (OC,)
    out_ptr,        # output: (N, OC, Do, Ho, Wo)
    N, IC, ID, IH, IW,
    Do, Ho, Wo,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    POOL: tl.constexpr,        # 4
    POOL3: tl.constexpr,       # POOL**3 = 64
    K_FLAT: tl.constexpr,      # KD*KH*KW*IC = 81
    K_PAD: tl.constexpr,       # padded K, e.g. 128
):
    pid = tl.program_id(0)
    n = tl.program_id(1)

    wo = pid % Wo
    ho = (pid // Wo) % Ho
    do = pid // (Wo * Ho)

    d0 = do * POOL
    h0 = ho * POOL
    w0 = wo * POOL

    # ---- Build im2col patch matrix [POOL3, K_PAD] ----
    k_off = tl.arange(0, POOL3)                      # [POOL3]
    dk = k_off // (POOL * POOL)
    hk = (k_off // POOL) % POOL
    wk = k_off % POOL

    kf = tl.arange(0, K_PAD)                         # [K_PAD]
    # layout: kd * (KH*KW*IC) + kh * (KW*IC) + kw * IC + ic
    kd_ = (kf // (KH * KW * IC_C)) % KD
    kh_ = (kf // (KW * IC_C)) % KH
    kw_ = (kf // IC_C) % KW
    ic_ = kf % IC_C
    k_mask = kf < K_FLAT                              # [K_PAD]

    d_idx = dk[:, None] + kd_[None, :]                # [POOL3, K_PAD]
    h_idx = hk[:, None] + kh_[None, :]
    w_idx = wk[:, None] + kw_[None, :]
    ic_idx = ic_[None, :] + tl.zeros([POOL3, 1], tl.int32)

    addr = ((n * IC + ic_idx) * ID + (d0 + d_idx)) * (IH * IW) + (h0 + h_idx) * IW + (w0 + w_idx)
    patches = tl.load(x_ptr + addr, mask=k_mask[None, :], other=0.0)  # [POOL3, K_PAD]

    # ---- Load weight_t [K_PAD, OC] ----
    w_row = tl.arange(0, K_PAD)
    w_col = tl.arange(0, OC)
    w_addr = w_row[:, None] * OC + w_col[None, :]
    w_mask = (w_row[:, None] < K_FLAT)
    w_mat = tl.load(w_ptr + w_addr, mask=w_mask, other=0.0)  # [K_PAD, OC]

    # ---- GEMM: conv_out = patches @ w_mat ----
    conv_out = tl.dot(patches, w_mat)  # [POOL3, OC]

    bias = tl.load(b_ptr + tl.arange(0, OC))           # [OC]
    conv_out = conv_out + bias[None, :]

    # ---- Softmax along OC ----
    m = tl.max(conv_out, axis=1)                       # [POOL3]
    e = tl.exp(conv_out - m[:, None])
    s = tl.sum(e, axis=1)                              # [POOL3]
    sm = e / s[:, None]                                # [POOL3, OC]

    # ---- Max-pool over POOL3 spatial positions ----
    result = tl.max(sm, axis=0)                        # [OC]

    out_base = ((n * OC + 0) * Do + do) * Ho * Wo + ho * Wo + wo
    out_offs = tl.arange(0, OC) * (Do * Ho * Wo)
    tl.store(out_ptr + out_base + out_offs, result)


def fused_conv_softmax_pool(x, weight_t, bias, OC, KD, KH, KW, IC, pool_total=4):
    N, IC_in, ID, IH, IW = x.shape

    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    Do = OD // pool_total
    Ho = OH // pool_total
    Wo = OW // pool_total

    out = torch.empty((N, OC, Do, Ho, Wo), device=x.device, dtype=torch.float32)

    total = Do * Ho * Wo
    grid = (total, N)

    K_FLAT = KD * KH * KW * IC
    # Pad K to next power of two >= max(K_FLAT, 16)
    K_PAD = 1
    while K_PAD < max(K_FLAT, 16):
        K_PAD *= 2

    POOL = pool_total
    POOL3 = POOL * POOL * POOL

    fused_conv_softmax_maxpool_kernel[grid](
        x, weight_t, bias, out,
        N, IC, ID, IH, IW,
        Do, Ho, Wo,
        OC=OC,
        KD=KD, KH=KH, KW=KW,
        IC_C=IC,
        POOL=POOL,
        POOL3=POOL3,
        K_FLAT=K_FLAT,
        K_PAD=K_PAD,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_total = pool_kernel_size * pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._cached_weight_t = None
        self._cached_weight_version = None

    def _get_weight_t(self):
        w = self.conv.weight
        ver = w._version
        if self._cached_weight_t is None or self._cached_weight_version != ver or self._cached_weight_t.device != w.device:
            # weight: (OC, IC, KD, KH, KW) -> (KD, KH, KW, IC, OC) -> (K_FLAT, OC)
            w_t = w.detach().permute(2, 3, 4, 1, 0).contiguous()
            OC = w.shape[0]
            self._cached_weight_t = w_t.view(-1, OC).contiguous()
            self._cached_weight_version = ver
        return self._cached_weight_t

    def forward(self, x):
        x = x.contiguous()
        weight_t = self._get_weight_t()
        bias = self.conv.bias.contiguous()
        OC, IC, KD, KH, KW = self.conv.weight.shape
        return fused_conv_softmax_pool(
            x, weight_t, bias,
            OC=OC, KD=KD, KH=KH, KW=KW, IC=IC,
            pool_total=self.pool_total,
        )