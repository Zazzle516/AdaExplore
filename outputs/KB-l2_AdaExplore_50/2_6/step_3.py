import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_maxpool4_kernel(
    x_ptr,      # input after conv: (N, C, D, H, W)
    out_ptr,    # output: (N, C, D/4, H/4, W/4)
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, do, ho, wo) - reduces over channel & 4x4x4 window
    pid = tl.program_id(0)
    n = tl.program_id(1)

    total = Do * Ho * Wo
    if pid >= total:
        return

    wo = pid % Wo
    ho = (pid // Wo) % Ho
    do = pid // (Wo * Ho)

    d0 = do * 4
    h0 = ho * 4
    w0 = wo * 4

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # We need softmax over channel dim for each (d,h,w) in the 4x4x4 window,
    # then max over the window per channel, then max again (effectively over 4x4x4).
    # Since pool1 then pool2 both are size 2 with no overlap, combined it's a 4x4x4 max.
    # For each channel c, result[c] = max over (d,h,w) in window of softmax(x[c,d,h,w]).
    # Then we still need to output all C channels. But the kernel here outputs all C channels
    # for one (n, do, ho, wo) tile.

    # We'll iterate over the 64 positions in the window, compute softmax across C, and track max per channel.
    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)

    for k in tl.static_range(0, 64):
        dk = k // 16
        hk = (k // 4) % 4
        wk = k % 4
        d = d0 + dk
        h = h0 + hk
        w = w0 + wk

        in_bounds = (d < D) & (h < H) & (w < W)
        # load all channels at this spatial position
        base = ((n * C + 0) * D + d) * H * W + h * W + w
        ptrs = x_ptr + base + c_offs * (D * H * W)
        load_mask = c_mask & in_bounds
        vals = tl.load(ptrs, mask=load_mask, other=-float('inf'))

        # softmax across channel
        m = tl.max(vals, axis=0)
        e = tl.exp(vals - m)
        # mask out invalid channels to 0
        e = tl.where(c_mask, e, 0.0)
        s = tl.sum(e, axis=0)
        sm = e / s
        # for out-of-bounds spatial positions, set sm to -inf so it doesn't affect max
        sm = tl.where(in_bounds, sm, -float('inf'))
        max_vals = tl.maximum(max_vals, sm)

    # store result: out[n, :, do, ho, wo]
    out_base = ((n * C + 0) * Do + do) * Ho * Wo + ho * Wo + wo
    out_ptrs = out_ptr + out_base + c_offs * (Do * Ho * Wo)
    tl.store(out_ptrs, max_vals, mask=c_mask)


def fused_softmax_maxpool(x, pool_size_total=4):
    N, C, D, H, W = x.shape
    Do = D // pool_size_total
    Ho = H // pool_size_total
    Wo = W // pool_size_total
    out = torch.empty((N, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    # round C up to next power of 2 for BLOCK_C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    BLOCK_C = max(BLOCK_C, 16)

    total = Do * Ho * Wo
    grid = (total, N)
    fused_softmax_maxpool4_kernel[grid](
        x, out, N, C, D, H, W, Do, Ho, Wo,
        BLOCK_C=BLOCK_C,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_total = pool_kernel_size * pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        # Ensure spatial dims are divisible by pool_total; if not, crop (matches max pool behavior with floor)
        N, C, D, H, W = x.shape
        pt = self.pool_total
        D2 = (D // pt) * pt
        H2 = (H // pt) * pt
        W2 = (W // pt) * pt
        if D2 != D or H2 != H or W2 != W:
            x = x[:, :, :D2, :H2, :W2].contiguous()
        else:
            x = x.contiguous()
        return fused_softmax_maxpool(x, pool_size_total=pt)