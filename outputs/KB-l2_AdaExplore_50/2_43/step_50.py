import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 64}, num_warps=8, num_stages=2),
    ],
    key=['C', 'Do', 'Ho', 'Wo'],
)
@triton.jit
def fused_pool_lse_relu_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    Do, Ho, Wo,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_od, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    total_spatial = Do * Ho * Wo
    total = N * total_spatial
    s_start = pid * BLOCK_S

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    neg_inf = float('-inf')

    for s_off in tl.static_range(0, BLOCK_S):
        s = s_start + s_off
        if s < total:
            pid_n = s // total_spatial
            sp = s - pid_n * total_spatial
            od = sp // (Ho * Wo)
            rem = sp - od * (Ho * Wo)
            oh = rem // Wo
            ow = rem - oh * Wo

            id0 = od * 2
            ih0 = oh * 2
            iw0 = ow * 2

            base = pid_n * stride_xn + offs_c * stride_xc

            max_val = tl.full((BLOCK_C,), neg_inf, dtype=tl.float32)

            for dd in tl.static_range(0, 2):
                for hh in tl.static_range(0, 2):
                    for ww in tl.static_range(0, 2):
                        idx = base + (id0 + dd) * stride_xd + (ih0 + hh) * stride_xh + (iw0 + ww) * stride_xw
                        v = tl.load(x_ptr + idx, mask=mask_c, other=neg_inf)
                        max_val = tl.maximum(max_val, v)

            m = tl.max(tl.where(mask_c, max_val, neg_inf), axis=0)
            shifted = tl.where(mask_c, max_val - m, neg_inf)
            e = tl.exp(shifted)
            e = tl.where(mask_c, e, 0.0)
            ssum = tl.sum(e, axis=0)
            lse = tl.log(ssum) + m
            res = tl.maximum(lse, 0.0)

            out_idx = pid_n * stride_on + od * stride_od + oh * stride_oh + ow * stride_ow
            tl.store(out_ptr + out_idx, res)


def fused_pool_lse_relu(x):
    N, C, D, H, W = x.shape
    Do = D // 2
    Ho = H // 2
    Wo = W // 2
    out = torch.empty((N, 1, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)

    total = N * Do * Ho * Wo
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_S']),)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Do, Ho, Wo,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Channels-last 3D often runs much faster on Ada/Ampere for fp32 conv3d
        self.conv = self.conv.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv(x)
        x = fused_pool_lse_relu(x)
        return x