import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def phase_conv_kernel(
    x_ptr,           # [N, IC, H_IN, W_IN]
    w_ptr,           # [OC, IC, 2, 2, 4] per-phase weight: phase = ph*2+pw, taps over (kh_sub, kw_sub)
    bias_ptr,        # [OC]
    out_ptr,         # [N, OC, H_OUT, W_OUT]
    N, IC, OC,
    H_IN, W_IN,
    H_OUT, W_OUT,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    PHASE: tl.constexpr,
):
    """
    For ConvTranspose2d with stride=2, kernel=4, pad=0:
      H_OUT = (H_IN-1)*2 + 4 = 2*H_IN + 2
      W_OUT = 2*W_IN + 2
    For each output phase (ph, pw) in {0,1}x{0,1}:
      out[n, oc, 2*hi + ph + dh, 2*wi + pw + dw] += sum_ic x[n,ic,hi,wi] * w[ic,oc,kh,kw]
      where kh = ph + 2*dh_sub? Actually, derivation:
        h_out - kh = 2 * h_in  =>  kh = h_out - 2*h_in
        valid kh in [0..3]. For h_out = 2*hi_grid + ph (ph in {0,1}), h_in = hi_grid - (kh - ph)/2
        kh has same parity as ph. So kh in {ph, ph+2}.
    For PHASE = ph*2 + pw, we run a 2x2 conv over input with output spatial size = H_IN+1 (since h_out_grid: when ph=0, h_out in {0,2,...,2*H_IN}, that's H_IN+1 values; similarly ph=1, h_out in {1,3,...,2*H_IN+1}, H_IN+1 values). Total H_OUT = 2*(H_IN+1) = 2*H_IN+2. Good.
    
    For phase ph, output sub-grid index hi_grid in [0, H_IN] (size H_IN+1):
      actual h_out = 2*hi_grid + ph
      taps: kh in {ph, ph+2}, with kh = ph + 2*dh_sub, dh_sub in {0,1}
        h_in = hi_grid - dh_sub
        valid: 0 <= h_in < H_IN
    
    Per-phase weight stored as [IC, OC, 2, 2] where dim2=dh_sub, dim3=dw_sub.
    """
    pid_n_oc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    OC_TILES = (OC + BLOCK_OC - 1) // BLOCK_OC
    pid_n = pid_n_oc // OC_TILES
    pid_oc = pid_n_oc % OC_TILES
    
    ph = PHASE // 2
    pw = PHASE % 2
    
    H_SUB = H_IN + 1
    W_SUB = W_IN + 1
    
    hw_start = pid_hw * BLOCK_HW
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (H_SUB * W_SUB)
    
    hi_grid = offs_hw // W_SUB  # [BLOCK_HW]
    wi_grid = offs_hw % W_SUB
    
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC
    
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)
    
    # Loop over 2x2 sub-taps
    for dh_sub in tl.static_range(0, 2):
        h_in = hi_grid - dh_sub  # [BLOCK_HW]
        h_valid = (h_in >= 0) & (h_in < H_IN)
        
        for dw_sub in tl.static_range(0, 2):
            w_in = wi_grid - dw_sub
            w_valid = (w_in >= 0) & (w_in < W_IN)
            spatial_valid = h_valid & w_valid & mask_hw
            
            # x base offset for this (n, h_in, w_in), iterating over ic
            # x layout: [N, IC, H_IN, W_IN]
            x_spatial_off = pid_n * (IC * H_IN * W_IN) + h_in * W_IN + w_in  # [BLOCK_HW]
            
            # w base offset for this tap: weight layout [IC, OC, 2, 2]
            # w[ic, oc, dh_sub, dw_sub]
            w_tap_off = dh_sub * 2 + dw_sub  # scalar
            
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                mask_ic = offs_ic < IC
                
                # Load x: [BLOCK_HW, BLOCK_IC]
                x_ptrs = x_ptr + x_spatial_off[:, None] + offs_ic[None, :] * (H_IN * W_IN)
                x_mask = spatial_valid[:, None] & mask_ic[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
                
                # Load w: [BLOCK_IC, BLOCK_OC]
                # weight layout [IC, OC, 4]: ic * OC*4 + oc * 4 + w_tap_off
                w_ptrs = w_ptr + offs_ic[:, None] * (OC * 4) + offs_oc[None, :] * 4 + w_tap_off
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
                
                acc += tl.dot(x_vals, w_vals)
    
    # Epilogue: + bias + add_value, min(.,0), GELU, * multiply
    bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :] + add_value
    acc = tl.minimum(acc, 0.0)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))
    out = gelu * multiply_value
    
    # Store into output at actual coords
    # h_out = 2*hi_grid + ph, w_out = 2*wi_grid + pw
    h_out = 2 * hi_grid + ph
    w_out = 2 * wi_grid + pw
    out_base = pid_n * (OC * H_OUT * W_OUT) + offs_oc[None, :] * (H_OUT * W_OUT) + h_out[:, None] * W_OUT + w_out[:, None]
    store_mask = mask_hw[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_base, out, mask=store_mask)


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
        
        # Precompute per-phase weight layout. Original weight [IC, OC, KH, KW] = [IC, OC, 4, 4]
        # For each PHASE = (ph, pw), the weight taps are:
        #   kh = ph + 2*dh_sub, kw = pw + 2*dw_sub, dh_sub, dw_sub in {0,1}
        # We pack a tensor [4 phases, IC, OC, 2, 2] then flatten last two to 4.
        # Stored as [4, IC*OC*4] contiguous.
        self._prepare_weights()
    
    def _prepare_weights(self):
        with torch.no_grad():
            w = self.conv_transpose.weight.data  # [IC, OC, KH, KW]
            IC, OC, KH, KW = w.shape
            # Build [4, IC, OC, 2, 2]
            phase_w = torch.empty((4, IC, OC, 2, 2), dtype=w.dtype, device=w.device)
            for phase in range(4):
                ph = phase // 2
                pw = phase % 2
                for dh_sub in range(2):
                    for dw_sub in range(2):
                        kh = ph + 2 * dh_sub
                        kw = pw + 2 * dw_sub
                        if kh < KH and kw < KW:
                            phase_w[phase, :, :, dh_sub, dw_sub] = w[:, :, kh, kw]
                        else:
                            phase_w[phase, :, :, dh_sub, dw_sub] = 0.0
            # Flatten [2,2] -> [4], layout [4, IC, OC, 4]
            phase_w = phase_w.reshape(4, IC, OC, 4).contiguous()
        self.register_buffer('phase_weight', phase_w, persistent=False)
        self._cached_weight_version = self.conv_transpose.weight._version
    
    def _get_phase_weight(self):
        w = self.conv_transpose.weight
        if w._version != self._cached_weight_version or self.phase_weight.device != w.device:
            self._prepare_weights()
        return self.phase_weight
    
    def forward(self, x):
        N, IC, H_IN, W_IN = x.shape
        OC = self.out_channels
        KH = 4
        KW = 4
        STRIDE = 2
        
        H_OUT = (H_IN - 1) * STRIDE + KH  # = 2*H_IN + 2
        W_OUT = (W_IN - 1) * STRIDE + KW  # = 2*W_IN + 2
        
        x = x.contiguous()
        phase_w = self._get_phase_weight()
        bias = self.conv_transpose.bias.contiguous()
        
        out = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)
        
        H_SUB = H_IN + 1
        W_SUB = W_IN + 1
        
        BLOCK_HW = 64
        BLOCK_OC = 64
        BLOCK_IC = 32
        
        OC_TILES = (OC + BLOCK_OC - 1) // BLOCK_OC
        HW_TILES = (H_SUB * W_SUB + BLOCK_HW - 1) // BLOCK_HW
        
        grid = (N * OC_TILES, HW_TILES)
        
        for phase in range(4):
            phase_conv_kernel[grid](
                x, phase_w[phase], bias, out,
                N, IC, OC,
                H_IN, W_IN,
                H_OUT, W_OUT,
                self.add_value, self.multiply_value,
                BLOCK_HW=BLOCK_HW,
                BLOCK_OC=BLOCK_OC,
                BLOCK_IC=BLOCK_IC,
                PHASE=phase,
                num_warps=4,
                num_stages=3,
            )
        
        return out