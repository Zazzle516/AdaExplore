# norm skills

Covers BatchNorm1d/2d/3d, GroupNorm, LayerNorm, InstanceNorm1d/2d/3d, RMSNorm.

## Design (large step)

- BatchNorm / InstanceNorm: at eval the running mean/var are constant — fold
  the normalization plus affine into a single scale/shift applied in one
  elementwise kernel (or into the producing kernel's epilogue). Use Welford
  only when stats must be computed at runtime.
- LayerNorm / GroupNorm: two-pass (compute mean, then variance) over the
  normalized axis inside one program; keep the whole reduction in-kernel so it
  is not split across launches.
- RMSNorm: skip the mean, normalize by `rsqrt(mean(x^2) + eps)`.
- Fuse the normalization with the adjacent activation, and fold a pure affine
  into a preceding Linear/Conv only when it is genuinely a pure affine.

## Tuning (small step)

- Use one program per normalization group — per `(N, C)` for LN, per
  `(N, group)` for GN, per channel for BN — and block the reduction axis with
  `BLOCK_SIZE` over the feature dim.
- For small feature dims, use a persistent kernel that keeps the row resident
  rather than re-loading; for large dims, tile the reduction and combine.
- Tune `num_warps` to the reduction width (4 for short rows, 8 for wide) and
  `num_stages` 2-3.
- Load the normalized axis contiguously; reorder to `channels_last` when the
  stat axis would otherwise be strided.
