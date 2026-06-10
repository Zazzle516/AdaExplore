# conv skills

Covers Conv1d/2d/3d and ConvTranspose1d/2d/3d.

## Design (large step)

- Conv2d/3d: implement as im2col-style indexing into a tiled GEMM — one
  program per `(N, OC tile, output-spatial tile)`. The im2col gather can stay
  implicit (compute input offsets inside the kernel) to avoid materializing
  the unfolded tensor.
- ConvTranspose1d/2d/3d: implement as a scatter-add of the `input × kernel`
  outer product into the output grid, handling `stride` / `padding` /
  `output_padding` in the index math. Do not approximate it with a plain conv.
- Fold BatchNorm into the conv weight/bias at eval time (the BN running stats
  are constant), and absorb a trailing bias add and activation (ReLU, etc.)
  into the same kernel epilogue.
- When the torch convolution dominates runtime (symptom: `fast_p < 0.8` while
  the epilogue is already fused), replacing the conv with a custom kernel is
  the high-headroom move — not further tuning of the tail.

## Tuning (small step)

- Tune the GEMM tile `BLOCK_M` / `BLOCK_N` / `BLOCK_K` (start 64/64/32, sweep
  powers of two) against the `(OC, output-spatial, IC*kh*kw)` shapes.
- Set `num_warps` (4 or 8) and `num_stages` (2-4) per config; wrap in
  `triton.autotune` over a small grid.
- Use a `channels_last` layout so the channel axis is contiguous and input
  loads coalesce across the GEMM-K dimension.
- Hoist constant index math (stride/pad offsets) out of the inner loop and
  mask the spatial tail rather than branching.
