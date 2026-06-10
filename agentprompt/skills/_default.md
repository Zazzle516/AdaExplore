# generic skills

Generic toolkit for operators without a dedicated skill file. Use alongside
any family skills that were detected.

## Design (large step)

- Identify the heaviest operator in the reference forward (largest tensor it
  touches, or the one with the most FLOPs) and make that your primary kernel
  target — the epilogue elementwise ops are cheap by comparison.
- Fuse adjacent elementwise / broadcast ops (bias add, scalar multiply,
  activation) into the producing kernel rather than launching a separate
  kernel per op.
- Pick one program per independent output tile so the work decomposes cleanly
  across the grid; keep the reduction axis inside a single program when a
  reduction is involved.
- Apply standard inference-time folds where they preserve the op: BatchNorm
  into a preceding affine at eval, scalar multipliers absorbed into adjacent
  affine ops, identity activations elided.

## Tuning (small step)

- Sweep `BLOCK_SIZE` (and per-axis `BLOCK_*` tiles) over powers of two; match
  the tile to the contiguous axis so loads coalesce.
- Tune `num_warps` (4 or 8 are the common sweet spots) and `num_stages` (2-4)
  for the dominant kernel; wrap the launch in `triton.autotune` over a short
  config list.
- Make the innermost loaded axis the contiguous one; insert `.contiguous()` or
  a `channels_last` layout only when it removes a strided load in the hot loop.
- Vectorize loads (process `BLOCK_SIZE` elements per program with
  `tl.constexpr` bounds) and mask the tail instead of branching.
