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
- Pick tiles that fit the GPU's shared-memory budget on the **first** attempt
  — otherwise the Triton launcher's gcc build fails as a
  `subprocess.CalledProcessError` and the rewrite is silently dropped. For
  {dtype} on {gpu_label} (~{smem_kb_per_sm} KB SMEM per SM) the operand tiles must satisfy
  `(BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N) · {dtype_bytes} · num_stages ≲ {smem_budget_kb} KB`. Tiles like
  `BLOCK_M=BLOCK_N=128, BLOCK_K=32, num_stages=2` overflow this and will not
  load. Wrap the kernel in `@triton.autotune` from the first emission, keyed
  on `(OC, output-spatial, IC*KH*KW)`, with a safe starting grid such as
  `(BLOCK_M, BLOCK_N, BLOCK_K) ∈ {(32,32,16), (64,32,16), (32,64,16), (64,64,32)}`
  at `num_warps=4`, `num_stages=2`. Larger tiles can be added once a smaller
  variant in the same autotune list is known to compile.

## Tuning (small step)

- Tune the GEMM tile `BLOCK_M` / `BLOCK_N` / `BLOCK_K` against the
  `(OC, output-spatial, IC*kh*kw)` shapes, but verify each config fits the
  SMEM budget — see the Design block's
  `(BM·BK + BK·BN) · {dtype_bytes} · num_stages ≲ {smem_budget_kb} KB` heuristic for {dtype} on {gpu_label}. If a
  previous attempt failed with a `subprocess.CalledProcessError` from the
  Triton launcher gcc step, the cause is almost always an oversized tile;
  drop to the safe starting grid before sweeping upward.
- Set `num_warps` (4 or 8) and `num_stages` (2-4) per config; wrap in
  `triton.autotune` over a small grid. Higher `num_stages` multiplies the
  SMEM cost — keep it at 2 until smaller tiles are confirmed to compile.
- Use a `channels_last` layout so the channel axis is contiguous and input
  loads coalesce across the GEMM-K dimension.
- Hoist constant index math (stride/pad offsets) out of the inner loop and
  mask the spatial tail rather than branching.
