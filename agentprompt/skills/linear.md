# linear skills

Covers Linear, matmul, bmm, einsum, addmm.

## Design (large step)

- Implement as a tiled GEMM with `tl.dot`: one program per `(M tile, N tile)`,
  accumulating over the K dimension in `BLOCK_K` chunks.
- Fuse the bias add and any downstream activation (ReLU, GELU, SiLU) into the
  accumulator epilogue before the store — never launch a separate kernel for
  the bias or activation.
- For `bmm` / batched `matmul`, add the batch index to the program grid and
  offset the A/B base pointers per batch.
- Absorb a trailing scalar multiply or affine into the epilogue; do not
  precompute a reduced weight that turns the GEMM into a matvec (see safety
  contract).

## Tuning (small step)

- Sweep tile shapes `BLOCK_M` / `BLOCK_N` / `BLOCK_K` for `tl.dot` (common:
  128/128/32, 64/64/32, 128/64/64) sized to the actual `(M, N, K)`.
- Add a `GROUP_M` block-swizzle (e.g. 8) so program scheduling reuses L2
  across the N tiles of a row group.
- Tune `num_warps` (4 or 8) and `num_stages` (3-4 for large K to overlap the
  K-loop loads) and expose them via `triton.autotune`.
- Keep the contiguous operand axis on the inner load; transpose B once up
  front rather than strided-loading it in the hot loop.
