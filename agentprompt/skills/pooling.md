# pooling skills

Covers MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d.

## Design (large step)

- One program per `(N, C, output-spatial tile)`; gather the pooling window
  with coalesced strided reads and reduce (max or mean) in registers.
- Combine the pool with an adjacent activation or scalar op in the same kernel
  epilogue rather than launching separately.
- AdaptiveAvgPool: the input→output index map depends only on the static
  shapes — precompute the per-output-element window bounds at init and pass
  them in, instead of recomputing the division inside the kernel.
- For global average pooling (output spatial size 1), treat it as a reduction
  over the full spatial axis (see reduction skills).

## Tuning (small step)

- Tune `BLOCK` over the output-spatial tile so it matches the kernel/stride
  size; align reads so each window row is contiguous.
- Vectorize the strided window loads and mask the padded border instead of
  branching.
- Set `num_warps` (4 or 8) to the window width and keep `num_stages` 2.
- Use `channels_last` when it makes the pooled channel axis contiguous.
