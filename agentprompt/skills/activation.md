# activation skills

Covers GELU, Sigmoid, Tanh, Mish, ReLU/LeakyReLU, SiLU, Softplus, ELU.

## Design (large step)

- Do not run an activation as a standalone Triton kernel in the final design —
  fuse it into the kernel that produces its input (conv-bias-ReLU,
  matmul-bias-GELU, norm-activation). A standalone activation kernel is
  memory-bound and only justified as an early starter sketch.
- Apply the activation in the producer's epilogue, on the accumulator, before
  the store — this avoids an extra full read/write of the activation tensor.
- Use the closed-form for the activation (e.g. `x * sigmoid(x)` for SiLU,
  `0.5 * x * (1 + erf(x / sqrt(2)))` or the tanh approximation for GELU) so it
  stays a few flops on data already in registers.

## Tuning (small step)

- If you must keep a standalone activation kernel, vectorize loads
  (`BLOCK_SIZE` elements per program via `tl.constexpr`) and mask the tail.
- Pick `num_warps` (4 or 8) to saturate memory bandwidth — the kernel is
  bandwidth-bound, so larger blocks with fewer launches usually win.
- Keep `num_stages` low (2); there is little compute to overlap.
- Process the contiguous flattened tensor so every load coalesces.
