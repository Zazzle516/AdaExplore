# reduction skills

Covers sum, mean, min, max, prod, softmax, log_softmax, logsumexp,
argmax, argmin.

## Design (large step)

- Block-level reduction: one program per output element (or output tile),
  reducing its slice of the input with a shared-memory / register tree reduce
  over `BLOCK_SIZE` chunks of the reduction axis.
- softmax / log_softmax / logsumexp: use the numerically stable form — compute
  the running max, subtract it before `exp`, then normalize (online softmax in
  a single pass when the row fits, two-pass otherwise).
- Fuse the reduction with the op that produces its input and with any scalar
  scaling applied to the result, so the reduced tensor is never written out
  just to be read back.
- Keep every operator in the chain executing at runtime — do not fold a
  reduction into a weight rewrite (see safety contract).

## Tuning (small step)

- Tune `BLOCK_SIZE` over the reduction axis (powers of two); for a reduction
  dim that fits, use a persistent kernel that keeps the row resident.
- Size `num_warps` to the reduction width (4 for short, 8 for wide rows) and
  keep `num_stages` 2-3.
- Lay the reduction axis out contiguously so the strided combine is avoided;
  one program per output element keeps the grid simple and coalesced.
