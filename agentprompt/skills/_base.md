# Safety contract

These two rules apply to every kernel you write, regardless of which
operators appear in the reference and regardless of whether you are
designing from scratch or tuning an existing kernel.

1. **No graph-level algebraic shortcuts.** Every operator in the reference
   forward must execute at runtime on the actual input tensor. Do **not**
   collapse a heavy op into a downstream reduction at init time (for example,
   precomputing a column/row sum of a weight so that forward runs a matvec
   instead of the full operator). These rewrites pass the loose
   `atol=rtol=5e-2` correctness check but are not the optimization target —
   the comparison baseline runs the graph as written.

2. **Custom kernels are authorized.** Any `nn.*` operator in the reference is
   a valid replacement target. You may rewrite it with a Triton kernel or a
   CUDA C++ extension (`torch.utils.cpp_extension`). Do not assume any op is
   the immovable part of the graph — the heaviest op usually has the most
   headroom.
