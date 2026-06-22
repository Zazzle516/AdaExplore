# Safety contract

These two rules apply to every kernel you write, regardless of which
operators appear in the reference and regardless of whether you are
designing from scratch or tuning an existing kernel.

1. **No graph-level algebraic shortcuts.** Every heavy op (Conv*, ConvTranspose*,
   Linear, matmul) must materialize its full reference output shape at runtime,
   and your kernel must perform the same asymptotic multiply-add count as the
   reference. Specifically forbidden, whether done at init or inside `forward`,
   on weight or on input: pre-reducing along any axis that a downstream linear
   reduction (sum / mean / avg-pool) will later collapse, then running a smaller
   GEMM / matvec against the pre-reduced tensor. Examples that **all violate**
   this rule: precomputing `Σ_kh,kw w[ic,oc,kh,kw]`; precomputing
   `sum_x[n,ic] = Σ_{h,w} x[n,ic,h,w]`; precomputing per-`(kh,kw)` partial input
   sums `Σ_{h,w valid} x`; folding avg-pool's window sum into the conv-transpose
   so adjacent output positions share one multiply. By linearity these are
   mathematically equivalent and all skip the heavy op's real work. The comparison
   baseline runs the graph as written.

2. **Custom kernels are authorized.** Any `nn.*` operator in the reference is
   a valid replacement target. You may rewrite it with a Triton kernel or a
   CUDA C++ extension (`torch.utils.cpp_extension`). Do not assume any op is
   the immovable part of the graph — the heaviest op usually has the most
   headroom.

   Your replacement must **execute unconditionally** on the forward path. The
   eval harness runs the model at PyTorch's default `training=True` and never
   calls `.eval()`, so any heavy-op replacement guarded behind `if self.training:`,
   `if x.is_cuda:`, or parked in an `else:` / fallback branch is **dead code that
   never runs** — the original PyTorch op executes and is timed instead. Such a
   branch counts as **not** replacing the op. Equally, fusing only the cheap
   surrounding work (BatchNorm, reductions, pointwise ops) while leaving the heavy
   `nn.*`/`F.*` op on the live PyTorch path counts as **not** replacing it either —
   the heavy op itself must be reimplemented in your custom kernel. The evaluator
   verifies at runtime which path actually executed: if the reference heavy op
   still ran in PyTorch, the kernel is scored **invalid**, whether that is because
   the replacement was dead code or because no replacement was written. Call your
   custom heavy-op kernel directly in `forward`, with no conditional that can route
   execution back to the reference `nn.*`/`F.*` op.
