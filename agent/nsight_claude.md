# Mandatory heavy-op rewrite when `fast_p < 0.8`

## Context

The evaluator currently emits next-step guidance as "optional" and the
proposer's `OPTIONAL_CONTEXT` literally says *"ignore them if a different
approach is better."* This is too weak when the proposed kernel is slower
than the PyTorch baseline (`fast_p < 0.8`) AND it kept the reference heavy
operator (`nn.Conv*` / `nn.ConvTranspose*` / `nn.Linear` / `F.conv*` /
`F.linear`) intact. In that situation, the bottleneck IS the kept op, and
tuning around it cannot close the gap.

Worked example: `outputs/KB-l2_AdaExplore_50/2_15`. Reference is
ConvTranspose3d → BN3d → subtract-spatial-mean. `step_5` followed the
evaluator's advice and wrote a `_convt3d_gather_kernel`, but failed
correctness by `max_diff=0.003` (likely a small indexing bug). MCTS gave it
reward 0; every step from 6 to 50 reverted to leaving `nn.ConvTranspose3d`
in place and only fusing BN+sub. Final best speedup: 0.32×.

This change makes "rewrite the heavy op" a mandate — propagated through the
evaluator prompt, the parsed `direction`, the proposer's context wrapper,
and MCTS routing — so a single failed attempt does not collapse the
search. A placeholder `## Profiler Summary` section is added in the same
pass so a future Nsight hook can fill it without touching prompt code.

## Trigger condition

The mandate fires when ALL of:
- `run_info.correctness == True` (failure path already routes to
  `COMPILE_FAILURE` and `<direction>large</direction>`),
- `0 < fast_p < 0.8`,
- `detect_heavy_op_intact(arc_src, kernel) != []`.

It is mutually exclusive with the existing `adversarial = ratio >= 5.0` gate
at `agentprompt/evaluator_prompt.py:213`.

## Changes

### 1. New detector — `agentprompt/Utils/detect.py`

Add `detect_heavy_op_intact(ref_arch_src, proposed_kernel_src) -> list[str]`
after `detect_shortcut_risk` (~L284). Reuse the existing AST helpers
(`_find_model_class`, `_build_init_map`, `_resolved_call_name`,
`_classify`, `_HEAVY_CONV`, `_HEAVY_LINEAR`) to collect heavy-op names from
the reference's `forward`. Then regex-scan the proposed kernel for
`nn\.(Conv[123]d|ConvTranspose[123]d|Linear)\s*\(` and
`F\.(conv[123]d|conv_transpose[123]d|linear)\s*\(`. Return the intersection
formatted as `"nn.ConvTranspose3d"` / `"F.linear"`. Empty/malformed sources
return `[]`. Pattern matches construction or call — an unreferenced
`nn.X` in `__init__` is a tolerated false positive; cost is one extra
"rewrite" prompt, not a correctness issue (documented in docstring).

### 2. Evaluator prompt — `agentprompt/evaluator_prompt.py`

Add two new templates after `STRUCTURAL_ALERT` (L177):

```python
MANDATORY_REWRITE = """## Mandatory rewrite required

The previous kernel kept {heavy_ops_list} from the reference intact while
its measured fast_p={ratio:.3f} is below 0.80. Tuning around the kept
PyTorch operator cannot close this gap -- the heavy op IS the bottleneck.

You MUST:
  1. Emit <direction>large</direction>.
  2. In <large_guidance>, prescribe a concrete Triton implementation
     strategy that REPLACES {heavy_ops_list} (not wraps it). For
     ConvTranspose*d: scatter-add over input voxels or gather over
     output voxels with explicit stride/padding indexing. For Conv*d:
     tiled im2col-GEMM or direct cross-correlation with a K-loop.
     For Linear: split-K GEMM with tl.dot on per-block tiles.
  3. If the prior attempt at this op was near-correct (small max_diff),
     name the indexing axis most likely responsible and how to fix it.

Prior attempt diagnostics (if any): {prior_diag}

"""

PROFILER_SUMMARY = """## Profiler Summary

{profile_summary}

"""
```

Wire-in in `generate_evaluator_prompt`, between the `STRUCTURAL_ALERT`
loop (L231) and `prompt += TASK_INSTRUCTION.format(...)` (L233):

```python
heavy_ops_intact = []
if correctness and 0 < ratio < 0.8:
    heavy_ops_intact = detect_heavy_op_intact(
        task_params.get("arc_src"),
        format_dict.get("custom_triton_kernels"),  # stripped form is fine
    )
    if heavy_ops_intact:
        prior_diag = (
            (run_info.metadata.get("max_difference")
             or run_info.metadata.get("prior_failure_summary")
             or "none")
            if isinstance(run_info, KernelExecResult) else "none"
        )
        prompt += MANDATORY_REWRITE.format(
            heavy_ops_list=", ".join(heavy_ops_intact),
            ratio=ratio,
            prior_diag=prior_diag,
        )

profile_summary = task_params.get("profile_summary") or (
    run_info.metadata.get("profile_summary")
    if isinstance(run_info, KernelExecResult) else None
)
if profile_summary:
    prompt += PROFILER_SUMMARY.format(profile_summary=profile_summary)
```

`generate_evaluator_prompt`'s signature stays unchanged — the parse-side
override in `actions.py` recomputes the same condition.

### 3. Parse-time direction override — `agent/actions.py`

Extend `run_evaluator` (L61) to return a 5-tuple:
`(small_guidance, large_guidance, direction, valid, mandatory_rewrite)`.

After the existing four extractions (L88), add:

```python
ratio = 0.0
if isinstance(metrics, KernelExecResult) and metrics.correctness:
    ratio = metrics.runtime_stats.get("fast_p", 0) or 0
heavy_ops_intact = (
    detect_heavy_op_intact(ref_arch_src, kernel)
    if 0 < ratio < 0.8 else []
)
mandatory_rewrite = bool(heavy_ops_intact)
if mandatory_rewrite:
    direction = "large"
return small_guidance, large_guidance, direction, valid, mandatory_rewrite
```

Import `detect_heavy_op_intact`. Extend `single_large_step` (L143) with a
`mandatory_rewrite: bool = False` kwarg and thread it into
`generate_proposer_prompt(..., mandatory_rewrite=mandatory_rewrite)`.

### 4. Proposer prompt — `agentprompt/proposer_prompt.py`

Add `MANDATORY_CONTEXT` next to `OPTIONAL_CONTEXT` (L81):

```python
MANDATORY_CONTEXT = """## Required design direction

This guidance is MANDATORY. The previous kernel kept the reference heavy
operator intact and measured fast_p < 0.80; the bottleneck IS that
operator. You MUST replace it with a custom Triton kernel -- tuning the
existing structure cannot close the gap. Apply the design directions
below verbatim unless you can identify a concrete bug in them.

{large_guidance}

"""
```

Add `mandatory_rewrite: bool = False` to `generate_proposer_prompt`. At
L115 replace the conditional with:

```python
if large_guidance:
    tmpl = MANDATORY_CONTEXT if mandatory_rewrite else OPTIONAL_CONTEXT
    prompt += tmpl.format(large_guidance=large_guidance)
```

### 5. MCTS node + routing — `agent/mcts.py`

- Add `mandatory_rewrite: bool = False` to `MCTSNode` (L66-72 dataclass).
- Update `_create_node` to accept and persist the flag.
- In `expand_large` (~L511) and `expand_small` (~L579), unpack the 5-tuple
  from `run_evaluator` and pass `mandatory_rewrite=` into the new node.
- In `expand_large`, pass `mandatory_rewrite=selected_node.mandatory_rewrite`
  into `single_large_step` so the proposer expanding *off* a mandated
  parent uses `MANDATORY_CONTEXT`.
- In `step` (L686-706), after the existing `direction_bias` block, add:

  ```python
  if getattr(selected_node, "mandatory_rewrite", False) \
          and selected_node.created_by != "dummy_root":
      p_large = 0.95
  ```

- In `_save_step_log` (~L800), include `"mandatory_rewrite":
  node.mandatory_rewrite` in the dict so replays carry the flag.

### 6. Small-loop interaction — `agent/small_loop.py`

The loop calls `run_evaluator` at L170/L194/L218. Unpack the new 5-tuple at
each site. Initialize `mandatory_rewrite = False` before the loop. At the
top of each iteration after the seed-kernel block, when
`mandatory_rewrite` is True, issue `single_large_step(..., mandatory_rewrite=True)`
instead of the planned small step, log, and `continue`.

### 7. Dummy paths

`dummy_small_step` / `dummy_large_step` (`actions.py:210-250`) skip
`run_evaluator`. Callers must default `mandatory_rewrite=False` when
constructing nodes from these — already covered by the dataclass default.

## Error-history carry-forward to the evaluator

### Motivation

The evaluator that emits all guidance only ever sees the single most recent
result: `generate_evaluator_prompt(run_info=...)` takes one `KernelExecResult`
and has no history parameter (`agentprompt/evaluator_prompt.py:179`). This is
the lossy signal behind the `2_15` collapse — `step_5`'s heavy-op rewrite was
*near-correct* (`max_diff=0.003`), but the only thing propagated forward was
MCTS reward 0. The evaluator could not say "you already tried this op and were
0.003 off — fix the indexing axis" because it never saw the prior attempt.

The fix: feed the evaluator an **error history** of prior attempts, accumulated
**along the current small-step path** and **reset whenever a large step is
taken**. A large step is a fresh approach, so diagnostics from the old approach
are stale.

### A.1 — MCTS (primary): derive the history from `get_path_to_cut()`

No new `MCTSNode` field and no explicit reset are needed. `get_path_to_cut`
(`agent/mcts.py:199`) already walks the parent chain and terminates at the most
recent `created_by == "large_step"` ancestor (loop at L210; the boundary
large-step node is included). So the path *is* the "same approach" window, and a
large step resets it for free:

- In `expand_small` (`agent/mcts.py:535`), the call site already computes
  `path = node.get_path_to_cut()` (L541). Build `error_history` from those nodes'
  `metrics` and pass it into `run_evaluator(..., error_history=...)`.
- In `expand_large` (`agent/mcts.py:468`), the new branch's path-to-cut is just
  the fresh node, so `error_history == []` — the reset, automatically.

Each entry is built from a path node's `KernelExecResult` (**all attempts, not
only failures**, per the chosen design):

```python
def _history_entry(ref_arch_src, node):
    m = node.metrics
    return {
        "correctness": getattr(m, "correctness", False),
        "max_diff": m.metadata.get("max_difference"),          # set on correctness-fail
        "error": (m.metadata.get("compilation_error_parsed")
                  or m.metadata.get("compilation_error")
                  or m.metadata.get("runtime_error")),
        "fast_p": (m.runtime_stats or {}).get("fast_p"),       # set when correct
        "heavy_ops_intact": detect_heavy_op_intact(ref_arch_src, node.kernel),
    }
```

The `heavy_ops_intact` tag (from change #1) is what lets the evaluator tell
"*tried to replace* the op, failed by `max_diff`" apart from "*kept* the op,
just slow" — the explicit per-op signal on top of the path scoping.

### A.2 — IRS `small_loop`: degenerate case (scope discussion)

The IRS loop (`agent/small_loop.py:run_small_loop`) takes **exactly one large
step** — the initial proposal at `previous_kernels == []` (L179-201) — then
**only small steps** for the rest of the run (L203-233). So "reset on large"
fires once at the start and is otherwise vacuous; the mechanism degenerates to a
**single accumulating window**.

The existing `previous_kernels` / `previous_metrics` window already carries
history to the *tuner*, but it is the wrong vehicle for the evaluator: (a)
`run_evaluator` is still called with only the latest single kernel
(L170 / L194 / L218), and (b) the window is capped at `max_memory_round` and
pops from the front (L224-226), so an early correctness failure drops out even
though it stays relevant to later guidance.

**Recommended (IRS light):** keep a separate `error_history` list in
`run_small_loop`, initialized `[]`, appended every step, cleared only on the
initial large-step branch, and **not** bounded by `max_memory_round` (or bounded
by a larger, separate cap). Pass it into the three `run_evaluator` calls. This
reuses the same evaluator `error_history` parameter; the only IRS-specific code
is a few lines of bookkeeping. **MCTS-only is an acceptable fallback** if the
IRS churn isn't judged worth it — the mechanism is designed so IRS can adopt it
independently and later.

### A.3 — Evaluator prompt wiring

Add `error_history: list = None` to `generate_evaluator_prompt`
(`agentprompt/evaluator_prompt.py:179`) and to `run_evaluator`
(`agent/actions.py:61`). Add an `ERROR_HISTORY` template (one compact line per
attempt):

```python
ERROR_HISTORY = """## Prior attempts on this approach (since the last redesign)

These attempts were all made *after* the most recent large-step redesign, so
they share the current approach. Use them to avoid repeating a dead end and to
target a near-miss fix instead of abandoning the approach.

{history_lines}

"""
# each line, e.g.:
#   - attempt 3 [kept nn.ConvTranspose3d]: correctness=False, max_diff=0.003
#   - attempt 4 [replaced nn.ConvTranspose3d]: correctness=True, fast_p=0.41
```

Append it in `generate_evaluator_prompt` after the `COMPILE_FAILURE` block
(`agentprompt/evaluator_prompt.py:250`) and before the `GOAL` / `ADVERSARIAL_GOAL`
tail (L252), guarded by `if error_history:`.

This **supersedes the single-value `prior_diag`** in `MANDATORY_REWRITE` (change
#2): `prior_diag` becomes a derived view of the same history — or
`MANDATORY_REWRITE` simply references the `ERROR_HISTORY` block — so the two do
not carry duplicate, divergent diagnostics.

### A.4 — Composition with the existing changes

`run_evaluator` now takes `error_history` as an **input** in addition to
returning the `mandatory_rewrite` 5-tuple **output** added in change #3. The
`expand_small` / `expand_large` updates in change #5 assemble the history from
`get_path_to_cut()` at the `run_evaluator` call site (the path is already in
hand in `expand_small`). The dummy paths (change #7) pass `error_history=None`.

## Verification

1. **Detector unit test** (inline `__main__` block in `detect.py`):
   - `arc_src` with `nn.ConvTranspose3d` + a kernel that keeps
     `self.conv_transpose = nn.ConvTranspose3d(...)` → `["nn.ConvTranspose3d"]`.
   - Same `arc_src` + a kernel that contains only `@triton.jit` / `tl.*` →
     `[]`.
   - Empty / malformed sources → `[]`.
2. **End-to-end replay** of `outputs/KB-l2_AdaExplore_50/2_15`: re-run from
   step 5. Confirm via `step_6_prompt.txt` that the proposer prompt now
   contains the `## Required design direction` header (MANDATORY_CONTEXT),
   and `step_6` calls `expand_large` (visible in tree.json / step log
   `created_by`). Confirm the global best speedup exceeds 0.8× — the
   success criterion is "the search no longer collapses onto the
   BN+sub-only branch."
3. **Regression**: re-run a problem where the best step had `fast_p >= 0.8`.
   Grep saved evaluator prompts for `Mandatory rewrite required` — must be
   absent.
4. **Negative case**: a kernel with `fast_p < 0.8` that already replaced
   the heavy op (pure Triton matmul replacing `nn.Linear`) — confirm the
   detector returns `[]`, the mandate does NOT fire, and `direction` is
   whatever the LLM emitted.
5. **Adversarial gate untouched**: a kernel with `fast_p >= 5.0` must
   still take the `ADVERSARIAL_GOAL` branch; verify the two conditions
   are mutually exclusive in the assembled prompt.
6. **Error-history reset/accumulate**: in a saved MCTS run, pick a node two
   small steps below a large step and confirm its evaluator prompt's
   `## Prior attempts on this approach` block lists exactly the attempts since
   that large step (not earlier ones) — i.e. `get_path_to_cut()` scoping holds.
   Then pick a fresh `large_step` node and confirm the block is **absent**
   (empty history → reset). Spot-check that a near-miss entry shows
   `correctness=False, max_diff=...` with the correct `heavy_ops_intact` tag.

## Out of scope (deferred follow-up)

- Actual Nsight / `ncu` capture in `src/eval.py`. The `## Profiler
  Summary` placeholder is added so the future hook only needs to set
  `KernelExecResult.metadata["profile_summary"]`.
- Per-attempt diagnostic carry-forward is now designed in *Error-history
  carry-forward to the evaluator* above (it subsumes the old single-value
  `prior_diag`). Only *richer per-axis signals* — which voxel index diverged,
  per-element error maps, etc. — remain deferred to a separate change.
