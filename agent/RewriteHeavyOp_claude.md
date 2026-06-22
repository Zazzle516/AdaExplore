# Mode A escape hatch — stop forcing redesign once the heavy op is genuinely replaced (runtime-verified)

## Context

Mode A ("slow", `correct and 0 < fast_p < 0.8`) assumes the retained PyTorch
heavy op is the bottleneck, so it forces `direction="large"` and emits
redesign-only guidance — steering MCTS to keep calling the proposer for a
from-scratch rewrite. The failure we want to bound: a problem can sit in Mode A
indefinitely, forcing redesign after redesign even when redesign has already
been genuinely attempted and is still slow (e.g. `2_15` step 1 — a real Triton
`ConvTranspose3d` that runs unconditionally yet lands at `fast_p≈0.11` because
hand-written Triton cannot beat cuDNN).

We add a gated **escape hatch**: stop forcing redesign once the agent has
*actually* replaced the heavy op in Triton **on the path that runs** and is still
slow — at that point fall back to the normal `GOAL` mode so tuning (small steps)
becomes reachable again.

### Detection — reuse the runtime execution signal (already implemented)

The dead-branch work already added runtime ground truth: `src/eval.py` records
which heavy aten ops dispatched during `ModelNew`'s measured forward and writes
`metadata["heavy_op_executed_in_pytorch"]`. Combined with the static
`replaced_heavy_op_built` gate (a Triton heavy-op kernel is defined in source),
this *already* tells us whether a replacement genuinely executed — **no new AST
detector is needed**:

    genuine_replacement = replaced_heavy_op_built(kernel, ref_arch_src) \
                          and not metadata.get("heavy_op_executed_in_pytorch")

### Two constraints (both required to exit Mode A)

1. **Current kernel genuinely replaced the heavy op** — `genuine_replacement`
   (above) is True for the kernel under evaluation.
2. **A recent large_step also genuinely replaced it** — scanning the last 5
   entries of `self.all_nodes` (global chronological order), at least one *other*
   node with `replace_large_success == True` (the per-node field defined below).
   Filtering on `created_by == "large_step"` alone is **not** sufficient: that
   only says a redesign was *attempted*, not that it *succeeded* — a large step
   can still emit a kept-op or dead-branch kernel that never replaced the op. The
   node stores the verified `genuine_replacement` result at creation time, so the
   scan reads a stamped boolean instead of re-deriving it from each node's kernel.

### Per-node field: `replace_large_success`

`MCTSNode` gains `replace_large_success: bool = False`, stamped once when the node
is created:

    replace_large_success = (created_by == "large_step")
                            and genuine_replacement(kernel, metrics, ref_arch_src)

So it is True only for a large_step whose Triton replacement genuinely executed;
small steps, root, failed/kept-op/dead-branch large steps all stay False.
Constraint #2 is then `any(n.replace_large_success for n in self.all_nodes[-5:])`.
Because `redesign_exhausted` is computed *before* the current node is appended,
the scan sees only prior nodes — the "*other* node" requirement falls out for
free.

Only when **both** hold does Mode A release to `GOAL`. (Constraint #2 is the
conservative guard: it requires redesign to have been genuinely attempted more
than once before tuning is unlocked. If you would rather release a single honest
rewrite to tuning immediately, drop #2 and gate on #1 alone — see the `2_15`
note below for the consequence.)

### Locked design decisions

- **Detector = runtime execution signal (ground truth).** Reuses
  `heavy_op_executed_in_pytorch` (from the dead-branch recorder) plus the static
  `replaced_heavy_op_built`. No `self.training`-only AST heuristic — all guard
  forms are handled because the recorder sees what actually dispatched.
- **Both constraints required** to exit (conservative — stays in Mode A longer).
- **`2_15` stays in Mode A — correctly.** Its cheat kernels (2/4/5/9) keep the
  conv on the live path (`heavy_op_executed_in_pytorch == True`) so
  `genuine_replacement` is False; its kept-op kernels (3/7) never build a Triton
  conv so `replaced_heavy_op_built` is False. Step 1 *does* satisfy constraint #1,
  but it is the **only** genuine replacement in the run, so constraint #2 (a
  *second* genuine large_step in the window) is never met and the gate stays
  closed. This is a **general anti-stuck mechanism**, not a `2_15`-specific patch.

## Files modified

| File | Change |
|------|--------|
| `agentprompt/evaluator_prompt.py` | `generate_evaluator_prompt` gains optional `redesign_exhausted: bool=False`. In the mode-selection block, when the slow gate matches **and** `redesign_exhausted` is True, choose `"default"` instead of `"slow"`. `dead_branch_rewrite` still forces `"slow"` (mutually exclusive with `redesign_exhausted`). Return tuple unchanged `(prompt, mode)`. |
| `agent/actions.py` | Factor out a module-level helper `genuine_replacement(kernel, metrics, ref_arch_src) -> bool` = `bool(metrics) and replaced_heavy_op_built(kernel, ref_arch_src) and not (getattr(metrics,"metadata",{}) or {}).get("heavy_op_executed_in_pytorch")`. `run_evaluator`'s dead-branch line and the escape-hatch constraint #1 both call it (mcts imports it too). `run_evaluator` gains optional `redesign_exhausted: bool=False` and forwards it. The `if mode == "slow": direction="large"` force is untouched. 5-tuple return unchanged. |
| `agent/mcts.py` (node) | `MCTSNode` gains `replace_large_success: bool = False`. `_create_node` gains a `replace_large_success: bool = False` param and stores it on the node. |
| `agent/mcts.py` (steps) | At the **large_step** site (`:526`): compute `current_genuine = genuine_replacement(proposal_kernel, proposal_metrics, self.ref_arch_src)` **before** calling `run_evaluator`; pass `redesign_exhausted = current_genuine and any(n.replace_large_success for n in self.all_nodes[-5:])` into `run_evaluator`, and pass `replace_large_success=current_genuine` into `_create_node`. At the **small_step** site (`:587`): same `redesign_exhausted` computation, but `replace_large_success=False` (a small step is never a redesign). The `self.all_nodes` scan sees only prior nodes because the new node is appended *after* this. |

**Unchanged:** the 4 loop call sites (`large_loop.py:153`, `small_loop.py:170/197/223`)
keep calling `run_evaluator` without the new arg → `redesign_exhausted=False` →
Mode A behavior there is exactly as today (these loops keep no per-step type, and
`large_loop` already ignores `direction`).

## Computing `redesign_exhausted`

1. **Constraint #1 (current kernel):** `genuine_replacement(kernel, metrics, ref_arch_src)`
   = `replaced_heavy_op_built(kernel, ref_arch_src)` AND
   `not metrics.metadata.get("heavy_op_executed_in_pytorch")`. Computed at the
   step site before `run_evaluator`; for a large step this same value is stamped
   onto the new node as `replace_large_success`.
2. **Constraint #2 (history):** `any(n.replace_large_success for n in self.all_nodes[-5:])`
   — a stamped boolean read, no re-derivation. The new node is appended after this,
   so the window holds only prior nodes.
3. **Fail safe:** missing metadata / parse error → `genuine_replacement` returns
   **False** → `redesign_exhausted=False` (stays in Mode A — conservative). Past
   nodes that errored at creation simply carry `replace_large_success=False`.

## Verification

Read-only / local (no GPU):

1. **Import + arity:** `python -c "import agent.actions, agent.mcts, agentprompt.evaluator_prompt, agentprompt.Utils.detect"`;
   `grep -rn "run_evaluator(" agent/` → 6 sites, all still unpack 5 values.
2. **`genuine_replacement` probes** (real `2_15` kernels + their metrics):
   - step 1 (`heavy_op_executed_in_pytorch: false`, builds a Triton conv) → **True**.
   - steps 2/4/5/9 (`heavy_op_executed_in_pytorch: true`) → **False**.
   - steps 3/7 (`replaced_heavy_op_built` False) → **False**.
3. **Mode-selection probe:** `generate_evaluator_prompt` with synthetic correct +
   `fast_p=0.4`: `redesign_exhausted=False` → `mode=="slow"` (SLOW_GOAL present,
   unchanged); `redesign_exhausted=True` → `mode=="default"` (GOAL present,
   SLOW_GOAL absent).
4. **mcts gate probe:** stub `all_nodes` with a node carrying
   `replace_large_success=True` + a current genuine-replacement kernel →
   `redesign_exhausted` True; set the prior node's flag False, or make the current
   kernel non-genuine → False (both-required semantics). Also assert a node created
   from a kept-op / dead-branch large step gets `replace_large_success=False`.
5. **`2_15` regression (documented):** only step 1 ever gets
   `replace_large_success=True`, and when step 1 itself is evaluated no *prior*
   node carries the flag, so constraint #2 fails everywhere ⇒ `redesign_exhausted
   == False` across all `2_15` steps ⇒ stays in Mode A — expected, not a failure.
6. **Live smoke (optional, GPU):** a problem where the agent genuinely replaces a
   Conv/Linear on the live path across two large steps and stays <0.8× → confirm
   the evaluator switches from SLOW_GOAL to GOAL and a small step becomes reachable.

## Risks

- **Recorder error → `heavy_op_executed_in_pytorch` defaults False.** Bounded by
  the `replaced_heavy_op_built` AND (escape needs a *built* Triton heavy op, so a
  bare recorder failure on a kept-op kernel cannot trigger it) and by constraint #2
  (a second genuine large_step). Worst case is an early release to tuning, not a
  correctness or scoring error.
- **Detector false-negative → stays in Mode A:** identical to today's behavior, no
  regression.
- **all_nodes window:** last-5 is global chronological; node 1 (always a
  `large_step`) ages out of the window naturally, so the gate is not dominated by
  the forced first step.

---

# Runtime verification — invalidate any kernel whose heavy op still runs in PyTorch (not just the dead-branch cheat)

## Context

The eval harness `src/eval.py` instantiates both `Model` and `ModelNew` and
**never calls `.eval()`** (verified: no `.eval()`/`.training` anywhere in
`eval.py`), so both run at PyTorch's default `training=True`.

**Policy (this revision).** The heavy op (Conv*, ConvTranspose*, Linear, matmul)
**must** be replaced by a custom Triton implementation that actually executes. If
the reference heavy op still dispatched through PyTorch/aten during the measured
forward, the kernel did **not** replace it — and the result is forced to
**invalid**, regardless of intent. This is no longer a cheat detector; it is a
hard requirement. Three previously-distinct cases now collapse to one verdict:

- **Dead-branch cheat** — built a Triton conv, parked it in a never-taken branch
  (`else:` of `if self.training:`, an `if x.is_cuda:` guard, an aliased fallback).
  Conv runs in PyTorch → **invalid**.
- **Kept-op kernel** — never built a Triton conv at all, just fused the
  surrounding BN/reduction and left `self.conv_transpose(x)` on the live path.
  Conv runs in PyTorch → **invalid** (this is the behavior change: these were
  previously *valid* and sat in Mode A at `fast_p≈0.32`).
- **Genuine replacement** — a `@triton.jit` heavy op that runs unconditionally;
  the reference op does **not** dispatch through aten → **valid** (e.g.
  `2_15`'s slow but honest Triton ConvTranspose3d).

`outputs/KB-l2_AdaExplore_50/2_15` is the motivating run: of its correct kernels,
only the one honest Triton ConvTranspose3d has `heavy_op_executed_in_pytorch ==
False`; every other correct kernel (whether a dead-branch cheat or an honest
kept-op BN fusion) kept the conv in PyTorch and is now invalid.

The mere presence of a Triton kernel in source no longer earns credit. This change
keeps the **runtime verification** of the executed path, **gates on the runtime
signal alone**, **feeds back** corrective guidance (with the message tailored by a
secondary static check), and **hardens** the proposer/tuner prompts. The same
runtime signal feeds the Mode-A escape hatch's `genuine_replacement` check.

Locked decisions: detection = **runtime verification** (primary, ground truth);
gate = **runtime signal alone** (`heavy_op_executed_in_pytorch` ⇒ invalid);
the static `replaced_heavy_op_built` check is retained **only** to pick the
feedback wording (dead-branch vs. never-built); **harden** proposer/tuner prompts.

## Discriminator

The validity gate now reads **one** signal:

- **Runtime (ground truth):** during `ModelNew`'s measured forward
  (`training=True`), did the reference heavy op still dispatch through aten? Triton
  kernel launches do **not** go through `__torch_dispatch__`, so recording
  dispatched aten op names against a heavy-op allowlist reveals whether the heavy
  op truly ran in PyTorch. `heavy_op_executed_in_pytorch = ref_heavy_nonempty AND
  ref_heavy ⊆ executed_aten_ops` (already computed in `src/eval.py`).

`heavy_op_not_replaced = heavy_op_executed_in_pytorch` → **invalid**.

The static `replaced_heavy_op_built(src)` check is **no longer part of the gate**.
It is consulted only *after* the gate fires, to choose which corrective message to
emit: built-but-dead (`replaced_heavy_op_built == True`) → "dead-branch" wording;
never-built (`False`) → "you have not replaced the heavy op at all" wording. Both
land the kernel at `valid=False`.

## Files modified

| File | Change |
|------|--------|
| `src/heavyop_checker.py` (new module) | Home of the `TorchDispatchMode` subclass `_HeavyOpRecorder` + the module-level `_HEAVY_ATEN` allowlist (`aten::convolution`, `aten::_convolution`, `aten::conv_transpose3d`, `aten::addmm`, `aten::mm`, `aten::bmm`, `aten::linear`, …). `_HeavyOpRecorder.__torch_dispatch__` records the dispatched op name (normalized, e.g. `aten.convolution.default` → `aten::convolution`) then re-dispatches; `heavy_ops()` returns the recorded set intersected with `_HEAVY_ATEN`. Usable as a context manager. |
| `src/eval.py` | `from src.heavyop_checker import _HeavyOpRecorder`. In `run_and_check_correctness` (L801) wrap the **already-executed trial-0 forwards** — `model(*inputs)` (L853) and `model_new(*inputs)` (L858) — each in `with _HeavyOpRecorder() as rec:` (no extra passes). Set `metadata["heavy_op_executed_in_pytorch"] = bool(ref_heavy) and ref_heavy.issubset(new_heavy)` and `metadata["executed_heavy_ops"]`. Fully try/except-guarded → defaults `False` on any error. |
| `agentprompt/Utils/detect.py` | `replaced_heavy_op_built(kernel_src, arch_src)` — branch-agnostic: does `ModelNew` define/launch a `@triton.jit`/`tl.*` custom impl of the reference heavy op anywhere (not just the original `nn.*`/`F.*` call or an `F.conv_*` alias)? **No longer part of the validity gate** — used only to pick the corrective-feedback wording (built-but-dead vs. never-built) and, with `heavy_op_executed_in_pytorch`, the Mode-A escape hatch's `genuine_replacement` signal. Reuses `_find_model_class`/`_build_init_map`/`_HEAVY*`; fail-safe to `False`. Known blind spot (does not affect the gate): a hand-rolled `nn.Parameter` conv driven by a `@triton.jit` kernel (e.g. `2_15`'s method-style `_conv_transpose3d`) can read `False` here; because validity now keys on the runtime signal, such a kernel is still correctly **valid** when its Triton conv actually ran. |
| `agent/actions.py` | In `run_evaluator`, after the JSON parse: `heavy_op_unreplaced = bool(metrics) and bool(metrics.metadata.get("heavy_op_executed_in_pytorch"))`; if `heavy_op_unreplaced: valid = False` — **the gate, on the runtime signal alone**. Then `built = replaced_heavy_op_built(kernel, ref_arch_src)` selects the feedback flavor passed to `generate_evaluator_prompt`: `dead_branch_rewrite = heavy_op_unreplaced and built` (built-but-dead) vs. `heavy_op_unreplaced and not built` (never-built). 5-tuple return + 6 call sites unchanged. |
| `agentprompt/evaluator_prompt.py` | Keep `DEAD_BRANCH_ALERT`; add a sibling `HEAVY_OP_NOT_REPLACED_ALERT` for the never-built case ("runtime verification found the reference heavy op still ran in PyTorch and `ModelNew` defines no Triton replacement for it — you must add a custom Triton kernel that executes unconditionally on the forward path"). `generate_evaluator_prompt` takes the existing `dead_branch_rewrite: bool=False` plus a new `heavy_op_not_replaced: bool=False`; inject the matching alert **regardless of mode** (these land in Mode A, where `STRUCTURAL_ALERT` is skipped). Also widen the Mode-A slow-gate guard so an invalidated kernel does not also receive redesign/tuning guidance framed as if it were correct. |
| `agentprompt/skills/_base.md` | New rule appended **under item 2 ("Custom kernels are authorized.")**, reaching proposer/tuner/evaluator via `generate_skill_prompt` (`_base` is always emitted): the harness runs models at `training=True`; the heavy op **must** be replaced by a custom Triton kernel that executes **unconditionally** — leaving the original `nn.*`/`F.*` heavy op on the live path (in any branch, or because no Triton replacement was written) counts as *not replacing the op* and scores the kernel invalid. No separate edits to `proposer_prompt.py` / `tuner_prompt.py` (the shared `_base` path covers all three). |

Gate mechanics: `calculate_score(metric, False) → (1,0,0)` (`agent/utils.py:124`), so
any kernel whose heavy op ran in PyTorch sorts like an incorrect kernel and cannot
win as best. Remote eval needs **no code change** — metadata rides back through
`KernelExecResult(**result)` (`src/eval.py:1044`); the remote server must run this
updated code.

## Mode-A reconciliation

The Mode-A escape hatch (above) reuses this section's runtime signal:
`genuine_replacement = replaced_heavy_op_built AND not heavy_op_executed_in_pytorch`.
Under the broadened gate this is now consistent by construction — any kernel with
`heavy_op_executed_in_pytorch == True` is **invalid** *and* has
`genuine_replacement == False`, so an invalidated kernel can never be mistaken for
a genuine redesign. The escape hatch and the gate read the same
`heavy_op_executed_in_pytorch` signal; the static `replaced_heavy_op_built` term
now only distinguishes feedback wording (gate) and gates the escape hatch's
positive case (a built-and-executed replacement).

## Verification

Read-only / local (no GPU):

1. `python -c "import agent.actions, agentprompt.evaluator_prompt, agentprompt.Utils.detect, src.eval"`;
   `grep -rn "run_evaluator(" agent/` → 6 sites, all unpack 5 values.
2. `2_15` artifact probe (the current run in `outputs/KB-l2_AdaExplore_50/2_15`):
   read `heavy_op_executed_in_pytorch` from each `step_N_metrics.json`. Exactly one
   correct kernel (the honest Triton ConvTranspose3d) is `False` → stays **valid**;
   every other correct kernel is `True` → forced **invalid**. `replaced_heavy_op_built`
   only splits the invalid set into built-but-dead vs. never-built for messaging.
3. Evaluator-prompt probe: `heavy_op_not_replaced` path selects exactly one alert —
   `dead_branch_rewrite=True` → `DEAD_BRANCH_ALERT`; built==False →
   `HEAVY_OP_NOT_REPLACED_ALERT`; both present **even when `mode=="slow"`**.
4. `calculate_score(correct, False) → (1,0,0)`; `(correct, None) → (1,1,fast_p)`.

GPU / end-to-end:

5. Replay `2_15`: re-eval every correct kernel → all with
   `heavy_op_executed_in_pytorch==True` get `valid=False`, score `(1,0,0)`, alert
   present (dead-branch *or* never-built wording per `replaced_heavy_op_built`). The
   single honest Triton ConvTranspose3d → flag `False`, score unchanged
   `(1,1,~ratio)`, no alert.
6. Live smoke: a genuinely-executed Triton replacement is **not** flagged; a
   `self.training`-guarded fallback **and** a plain kept-op BN-fusion kernel are
   **both** flagged invalid and excluded from the elite pool.

## Risks

- **Stricter gate eliminates honest kept-op kernels.** This is the intended
  behavior change: a BN-only fusion that leaves the conv in PyTorch is now invalid
  even though it is correct and may be faster than a hand-written Triton conv. The
  search can no longer bank a partial fusion as "best" — it must produce an
  executing Triton heavy op. Accept this only if replacing the heavy op is the
  actual objective; if partial fusions should still count, this gate is too strict.
- **Recorder error → `heavy_op_executed_in_pytorch` defaults `False` → kernel
  treated as valid.** Fail-open (no false invalidation), but a recorder failure can
  let an unreplaced op through. Reuses existing forwards (no extra pass), fully
  try/except-guarded.
- **Remote eval:** metadata survives the round-trip; remote server must run the
  updated code, or every kernel reads `heavy_op_executed_in_pytorch` absent →
  `False` → gate never fires.
- **Allowlist coverage:** the gate only sees heavy ops in `_HEAVY_ATEN`. A
  reference op outside the allowlist yields `ref_heavy` empty →
  `heavy_op_executed_in_pytorch == False` → never gated. Keep the allowlist aligned
  with the heavy ops the benchmark actually uses.
