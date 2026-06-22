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

This is strictly stronger than the originally-proposed
`replaced_heavy_op_in_executed_path` source walk: the runtime observes the path
that *actually ran*, so it resolves **every** guard form (`self.training`,
`is_cuda`, stored flags), not just `if self.training:`. It is also mutually
exclusive with `dead_branch` by construction — `heavy_op_executed_in_pytorch` is
True for the cheat and False for a genuine replacement — so it composes cleanly
with the `dead_branch → mode="slow"` force.

### Two constraints (both required to exit Mode A)

1. **Current kernel genuinely replaced the heavy op** — `genuine_replacement`
   (above) is True for the kernel under evaluation.
2. **A recent large_step also genuinely replaced it** — scanning the last 5
   entries of `self.all_nodes` (global chronological order), at least one *other*
   `large_step` node whose stored kernel + metrics also satisfy
   `genuine_replacement`.

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
| `agent/actions.py` | `run_evaluator` already computes `meta` and `replaced_heavy_op_built(kernel, ref_arch_src)` for the dead-branch gate; `genuine_replacement` for the current kernel reuses both (`replaced_heavy_op_built and not meta.get("heavy_op_executed_in_pytorch")`). `run_evaluator` gains optional `redesign_exhausted: bool=False` and forwards it. The `if mode == "slow": direction="large"` force is untouched (only fires when still in slow mode). 5-tuple return unchanged. |
| `agent/mcts.py` | New private helper `MCTS._redesign_exhausted(new_kernel, new_metrics) -> bool` that ANDs constraint #1 (`genuine_replacement` on the current kernel + metrics) with constraint #2 (a *prior* `large_step` in `self.all_nodes[-5:]` whose stored kernel + metrics also satisfy `genuine_replacement`), using `self.ref_arch_src`. Both `run_evaluator` call sites compute it and pass `redesign_exhausted=`. Relies on nodes carrying their metrics metadata (already persisted per step). |

**Unchanged:** the 4 loop call sites (`large_loop.py:153`, `small_loop.py:170/197/223`)
keep calling `run_evaluator` without the new arg → `redesign_exhausted=False` →
Mode A behavior there is exactly as today (these loops keep no per-step type, and
`large_loop` already ignores `direction`).

## Computing `redesign_exhausted`

1. **Constraint #1 (current kernel):** `genuine_replacement(kernel, metrics)` =
   `replaced_heavy_op_built(kernel, ref_arch_src)` AND
   `not metrics.metadata.get("heavy_op_executed_in_pytorch")`. Both inputs are
   already in hand in `run_evaluator` (the dead-branch gate computes them).
2. **Constraint #2 (history):** walk `self.all_nodes[-5:]`; for each *prior*
   `large_step` node recompute `genuine_replacement` from its stored kernel +
   metrics. True if any qualifies.
3. **Fail safe:** missing metadata / parse error → treat as **not** genuine →
   `redesign_exhausted=False` (stays in Mode A — the conservative direction).

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
4. **mcts gate probe:** stub `all_nodes` with a genuine-replacement `large_step`
   node + a current genuine-replacement kernel → `_redesign_exhausted` True; flip
   either → False (both-required semantics).
5. **`2_15` regression (documented):** `_redesign_exhausted == False` across all
   `2_15` steps (step 1 meets #1 but not #2) ⇒ `2_15` stays in Mode A — expected
   and accepted, not a failure.
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

# Runtime verification — penalize heavy-op rewrites that never actually execute (dead-branch cheat)

## Context

The eval harness `src/eval.py` instantiates both `Model` and `ModelNew` and
**never calls `.eval()`** (verified: no `.eval()`/`.training` anywhere in
`eval.py`), so both run at PyTorch's default `training=True`.

In `outputs/KB-l2_AdaExplore_50/2_15` the agent exploited this. Steps 2/4/5/9 put
the original PyTorch heavy op in the **`if self.training:` branch** (which runs)
and parked the `@triton.jit` conv in the **`else:` branch** (which never runs);
step 6 even aliases its "Triton" conv to `F.conv_transpose3d`. So correctness
passes on the PyTorch path, timing measures PyTorch (~0.32×), and the kernel
*looks* like a heavy-op rewrite while being dead code. Steps 1/3/7 are
**legitimate** — they keep `self.conv_transpose(x)` unconditionally and only fuse
BN+mean-sub; these must **not** be penalized. All these cases are `correct=True`,
`fast_p≈0.27–0.32`, so they sit in **Mode A ("slow")**.

Nothing today observes which branch actually executed; the system credits "rewrite
attempted" from the mere presence of a Triton kernel in source. This change adds
**runtime verification** of the executed path, **gates** the cheat out of
selection, **feeds back** corrective guidance, and **hardens** the proposer/tuner
prompts. This runtime signal is also what the Mode-A escape hatch reuses for its
`genuine_replacement` check, in place of a source-level executed-path detector.

Locked decisions: detection = **runtime verification** (primary, ground truth);
enforcement = **gate + corrective feedback**; **harden** proposer/tuner prompts.

## Discriminator

Neither aten-presence alone nor source alone separates cheat from legit — both
keep the aten conv on the live path. The verdict **ANDs** two signals:

- **Runtime (ground truth):** during `ModelNew`'s measured forward
  (`training=True`), the reference heavy op still dispatched through aten (it ran
  in PyTorch/cuDNN, not Triton). Triton kernel launches do **not** go through
  `__torch_dispatch__`, so recording dispatched aten op names against a heavy-op
  allowlist reveals whether the heavy op truly ran.
- **Static intent (lightweight, in service of the gate):** `ModelNew` *defines* a
  Triton heavy-op replacement that is **not** on the executed path. This separates
  the cheat (built a Triton conv but it didn't run) from legit kept-op kernels
  (steps 1/3/7 — only a BN/sub kernel, never a Triton conv).

`heavy_op_dead_branch_rewrite = ref_heavy_nonempty AND ref_heavy ⊆ executed_aten_ops AND replaced_heavy_op_built(src)`.

## Files modified

| File | Change |
|------|--------|
| `src/heavyop_checker.py` (new module) | Home of the `TorchDispatchMode` subclass `_HeavyOpRecorder` + the module-level `_HEAVY_ATEN` allowlist (`aten::convolution`, `aten::_convolution`, `aten::conv_transpose3d`, `aten::addmm`, `aten::mm`, `aten::bmm`, `aten::linear`, …). `_HeavyOpRecorder.__torch_dispatch__` records the dispatched op name (normalized, e.g. `aten.convolution.default` → `aten::convolution`) then re-dispatches; `heavy_ops()` returns the recorded set intersected with `_HEAVY_ATEN`. Usable as a context manager. |
| `src/eval.py` | `from src.heavyop_checker import _HeavyOpRecorder`. In `run_and_check_correctness` (L801) wrap the **already-executed trial-0 forwards** — `model(*inputs)` (L853) and `model_new(*inputs)` (L858) — each in `with _HeavyOpRecorder() as rec:` (no extra passes). Set `metadata["heavy_op_executed_in_pytorch"] = bool(ref_heavy) and ref_heavy.issubset(new_heavy)` and `metadata["executed_heavy_ops"]`. Fully try/except-guarded → defaults `False` on any error. |
| `agentprompt/Utils/detect.py` | `replaced_heavy_op_built(kernel_src, arch_src)` — branch-agnostic: does `ModelNew` define/launch a `@triton.jit`/`tl.*` custom impl of the reference heavy op anywhere (not just the original `nn.*`/`F.*` call or an `F.conv_*` alias)? Used by both the dead-branch gate and (with `heavy_op_executed_in_pytorch`) the Mode-A escape hatch's `genuine_replacement` signal. Reuses `_find_model_class`/`_build_init_map`/`_HEAVY*`; fail-safe to `False`. Note the runtime signal subsumes the originally-proposed `replaced_heavy_op_in_executed_path` source walk — it observes the path that actually ran, so no live-branch AST detector is needed. Invariant: a kernel can't be both a genuine replacement (`heavy_op_executed_in_pytorch == False`) and a dead-branch rewrite (`== True`). |
| `agent/actions.py` | In `run_evaluator` (L129) after the JSON parse (L153): `dead_branch = bool(metrics) and metrics.metadata.get("heavy_op_executed_in_pytorch") and replaced_heavy_op_built(kernel, ref_arch_src)`; if `dead_branch: valid = False`. Pass `dead_branch` into `generate_evaluator_prompt`. 5-tuple return + 6 call sites unchanged. |
| `agentprompt/evaluator_prompt.py` | Add `DEAD_BRANCH_ALERT` (sibling of `STRUCTURAL_ALERT`) + a `dead_branch_rewrite: bool=False` param to `generate_evaluator_prompt`; inject the alert **regardless of mode** (the cheat lands in Mode A, where `STRUCTURAL_ALERT` is skipped at L249). Inject near L249–255. |
| `agentprompt/skills/_base.md` | New rule appended **under item 2 ("Custom kernels are authorized.")**, reaching proposer/tuner/evaluator via `generate_skill_prompt` (`_base` is always emitted): the harness runs models at `training=True`; the heavy-op replacement must execute unconditionally; do **not** guard it behind `if self.training` / `if x.is_cuda` / `else:` fallbacks — such branches are dead code and count as *not* replacing the op. No separate edits to `proposer_prompt.py` / `tuner_prompt.py` (the shared `_base` path covers all three). |

Gate mechanics: `calculate_score(metric, False) → (1,0,0)` (`agent/utils.py:124`), so
the cheat sorts like an incorrect kernel and cannot win as best. Remote eval needs
**no code change** — metadata rides back through `KernelExecResult(**result)`
(`src/eval.py:1044`); the remote server must run this updated code.

## Mode-A reconciliation

The Mode-A escape hatch (above) reuses this section's runtime signal: a
dead-branch "rewrite" has `heavy_op_executed_in_pytorch == True`, so its
`genuine_replacement` is False and it is **not** counted as a genuine redesign —
the problem is not prematurely released from redesign guidance. No separate
executed-path detector is needed; the gate and the escape hatch read the same
`heavy_op_executed_in_pytorch` + `replaced_heavy_op_built` pair.

## Verification

Read-only / local (no GPU):

1. `python -c "import agent.actions, agentprompt.evaluator_prompt, agentprompt.Utils.detect, src.eval"`;
   `grep -rn "run_evaluator(" agent/` → 6 sites, all unpack 5 values.
2. `detect.py` probes on real `2_15` artifacts: `replaced_heavy_op_built` →
   **True** for steps 2/4/5/6/9, **False** for 1/3/7. Combined with metrics,
   `genuine_replacement` (`built and not heavy_op_executed_in_pytorch`) → **True**
   only for step 1, **False** for the cheats (2/4/5/9, executed in PyTorch) and the
   kept-op kernels (3/7, nothing built). Hand-written kernel calling a `@triton.jit`
   conv unconditionally → built **True**, and (if it runs) `genuine_replacement` **True**.
3. Evaluator-prompt probe: `dead_branch_rewrite=True` → `DEAD_BRANCH_ALERT` present
   even when `mode=="slow"`.
4. `calculate_score(correct, False) → (1,0,0)`; `(correct, None) → (1,1,fast_p)`.

GPU / end-to-end:

5. Replay `2_15`: re-eval steps **2/4/5/9** → `heavy_op_executed_in_pytorch==True`,
   gate forces `valid=False`, score `(1,0,0)`, alert present. Steps **1/3/7** →
   flag `False`, score unchanged `(1,1,~0.32)`, no alert.
6. Live smoke: a genuinely-executed Triton replacement is **not** flagged, while a
   `self.training`-guarded fallback is flagged and excluded from the elite pool.

## Risks

- **False positive on legit kept-op kernels** (2_15 steps 1/3/7): prevented by the
  `replaced_heavy_op_built` AND — they never build a Triton heavy-op kernel.
- **Recorder overhead / errors:** reuses existing forwards (no extra pass), fully
  try/except-guarded → defaults `False` (no penalty).
- **Remote eval:** metadata survives the round-trip; remote server must run the
  updated code.
- **Non-`self.training` guards** (`is_cuda`, stored flags): on a CUDA-only harness
  the `is_cuda` branch is the one that runs, so the kernel isn't actually dead —
  runtime verification correctly does not flag it.
