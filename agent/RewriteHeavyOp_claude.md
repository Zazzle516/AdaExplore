# Heavy-op replacement — runtime-verified validity gate + Mode A escape hatch

This document covers two coupled changes that share **one runtime signal**
(`metadata["heavy_op_executed_in_pytorch"]`, produced by `src/eval.py` via
`src/heavyop_checker.py`):

1. **Runtime verification gate** — a kernel whose reference heavy op still
   dispatched through PyTorch/aten during the measured forward did **not** replace
   it, and is forced **invalid** (covers the dead-branch cheat, the kept-op kernel,
   and the never-built case with one test).
2. **Mode A escape hatch** — stop forcing redesign once the agent has *genuinely*
   replaced the heavy op on the live path and is still slow, so tuning (small
   steps) becomes reachable again.

Both read the **runtime signal alone** as ground truth. There is a **single
corrective wording** for every invalid case (dead-branch, kept-op, never-built) —
the kernel did not replace the heavy op, full stop. The static
`replaced_heavy_op_built` check is therefore **unused** and can be dropped: it
gates neither validity nor the escape hatch, and the unified wording no longer
needs it to pick a message.

---

# Part 1 — Runtime verification gate

## Context

The eval harness `src/eval.py` instantiates both `Model` and `ModelNew` and
**never calls `.eval()`** (verified: no `.eval()`/`.training` anywhere in
`eval.py`), so both run at PyTorch's default `training=True`.

**Policy.** The heavy op (Conv*, ConvTranspose*, Linear, matmul) **must** be
replaced by a custom Triton implementation that actually executes. If the
reference heavy op still dispatched through PyTorch/aten during the measured
forward, the kernel did **not** replace it — and the result is forced to
**invalid**, regardless of intent. This is not a cheat detector; it is a hard
requirement. Three previously-distinct cases collapse to one verdict:

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

## Discriminator

The validity gate reads **one** signal:

- **Runtime (ground truth):** during `ModelNew`'s measured forward
  (`training=True`), did the reference heavy op still dispatch through aten? Triton
  kernel launches do **not** go through `__torch_dispatch__`, so recording
  dispatched aten op names against a heavy-op allowlist reveals whether the heavy
  op truly ran in PyTorch. `heavy_op_executed_in_pytorch = ref_heavy_nonempty AND
  ref_heavy ⊆ executed_aten_ops` (computed in `src/eval.py`).

`heavy_op_not_replaced = heavy_op_executed_in_pytorch` → **invalid**.

All three invalid cases (dead-branch, kept-op, never-built) get the **same**
corrective wording: runtime verification found the reference heavy op still ran in
PyTorch, so `ModelNew` did not replace it — you must add a custom Triton kernel
that executes unconditionally on the forward path. There is no per-case message
and no secondary static check to choose between wordings; the static
`replaced_heavy_op_built` is not consulted.

Gate mechanics: `calculate_score(metric, False) → (1,0,0)` (`agent/utils.py:124`),
so any kernel whose heavy op ran in PyTorch sorts like an incorrect kernel and
cannot win as best. Remote eval needs **no code change** — metadata rides back
through `KernelExecResult(**result)` (`src/eval.py:1044`); the remote server must
run this updated code.

---

# Part 2 — Mode A escape hatch

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

## The genuine-replacement signal — runtime execution alone

The escape hatch's genuineness test reads the **runtime signal alone**. It does
**not** AND in the static `replaced_heavy_op_built`:

    genuine_replacement(kernel, metrics, ref_arch_src) =
        bool(metrics) and metrics.correctness
        and not metrics.metadata.get("heavy_op_executed_in_pytorch")

(`kernel`/`ref_arch_src` are kept in the signature for call-site symmetry but are
no longer read.) Rationale: `replaced_heavy_op_built` was calibrated on the
dead-branch cheat — it keys on `_reads_heavy_module_weight` (reading
`self.<conv>.weight`) and therefore recognizes only **one shape** of replacement.
An agent can mask a non-replacement with an if/else (or any branch), and the
static check "only looks for that specific case." It also has a documented blind
spot (Part 1's `detect.py` row): `2_15`'s method-style `_conv_transpose3d` driven
by a hand-rolled `nn.Parameter` reads `replaced_heavy_op_built == False` even
though its Triton conv genuinely executes — which would wrongly deny an honest
rewrite. The runtime signal is branch-agnostic ground truth ("did the reference
heavy op still dispatch through aten?"), so it already covers the dead-branch,
kept-op, and never-built forms with one test. We add the `correctness` term so a
wrong kernel whose op happens not to dispatch is never counted as a redesign
success.

## Exit condition (single, linear)

Mode A releases to `GOAL` on **one** condition:

- **Current kernel genuinely replaced the heavy op** — `genuine_replacement`
  (above) is True for the kernel under evaluation: it is correct and the reference
  heavy op did **not** dispatch through aten during the measured forward.

That is the whole gate. There is no second, history-scanning constraint and no
per-node bookkeeping.

### Why one condition is enough — the gate provides forward persistence

An earlier draft added a second constraint ("a *recent* large_step also genuinely
replaced it", via a `replace_large_success` flag scanned over the last few
`all_nodes`). Its purpose was to keep the search from releasing to tuning and then
drifting onto a kernel that quietly abandons the rewrite. **Part 1's validity gate
already makes that drift impossible:** any kernel that keeps the heavy op in
PyTorch is now *invalid* and scores `(1,0,0)`, so it can never beat a genuine
rewrite or win as best. The persistence the second constraint tried to prove by
looking *backward* over history is now enforced *forward* and *locally* by the gate
on every future evaluation — a backward window is redundant.

Dropping it also fixes a contradiction the two-constraint version carried: the
motivating example (`2_15` step 1 — an honest Triton ConvTranspose3d that is
genuinely executed yet slow at `fast_p≈0.11`) is the *only* genuine rewrite in its
run, so the second constraint (a *second* genuine large_step) was never met and the
escape hatch stayed shut — failing the exact case it exists to rescue. With the
single condition, step 1 satisfies the gate and correctly escapes to tuning.

## Computing `redesign_exhausted`

1. **The condition:** `redesign_exhausted = genuine_replacement(kernel, metrics, ref_arch_src)`
   = `metrics.correctness AND not metrics.metadata.get("heavy_op_executed_in_pytorch")`.
   Computed at the step site from the current kernel's metrics, before
   `run_evaluator`. No history scan, no node fields.
2. **Fail safe:** missing metadata / parse error → `genuine_replacement` returns
   **False** → `redesign_exhausted=False` (stays in Mode A — conservative).

## Locked design decisions

- **Genuineness test = runtime execution signal alone (ground truth).** Reads
  `heavy_op_executed_in_pytorch` (plus `correctness`) only. The static
  `replaced_heavy_op_built` is **not** part of the test — no AST-shape or
  `self.training` heuristic, because the recorder sees what actually dispatched
  regardless of branch form.
- **Single, linear exit condition.** Mode A releases as soon as the current kernel
  genuinely replaced the heavy op; forward persistence is guaranteed by the Part 1
  gate, not by a backward history scan. No `replace_large_success` field, no
  `all_nodes` window.
- **`2_15` step 1 escapes to tuning — correctly.** Its cheat kernels (2/4/5/9) and
  its kept-op kernels (3/7) all keep the conv on the live path
  (`heavy_op_executed_in_pytorch == True`), so `genuine_replacement` is False and
  they are *invalid* anyway. Step 1 is the honest, genuinely-executed Triton
  ConvTranspose3d (correct, op not in PyTorch) → `genuine_replacement` True → Mode A
  releases, so tuning (small steps) becomes reachable instead of forcing yet another
  from-scratch redesign. This is a **general anti-stuck mechanism**, not a
  `2_15`-specific patch.

## Mode-A reconciliation with the gate

Both the validity gate (Part 1) and the escape hatch read the same
`heavy_op_executed_in_pytorch` signal, so they are consistent by construction: any
kernel with `heavy_op_executed_in_pytorch == True` is **invalid** *and* has
`genuine_replacement == False`, so an invalidated kernel can never be mistaken for
a genuine redesign. The static `replaced_heavy_op_built` term is used by neither —
validity, the escape hatch, and the (now single) corrective wording all key on the
runtime signal alone.

---

# Files modified (consolidated)

| File | Change |
|------|--------|
| `src/heavyop_checker.py` (new module) | Home of the `TorchDispatchMode` subclass `_HeavyOpRecorder` + the module-level `_HEAVY_ATEN` allowlist (`aten::convolution`, `aten::_convolution`, `aten::conv_transpose3d`, `aten::addmm`, `aten::mm`, `aten::bmm`, `aten::linear`, …). `_HeavyOpRecorder.__torch_dispatch__` records the dispatched op name (normalized, e.g. `aten.convolution.default` → `aten::convolution`) then re-dispatches; `heavy_ops()` returns the recorded set intersected with `_HEAVY_ATEN`. Usable as a context manager. |
| `src/eval.py` | `from src.heavyop_checker import _HeavyOpRecorder`. In `run_and_check_correctness` (L801) wrap the **already-executed trial-0 forwards** — `model(*inputs)` (L853) and `model_new(*inputs)` (L858) — each in `with _HeavyOpRecorder() as rec:` (no extra passes). Set `metadata["heavy_op_executed_in_pytorch"] = bool(ref_heavy) and ref_heavy.issubset(new_heavy)` and `metadata["executed_heavy_ops"]`. Fully try/except-guarded → defaults `False` on any error. |
| `agentprompt/Utils/detect.py` | `replaced_heavy_op_built(kernel_src, arch_src)` is **no longer called** by the gate, the escape hatch, or the feedback wording. It may be left in place as dead code or removed; nothing in this design depends on it. |
| `agent/actions.py` | **(gate)** In `run_evaluator`, after the JSON parse: `heavy_op_unreplaced = bool(metrics) and bool(metrics.metadata.get("heavy_op_executed_in_pytorch"))`; if `heavy_op_unreplaced: valid = False` — the gate, on the runtime signal alone. Pass `heavy_op_unreplaced` straight to `generate_evaluator_prompt` (single flag, single alert — no `replaced_heavy_op_built` call, no dead-branch/never-built split). **(escape hatch)** Module-level helper `genuine_replacement(kernel, metrics, ref_arch_src) -> bool = bool(metrics) and bool(getattr(metrics,"correctness",False)) and not (getattr(metrics,"metadata",{}) or {}).get("heavy_op_executed_in_pytorch")` (no `replaced_heavy_op_built` term; `kernel`/`ref_arch_src` retained for call-site symmetry, unused). `run_evaluator` gains optional `redesign_exhausted: bool=False` and forwards it to `generate_evaluator_prompt`. The `if mode == "slow": direction="large"` force is untouched. 5-tuple return + 6 call sites unchanged. |
| `agentprompt/evaluator_prompt.py` | **(gate)** A **single** `HEAVY_OP_NOT_REPLACED_ALERT` covering all invalid cases ("runtime verification found the reference heavy op still ran in PyTorch and `ModelNew` did not replace it — you must add a custom Triton kernel that executes unconditionally on the forward path"). `generate_evaluator_prompt` takes one `heavy_op_not_replaced: bool=False`; inject the alert **regardless of mode** (it lands in Mode A, where `STRUCTURAL_ALERT` is skipped). Widen the Mode-A slow-gate guard so an invalidated kernel does not also receive redesign/tuning guidance framed as if it were correct. **(escape hatch)** Also gains optional `redesign_exhausted: bool=False`: in the mode-selection block, when the slow gate matches **and** `redesign_exhausted` is True, choose `"default"` instead of `"slow"`. `heavy_op_not_replaced` still forces `"slow"` (mutually exclusive with `redesign_exhausted`). Return tuple unchanged `(prompt, mode)`. |
| `agentprompt/skills/_base.md` | New rule appended **under item 2 ("Custom kernels are authorized.")**, reaching proposer/tuner/evaluator via `generate_skill_prompt` (`_base` is always emitted): the harness runs models at `training=True`; the heavy op **must** be replaced by a custom Triton kernel that executes **unconditionally** — leaving the original `nn.*`/`F.*` heavy op on the live path (in any branch, or because no Triton replacement was written) counts as *not replacing the op* and scores the kernel invalid. No separate edits to `proposer_prompt.py` / `tuner_prompt.py`. |
| `agent/mcts.py` (steps) | At the **large_step** site (`:526`) and the **small_step** site (`:587`): compute `redesign_exhausted = genuine_replacement(proposal_kernel, proposal_metrics, self.ref_arch_src)` **before** calling `run_evaluator`, and forward it. No node fields, no `all_nodes` scan — the value depends only on the current kernel's metrics. (No `MCTSNode`/`_create_node` change: the dropped second constraint was the only thing that needed `replace_large_success`.) |

**Unchanged:** the 4 loop call sites (`large_loop.py:153`, `small_loop.py:170/197/223`)
keep calling `run_evaluator` without the new arg → `redesign_exhausted=False` →
Mode A behavior there is exactly as today (these loops keep no per-step type, and
`large_loop` already ignores `direction`).

---

# Verification

## Read-only / local (no GPU)

1. **Import + arity:** `python -c "import agent.actions, agent.mcts, agentprompt.evaluator_prompt, agentprompt.Utils.detect, src.eval"`;
   `grep -rn "run_evaluator(" agent/` → 6 sites, all still unpack 5 values.
2. **`genuine_replacement` probes** (real `2_15` kernels + their metrics):
   - step 1 (correct, `heavy_op_executed_in_pytorch: false`) → **True**.
   - steps 2/4/5/9 (`heavy_op_executed_in_pytorch: true`) → **False**.
   - steps 3/7 (kept-op, conv ran in PyTorch → `heavy_op_executed_in_pytorch: true`) → **False**.
   (No dependence on `replaced_heavy_op_built` — a kernel is genuine iff its heavy
   op did not dispatch in PyTorch, whatever branch shape it used.)
3. **Gate / score:** `calculate_score(correct, False) → (1,0,0)`; `(correct, None) → (1,1,fast_p)`.
4. **`2_15` artifact probe** (`outputs/KB-l2_AdaExplore_50/2_15`): read
   `heavy_op_executed_in_pytorch` from each `step_N_metrics.json`. Exactly one
   correct kernel (the honest Triton ConvTranspose3d) is `False` → stays **valid**;
   every other correct kernel is `True` → forced **invalid** (all with the same
   alert).
5. **Evaluator-prompt probe (alert):** `heavy_op_not_replaced=True` injects the
   single `HEAVY_OP_NOT_REPLACED_ALERT` for every invalid case, **even when
   `mode=="slow"`**; `False` injects no alert.
6. **Evaluator-prompt probe (escape hatch):** `generate_evaluator_prompt` with
   synthetic correct + `fast_p=0.4`: `redesign_exhausted=False` → `mode=="slow"`
   (SLOW_GOAL present); `redesign_exhausted=True` → `mode=="default"` (GOAL present,
   SLOW_GOAL absent).
7. **escape-hatch condition probe:** `redesign_exhausted = genuine_replacement(current kernel)`
   directly — a correct, not-executed-in-PyTorch kernel → True; an incorrect kernel
   or one with `heavy_op_executed_in_pytorch=True` → False. No `all_nodes` stub and
   no node fields are involved (the second constraint was dropped).
8. **`2_15` (documented):** step 1 (honest, genuinely-executed Triton
   ConvTranspose3d) → `redesign_exhausted == True` ⇒ Mode A **releases** to tuning;
   every other correct kernel keeps the conv in PyTorch ⇒ invalid anyway. This is the
   intended fix of the motivating case, not a regression.

## GPU / end-to-end

9. **Replay `2_15`:** re-eval every correct kernel → all with
   `heavy_op_executed_in_pytorch==True` get `valid=False`, score `(1,0,0)`, the
   single `HEAVY_OP_NOT_REPLACED_ALERT` present. The single honest Triton
   ConvTranspose3d → flag `False`, score unchanged `(1,1,~ratio)`, no alert.
10. **Live smoke (gate):** a genuinely-executed Triton replacement is **not** flagged;
    a `self.training`-guarded fallback **and** a plain kept-op BN-fusion kernel are
    **both** flagged invalid and excluded from the elite pool.
11. **Live smoke (escape hatch):** a problem where the agent genuinely replaces a
    Conv/Linear on the live path and stays <0.8× → confirm the evaluator switches
    from SLOW_GOAL to GOAL on that step and a small step becomes reachable.

---

# Risks

- **Stricter gate eliminates honest kept-op kernels.** Intended behavior change: a
  BN-only fusion that leaves the conv in PyTorch is now invalid even though it is
  correct and may be faster than a hand-written Triton conv. The search can no
  longer bank a partial fusion as "best" — it must produce an executing Triton heavy
  op. Accept this only if replacing the heavy op is the actual objective.
- **Recorder error → `heavy_op_executed_in_pytorch` defaults `False`.** For the
  **gate** this is fail-open (a kernel is treated as valid), so a recorder failure
  can let an unreplaced op through. For the **escape hatch** the same default makes
  `genuine_replacement` True for a correct kernel, so a single fluke can release Mode
  A early. This is *not* a new failure mode: it is the same fail-open behavior the
  gate already accepts (a recorder error that hides an unreplaced op makes that
  kernel pass the gate too), and the worst case is an early release to tuning — not a
  correctness or scoring error, since a non-rewriting kernel reached during tuning is
  still invalidated by the gate. Reuses existing forwards (no extra pass), fully
  try/except-guarded.
- **Detector false-negative → stays in Mode A:** identical to today's behavior, no
  regression.
- **Remote eval:** metadata survives the round-trip; remote server must run the
  updated code, or every kernel reads `heavy_op_executed_in_pytorch` absent →
  `False` → gate never fires and the escape hatch never opens.
- **Allowlist coverage:** the gate only sees heavy ops in `_HEAVY_ATEN`. A reference
  op outside the allowlist yields `ref_heavy` empty → `heavy_op_executed_in_pytorch
  == False` → never gated. Keep the allowlist aligned with the heavy ops the
  benchmark actually uses.
