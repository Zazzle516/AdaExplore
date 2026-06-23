# Heavy-op replacement — implementation guide

Two coupled changes sharing one runtime signal,
`metadata["heavy_op_executed_in_pytorch"]` (produced by `src/eval.py` via
`src/heavyop_checker.py`):

1. **Runtime verification gate** — if the reference heavy op still dispatched
   through PyTorch/aten during the measured forward, the kernel did not replace it
   → force **invalid**.
2. **Mode A escape hatch** — when the current kernel genuinely replaced the heavy
   op on the live path and is still slow, release Mode A to `GOAL` so tuning (small
   steps) is reachable.

Both key on the runtime signal alone. The static `replaced_heavy_op_built` is not
called by either and is removed.

## Implementation status

The gate and its runtime signal are **already implemented**; the dead-branch
split is **still in the code and must be torn down**; the escape hatch is **not
yet built**.

| Area | State | Action |
|------|-------|--------|
| `src/heavyop_checker.py` (`_HeavyOpRecorder`, `heavy_ops()`) | **Done** | none |
| `src/eval.py` — recorder wraps trial-0 forwards, sets `heavy_op_executed_in_pytorch` + `executed_heavy_ops` (`L859–895`) | **Done** | none |
| `agent/actions.py` — `heavy_op_unreplaced` computed, `valid=False` gate | **Done** | none |
| `agent/actions.py` — `replaced_heavy_op_built` import + `built`/`dead_branch`/`heavy_op_not_replaced` split | **Present (wrong)** | **Tear down** → collapse to a single `heavy_op_unreplaced` flag |
| `agentprompt/evaluator_prompt.py` — `DEAD_BRANCH_ALERT` + `dead_branch_rewrite` param | **Present (wrong)** | **Tear down** → one alert |
| `agent/actions.py` / `evaluator_prompt.py` — `redesign_exhausted`, `genuine_replacement` | **Missing** | **Build** |
| `agent/mcts.py` — compute/forward `redesign_exhausted` at step sites | **Missing** | **Build** |

### Teardown: collapse the dead-branch split to a single alert

The current code routes invalid kernels through two messages via the static
`replaced_heavy_op_built`; this design uses **one** wording for every invalid case.
Remove, in order:

1. `agent/actions.py:14` — delete `from agentprompt.Utils.detect import replaced_heavy_op_built`.
2. `agent/actions.py` (`run_evaluator`, ~`:146–150`) — delete the `built`,
   `dead_branch`, and `heavy_op_not_replaced` locals; keep only
   `heavy_op_unreplaced`. Pass `heavy_op_not_replaced=heavy_op_unreplaced` (single
   flag) to `generate_evaluator_prompt`; drop the `dead_branch_rewrite=` argument.
3. `agentprompt/evaluator_prompt.py` — delete the `DEAD_BRANCH_ALERT` constant
   (`:191`) and the `dead_branch_rewrite` parameter (`:227`); in the mode-selection
   block (`:271`) set `heavy_op_unreplaced = heavy_op_not_replaced` and replace the
   `if dead_branch_rewrite / elif heavy_op_not_replaced` pair (`:308–309`) with a
   single `if heavy_op_not_replaced: prompt += HEAVY_OP_NOT_REPLACED_ALERT`.
4. `agentprompt/Utils/detect.py` — `replaced_heavy_op_built` (`:383`) is now
   uncalled (the only references were `actions.py:14`/`:148`). **Delete it.** Its
   exclusive private helpers — `_reads_heavy_module_weight` (`:364`),
   `_defines_triton_kernel` (`:342`), `_ref_heavy_op_names` (`:314`),
   `_find_class_named` (`:303`) — are then dead too; delete them as well. Keep
   `_build_init_map` (still used by other detectors at `:252`/`:325`).

## Signal definitions

The eval harness never calls `.eval()`, so both `Model` and `ModelNew` run at
`training=True`. Triton kernel launches do not pass through `__torch_dispatch__`,
so recording dispatched aten op names reveals whether the heavy op ran in PyTorch.

    # in src/eval.py, from the recorded aten ops vs. the reference heavy ops
    heavy_op_executed_in_pytorch = bool(ref_heavy) and ref_heavy.issubset(new_heavy)

    # gate (agent/actions.py)
    heavy_op_unreplaced = bool(metrics) and bool(metrics.metadata.get("heavy_op_executed_in_pytorch"))
    # heavy_op_unreplaced → valid = False

    # escape hatch (agent/actions.py, module-level helper)
    genuine_replacement(kernel, metrics, ref_arch_src) = \
        bool(metrics) and bool(getattr(metrics, "correctness", False)) \
        and not (getattr(metrics, "metadata", {}) or {}).get("heavy_op_executed_in_pytorch")

    # at each step site (agent/mcts.py)
    redesign_exhausted = genuine_replacement(kernel, metrics, ref_arch_src)

`kernel`/`ref_arch_src` are retained in the helper signature for call-site symmetry
but are not read. Mode-A exit is this single condition — no history scan, no node
fields. Fail safe: missing metadata / parse error → `genuine_replacement` False →
`redesign_exhausted` False (stays in Mode A); recorder error →
`heavy_op_executed_in_pytorch` defaults False (gate fail-open).

`calculate_score(metric, False) → (1,0,0)` (`agent/utils.py:124`), so an
invalidated kernel sorts like an incorrect one and cannot win as best.

## Files to modify

| File | Change |
|------|--------|
| `src/heavyop_checker.py` (new module) | **Done.** `TorchDispatchMode` subclass `_HeavyOpRecorder` + module-level `_HEAVY_ATEN` allowlist (`aten::convolution`, `aten::_convolution`, `aten::conv_transpose3d`, `aten::addmm`, `aten::mm`, `aten::bmm`, `aten::linear`, …). `__torch_dispatch__` records the normalized dispatched op name (e.g. `aten.convolution.default` → `aten::convolution`) then re-dispatches; `heavy_ops()` returns the recorded set ∩ `_HEAVY_ATEN`. Usable as a context manager. |
| `src/eval.py` | **Done** (`L29`, `L859–895`). Imports `_HeavyOpRecorder`; in `run_and_check_correctness` wraps the already-executed trial-0 forwards — `model(*inputs)` and `model_new(*inputs)` — each in `with _HeavyOpRecorder() as rec:` (no extra passes). Sets `metadata["heavy_op_executed_in_pytorch"] = bool(ref_heavy) and ref_heavy.issubset(new_heavy)` and `metadata["executed_heavy_ops"]`. Fully try/except-guarded → defaults `False` on any error. |
| `agentprompt/Utils/detect.py` | **Teardown.** Once the actions.py import is removed, `replaced_heavy_op_built` (`:383`) is uncalled — **delete it** along with its now-orphaned exclusive helpers (`_reads_heavy_module_weight`, `_defines_triton_kernel`, `_ref_heavy_op_names`, `_find_class_named`). Keep `_build_init_map` (used elsewhere). |
| `agent/actions.py` | **(gate — partly done, partly teardown)** The `heavy_op_unreplaced` flag and `if heavy_op_unreplaced: valid = False` gate already exist. Remove the `replaced_heavy_op_built` import and the `built`/`dead_branch`/`heavy_op_not_replaced` split (see Teardown above); pass the single `heavy_op_unreplaced` flag to `generate_evaluator_prompt`. **(escape hatch — build)** Add module-level helper `genuine_replacement(kernel, metrics, ref_arch_src) -> bool` (formula above). `run_evaluator` gains optional `redesign_exhausted: bool=False` and forwards it to `generate_evaluator_prompt`. The `if mode == "slow": direction="large"` force is untouched. 5-tuple return + 6 call sites unchanged. |
| `agentprompt/evaluator_prompt.py` | **(gate — teardown)** Delete `DEAD_BRANCH_ALERT` and the `dead_branch_rewrite` param; keep the single `HEAVY_OP_NOT_REPLACED_ALERT` (already present, `:210`) ("runtime verification found the reference heavy op still ran in PyTorch and `ModelNew` did not replace it — you must add a custom Triton kernel that executes unconditionally on the forward path"), injected regardless of mode (it lands in Mode A, where `STRUCTURAL_ALERT` is skipped). The Mode-A slow-gate guard already forces `"slow"` for an invalidated kernel. **(escape hatch — build)** Add optional `redesign_exhausted: bool=False`: in the mode-selection block, when the slow gate matches **and** `redesign_exhausted` is True, choose `"default"` instead of `"slow"`. `heavy_op_not_replaced` still forces `"slow"` (mutually exclusive with `redesign_exhausted`). Return tuple unchanged `(prompt, mode)`. |
| `agentprompt/skills/_base.md` | **Build.** Append a rule under item 2 ("Custom kernels are authorized."): the harness runs models at `training=True`; the heavy op must be replaced by a custom Triton kernel that executes unconditionally — leaving the original `nn.*`/`F.*` heavy op on the live path (in any branch, or because no replacement was written) counts as not replacing the op and scores the kernel invalid. Reaches proposer/tuner/evaluator via `generate_skill_prompt` (`_base` always emitted); no edits to `proposer_prompt.py` / `tuner_prompt.py`. |
| `agent/mcts.py` (steps) | **Build.** At the large_step site (`:517`) and the small_step site (`:587`): compute `redesign_exhausted = genuine_replacement(proposal_kernel, proposal_metrics, self.ref_arch_src)` before calling `run_evaluator`, and forward it. No `MCTSNode`/`_create_node` change. |

**Unchanged:** the 4 loop call sites (`large_loop.py:153`, `small_loop.py:170/197/223`)
keep calling `run_evaluator` without the new arg → `redesign_exhausted=False`.

## Verification

### Read-only / local (no GPU)

1. **Import + arity:** `python -c "import agent.actions, agent.mcts, agentprompt.evaluator_prompt, agentprompt.Utils.detect, src.eval"`;
   `grep -rn "run_evaluator(" agent/` → 6 sites, all unpack 5 values.
2. **`genuine_replacement` probes** (real `2_15` kernels + metrics): step 1
   (correct, `heavy_op_executed_in_pytorch: false`) → **True**; steps 2/4/5/9 and
   3/7 (`heavy_op_executed_in_pytorch: true`) → **False**.
3. **Gate / score:** `calculate_score(correct, False) → (1,0,0)`; `(correct, None) → (1,1,fast_p)`.
4. **`2_15` artifact probe** (`outputs/KB-l2_AdaExplore_50/2_15`): read
   `heavy_op_executed_in_pytorch` from each `step_N_metrics.json`. Exactly one
   correct kernel (the honest Triton ConvTranspose3d) is `False` → **valid**; every
   other correct kernel is `True` → **invalid** (same alert).
5. **Alert probe:** `heavy_op_not_replaced=True` injects the single
   `HEAVY_OP_NOT_REPLACED_ALERT` even when `mode=="slow"`; `False` → no alert.
6. **Escape-hatch mode probe:** `generate_evaluator_prompt` with synthetic correct
   + `fast_p=0.4`: `redesign_exhausted=False` → `mode=="slow"` (SLOW_GOAL present);
   `redesign_exhausted=True` → `mode=="default"` (GOAL present, SLOW_GOAL absent).
7. **Escape-hatch condition probe:** `redesign_exhausted = genuine_replacement(current kernel)`
   — correct + not-executed-in-PyTorch → True; incorrect or
   `heavy_op_executed_in_pytorch=True` → False.

### GPU / end-to-end

8. **Replay `2_15`:** every correct kernel with `heavy_op_executed_in_pytorch==True`
   → `valid=False`, score `(1,0,0)`, alert present; the honest Triton
   ConvTranspose3d → flag `False`, score `(1,1,~ratio)`, no alert, and Mode A
   releases to `GOAL` on that step.
9. **Live smoke (gate):** a genuinely-executed Triton replacement is not flagged; a
   `self.training`-guarded fallback and a plain kept-op BN-fusion kernel are both
   flagged invalid and excluded from the elite pool.
10. **Live smoke (escape hatch):** a problem where the agent genuinely replaces a
    Conv/Linear on the live path and stays <0.8× → evaluator switches from SLOW_GOAL
    to GOAL on that step and a small step becomes reachable.

## Operational caveats

- **Remote eval:** metadata rides back through `KernelExecResult(**result)`
  (`src/eval.py:1044`) — no code change, but the remote server **must run the
  updated code**, or `heavy_op_executed_in_pytorch` reads absent → `False` → gate
  never fires and the escape hatch never opens.
- **Allowlist coverage:** the gate only sees heavy ops in `_HEAVY_ATEN`. A reference
  op outside it yields `ref_heavy` empty → `heavy_op_executed_in_pytorch == False`
  → never gated. Keep `_HEAVY_ATEN` aligned with the benchmark's heavy ops.
- **Recorder error → fail-open:** `heavy_op_executed_in_pytorch` defaults `False`,
  so a recorder failure can let an unreplaced op pass the gate and can release Mode A
  early. Worst case is early tuning, not a scoring error (a non-rewriting kernel
  reached during tuning is still invalidated by the gate). Reuses existing forwards,
  fully try/except-guarded.
- **Behavior change:** honest kept-op kernels (BN-only fusion that leaves the conv
  in PyTorch) are now invalid even if correct and faster. Intended only if replacing
  the heavy op is the objective.
