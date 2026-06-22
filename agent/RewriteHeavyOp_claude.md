# Three-mode evaluator prompt: per-mode specialization + JSON response

## Context

The evaluator agent grades the most recent Triton kernel and emits guidance for
the next MCTS step. Today its prompt has only **two** shapes, selected by
measured speedup `ratio = run_info.runtime_stats["fast_p"]`:

- `ratio >= 5.0` → `ADVERSARIAL_GOAL` (anti-shortcut review)
- everything else → `GOAL`

This misses the **slow-correct** case (`0 < ratio < 0.8`). Worked example:
`outputs/KB-l2_AdaExplore_50/2_15` — reference is ConvTranspose3d → BN3d →
subtract-spatial-mean. The kernel kept `nn.ConvTranspose3d` intact and only
fused BN+sub; final best speedup 0.32×. When the reference heavy op
(Conv / ConvTranspose / Linear) is retained, that PyTorch op *is* the
bottleneck and tuning around it cannot close the gap. In that case the
evaluator wastes effort producing `small_guidance`, a `direction` decision
(should always be "large"), and a `valid` check (a slow correct kernel cannot
be an algebraic shortcut — shortcuts *produce* speedups).

We add a dedicated third mode that asks **only** for redesign guidance, and we
move the evaluator's **response format** from four XML-style tags
(`<small_guidance>`, `<large_guidance>`, `<direction>`, `<valid>`; plus
`<reasoning>` in adversarial mode) to a single **JSON** object. Only the
evaluator agent's *reply* becomes JSON — the goal templates sent *to* the
evaluator stay ordinary prose that instruct the model to respond with one JSON
object. We do not require the evaluator prompt itself to be JSON.

Mode A and Mode B/C use **separate response schemas**. In Mode A the kernel is
slow-correct, so `direction` is always `"large"` by construction — the model is
**not** asked for a `direction` field. To preserve the MCTS large-bias
(`mcts.py:694`, keyed on `evaluator_direction == "large"`),
`generate_evaluator_prompt` returns the selected `mode` alongside the prompt,
and `run_evaluator` forces `direction="large"` whenever `mode == "slow"` — a
single source of truth for the Mode-A gate.

Outcome: three mutually-exclusive prose prompt modes (slow / adversarial /
default), each requesting a JSON response, with **zero change to the 6
`run_evaluator` call sites** because `run_evaluator` keeps its 4-tuple return
`(small_guidance, large_guidance, direction, valid)`.

Beyond the mode split, each mode's prompt is **specialized to its objective**
rather than sharing a generic preamble. The shared `PROBLEM_STATEMENT` is
trimmed to a neutral role line: its skill-framing / fusion / "never propose
algebraic shortcuts" content is already carried by the always-present `_base`
skill section (the full no-shortcut contract) and re-stated by the goal blocks,
so the preamble is redundant. The skill prompt's `step_type` also varies by
mode — Mode A is redesign-only, so it loads Design content only (`"large"`),
while Modes B/C keep both Design and Tuning (`"both"`). To support this, `mode`
is computed **once, early**, and reused for the skill `step_type`, the
STRUCTURAL_ALERT gate, and the goal-block append.

## Locked decisions

- **Mode A gate is pure ratio** — `correct and 0 < fast_p < 0.8`. No source
  scan. A slow rewrite that
  already replaced the heavy op but is still <0.8× is exactly a case where
  redesign guidance is still right, and pure-ratio handles it.
- **No `response_format`** — rely on prompt-instructed JSON + a tolerant
  parser. The Claude backend (`_ClaudeChatCompletions.create`,
  `inference_server.py:60-104`) only forwards a whitelist
  (`temperature`/`top_p`/`stop`) and **silently drops** `response_format`;
  Azure/OpenAI would honor it. Depending on API JSON mode is backend-fragile,
  so the parser never relies on it. Prompt + parser works on all three
  backends.

## Why this is safe / contained

All 6 callers unpack the same 4-tuple and stay unchanged:
`mcts.py:511`, `mcts.py:579`, `small_loop.py:170/194/218`, `large_loop.py:153`.
Downstream semantics preserved: `calculate_score(metric, evaluator_valid)`
(`agent/utils.py:110-128`, `valid is False → (1,0,0)`), the direction bias
(`mcts.py:690-697`, expects exactly `"large"`/`"small"`/`None`), MCTSNode
fields (`mcts.py:66-72`), and `_save_step_log` keys (`mcts.py:798-801`).

## Files modified

| File | Change |
|------|--------|
| `agentprompt/evaluator_prompt.py` | Trim `PROBLEM_STATEMENT` (L13-17) to a neutral role one-liner; compute `mode` **once early** (replacing the `adversarial` var at L213); pass mode-specific `step_type` to `generate_skill_prompt` (`"large"` for Mode A, else `"both"`); gate `STRUCTURAL_ALERT` on `mode != "slow"`; rewrite `GOAL` + `ADVERSARIAL_GOAL` as prose that asks for a JSON response and add a specialized objective opener to `GOAL`; add new `SLOW_GOAL` (prose, JSON response, direction-free schema); replace the L252 binary selector with a mode-driven 3-way append; **return `(prompt, mode)`**. |
| `agent/actions.py` | Add JSON parser helpers; delete `_extract_direction`/`_extract_validity`; rewire `run_evaluator` to unpack `(prompt, mode)` and force `direction="large"` when `mode == "slow"`. Signature + 4-tuple return unchanged. Keep `_extract_tag` (still used by `extract_proposal_kernel`). `json` is already imported (L2). |

**Unchanged (verified):** `agent/inference_server.py`, `agent/utils.py`,
`agent/mcts.py`, `agent/small_loop.py`, `agent/large_loop.py`,
`agentprompt/proposer_prompt.py`, `agentprompt/tuner_prompt.py`,
`agentprompt/Utils/detect.py`.

**Referenced (not modified):** `agentprompt/Utils/__init__.py:72-103`
(`generate_skill_prompt`) — its `step_type` contract is relied on:
`"large"` → `_base` + family Design, `"small"` → `_base` + family Tuning,
`"both"` → `_base` + both. The `_base` section (always emitted) carries the
full no-shortcut contract that the trimmed `PROBLEM_STATEMENT` previously
duplicated.

---

## Step 1 — Mode selection, computed once early (`agentprompt/evaluator_prompt.py`)

`ratio` and `correctness` are already computed at L208-212. The `mode` is now
needed **before** the skill-prompt call (L218) so `step_type` can vary, not just
at the goal-append at L252. So compute `mode` once, replacing the
`adversarial = ratio >= 5.0` var at **L213**:

```python
# mode is the single source of truth for: skill step_type, STRUCTURAL_ALERT
# gating, the goal-block append, and the Mode-A direction force in run_evaluator.
if correctness and 0 < ratio < 0.8:
    mode = "slow"          # Mode A
elif correctness and ratio >= 5.0:
    mode = "adversarial"   # Mode B
else:
    mode = "default"       # Mode C (default + all compile/correctness failures)
```

Then `mode` drives three downstream points:

1. **Skill `step_type`** (L218) — see the new step below.
2. **STRUCTURAL_ALERT gate** (L225) — `if correctness and mode != "slow":`
   (the shortcut alert is irrelevant in Mode A — a slow correct kernel cannot be
   an algebraic shortcut). Equivalent to the prior `not (0 < ratio < 0.8)`.
3. **Goal-block append** (replaces the L252 binary selector), then
   `return prompt, mode`:
   ```python
   prompt += {"slow": SLOW_GOAL, "adversarial": ADVERSARIAL_GOAL}.get(mode, GOAL)
   return prompt, mode
   ```

- Failures have empty `runtime_stats` → `ratio == 0`, `correctness == False`
  → Mode C, preserving today's failure path (`COMPILE_FAILURE` still appended
  at L240-250).
- The `mode` string is the single source of truth for the Mode-A gate; the
  builder's caller (`run_evaluator`) uses it to force `direction="large"`.

## Step 1b — Trim PROBLEM_STATEMENT + per-mode skill `step_type`

**Trim `PROBLEM_STATEMENT` (L13-17)** to the neutral role sentence plus the
"Follow the Optimization Skills below" pointer. Drop the
kernel-tuning / fusion / replace-conv and "never propose graph-level algebraic
shortcuts" clauses — they are redundant:
- The **`_base` skill section is always emitted** by `generate_skill_prompt`
  (any `step_type`) and carries the full no-shortcut contract verbatim.
- `GOAL` (L94) and `ADVERSARIAL_GOAL` (L118) already reference "the `_base`
  skill section describes this contract" and define the `valid` contract.

Verified safe: no component references `PROBLEM_STATEMENT` ("as stated above"
etc.); the goal blocks reference the `_base` skill section, not the preamble.

**Vary the skill `step_type` (L218)** by mode:
```python
prompt += generate_skill_prompt(
    task_params.get("arc_src"),
    step_type=("large" if mode == "slow" else "both"),
    task_params=task_params,
)
```
Contract (`Utils/__init__.py:77-80`): `"large"` → `_base` + family **Design**;
`"both"` → `_base` + family **Design + Tuning**. Mode A is redesign-only, so its
Tuning content is noise; `"large"` drops it while `_base` (the no-shortcut
contract) is still emitted, so no safety content is lost.

## Step 2 — Rewrite goal templates to JSON, each with a specialized opener

Goal templates are appended via plain `prompt += ...`, **never** through
`.format` (only `TASK_INSTRUCTION`, `COMPILE_FAILURE`, `STRUCTURAL_ALERT` are
`.format`-ed, and `_extract_format_keys` only scans `TASK_INSTRUCTION`). So
literal JSON `{`/`}` in these templates need **not** be escaped/doubled — but
do not pass them through `.format`.

With `PROBLEM_STATEMENT` trimmed, each goal block must **open with its own
mode-specialized objective** (no shared generic framing to lean on):
- **`GOAL`** opener: decide whether to **tune or redesign** a kernel that is
  correct-but-moderate, failed, or wrong.
- **`ADVERSARIAL_GOAL`** opener: existing guilty-until-proven anti-shortcut
  framing (already specialized).
- **`SLOW_GOAL`** opener: existing "retained heavy op is the bottleneck —
  redesign only, tuning cannot close the gap" framing (already specialized);
  reinforced now by Design-only skill content.

Each template stays **prose** and ends with an explicit response instruction:
*"Respond with a SINGLE JSON object and nothing else — no prose outside it, no
markdown fences — using exactly these keys: …"*. The prompt body itself is not
JSON; only the evaluator's reply is. Preserve all existing prose semantics,
swapping only the output contract.

- **`GOAL`** (rewrite L63-110) — opens with its specialized objective (decide
  **tune vs. redesign** for a correct-but-moderate / failed / wrong kernel),
  then response keys: `small_guidance` (str), `large_guidance` (str),
  `direction` (`"large"`|`"small"`), `valid` (bool). Keep current meaning of
  each (direction = rewrite-vs-tune judgment; valid = false only on algebraic
  shortcut, true default incl. slow/failed/wrong; default-to-large on compile
  failure).
- **`ADVERSARIAL_GOAL`** (rewrite L112-160) — response keys: `reasoning` (str),
  `small_guidance`, `large_guidance`, `direction`, `valid`. Preserve
  guilty-until-proven prose; `reasoning` must name (a) the full-shape heavy-op
  output axis and (b) the full-MAC arithmetic loop; `valid:true` only when both
  named. `reasoning` is chain-of-thought and is **not** parsed into the tuple.
- **`SLOW_GOAL`** (new, direction-free schema) — response key: **only**
  `large_guidance` (str). **No `direction`, no `small_guidance`, no `valid`** —
  the model is not asked to decide direction; `"large"` is forced in
  `run_evaluator` from the builder-reported `mode == "slow"`. Prose: kernel is
  correct but `fast_p` in (0, 0.8); the retained heavy operator is the
  bottleneck; tuning cannot close the gap; prescribe a concrete Triton strategy
  that **replaces** (not wraps) the heavy op — reuse the per-op hints:
  ConvTranspose*d scatter-add/gather with explicit stride/padding indexing;
  Conv*d tiled im2col-GEMM or direct cross-correlation with a K-loop; Linear
  split-K GEMM with `tl.dot`.

## Step 3 — JSON parser (`agent/actions.py`)

`json` is already imported (L2) — no import change needed.

Add three helpers (mode-agnostic — keyed on **present fields**, never raises):

```python
def _coerce_json_object(output):
    """Best-effort single JSON dict from model output; dict or None."""
    if not output:
        return None
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    for candidate in (text, _last_brace_span(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            pass
    return None

def _last_brace_span(text):
    """Last balanced {...} substring, quote/escape aware; or None."""
    # single scan tracking string state + brace depth; return last depth-0 object

def _parse_evaluator_json(output):
    """-> (small, large, direction, valid); defaults ('', '', None, None)."""
    small, large, direction, valid = "", "", None, None
    obj = _coerce_json_object(output)
    if not isinstance(obj, dict):
        return small, large, direction, valid
    if isinstance(obj.get("small_guidance"), str):
        small = obj["small_guidance"].strip()
    if isinstance(obj.get("large_guidance"), str):
        large = obj["large_guidance"].strip()
    d = obj.get("direction")
    if isinstance(d, str) and d.strip().lower() in ("large", "small"):
        direction = d.strip().lower()
    v = obj.get("valid")
    if isinstance(v, bool):
        valid = v
    elif isinstance(v, str) and v.strip().lower() in ("true", "false"):
        valid = v.strip().lower() == "true"
    return small, large, direction, valid
```

- Delete `_extract_direction` (L28-31) and `_extract_validity` (L53-59).
- **Keep `_extract_tag`** (L19-26) — still used by `extract_proposal_kernel`
  for the `<kernel>` tag.
- In `run_evaluator`, unpack the builder's 2-tuple, parse the JSON response, and
  force the Mode-A direction:
  ```python
  evaluator_prompt, mode = generate_evaluator_prompt(...)
  evaluator_output = query_inference_server(...)
  small_guidance, large_guidance, direction, valid = _parse_evaluator_json(evaluator_output)
  if mode == "slow":
      direction = "large"   # SLOW_GOAL schema omits direction; force the large-bias here
  ```
  Signature (L61) and `return` (L88) unchanged.

### Per-mode → tuple mapping (falls out of the field-keyed parser)

| Mode | JSON keys present | small | large | direction | valid |
|------|-------------------|-------|-------|-----------|-------|
| A slow | `large_guidance` | `""` | parsed | `"large"` (forced in code from `mode`) | `None` |
| B adversarial | reasoning, small, large, direction, valid | parsed | parsed | parsed | parsed |
| C default | small, large, direction, valid | parsed | parsed | parsed | parsed |

- **Mode A direction** is forced in `run_evaluator` from the builder-reported
  `mode == "slow"` (single source of truth for the gate), realizing the
  large-bias through the existing `mcts.py:694-695` path — no parse-time guess,
  no caller change. The SLOW_GOAL schema omits `direction` entirely.
- **Mode A valid → None** → `calculate_score(metric, None)` takes the `else`
  branch → `(1, 1, fast_p)`: the slow kernel keeps its low score, not penalized
  as a cheat.
- **Mode A small → ""** → tuner gets no guidance (redesign-only). Intended.

---

## Verification

Read-only / local checks (no GPU needed):

1. **Import + arity:** `python -c "import agent.actions, agentprompt.evaluator_prompt"`;
   `grep -rn "= run_evaluator(" agent/` → exactly 6 sites, all unpack 4 values.
2. **Builder return + mode probe:** call `generate_evaluator_prompt` with
   synthetic `KernelExecResult`s; assert it returns a `(prompt, mode)` 2-tuple
   and the per-template sentinel phrases are mutually exclusive:
   - correct + `fast_p=0.4` → `mode=="slow"`; SLOW_GOAL present; ADVERSARIAL/GOAL
     + STRUCTURAL_ALERT absent; SLOW_GOAL text contains **no** `direction` /
     `small_guidance` / `valid` key instructions. Skill content shows Design
     sections but **no Tuning** sections (`step_type="large"`).
   - correct + `fast_p=6.0` → `mode=="adversarial"`; ADVERSARIAL_GOAL present;
     skill content shows both Design and Tuning sections (`step_type="both"`).
   - correct + `fast_p=2.0`, and a compile-fail case → `mode=="default"`; GOAL
     present (+ COMPILE_FAILURE on failure); skill `step_type="both"`.
   - **Trim check (any mode):** the assembled prompt's trimmed `PROBLEM_STATEMENT`
     no longer contains the "never propose … algebraic shortcuts" clause, while
     the `_base` no-shortcut contract is still present (carried by the skill
     prompt) in every mode.
3. **Parser probe:** feed `_parse_evaluator_json`:
   - clean object; ```json-fenced object; Mode A object
     `{"large_guidance":"x"}` → `("", "x", None, None)` (then `run_evaluator`
     forces `"large"`); prose + trailing object (recovery path); garbage →
     `("", "", None, None)`; a `large_guidance` value containing literal `{}`.
4. **Forced-direction probe:** stub `query_inference_server` to return a Mode A
   JSON object lacking `direction`; assert `run_evaluator` returns
   `direction=="large"` when the builder reports `mode=="slow"`.
5. **Validity-gate regression:** `calculate_score(correct, None)` → `(1,1,fast_p)`;
   `calculate_score(correct, False)` → `(1,0,0)`.
6. **Live smoke (human/GPU, optional):** one `small_loop` iteration or one MCTS
   step on a known problem; confirm guidance fields populate, no unpack error,
   step log JSON carries the four evaluator keys.

## Risks

- **gpt-5-mini JSON reliability** (default model): malformed JSON degrades to
  `("", "", None, None)` — same neutral behavior as today's missing-tag case.
- **Two JSON fences in one output:** the fence regex takes the first; switch to
  last-fence if observed. `_last_brace_span` covers un-fenced trailing JSON.
- **Malformed Mode A body:** even if the JSON response is unparseable, the
  forced `direction="large"` (set from `mode=="slow"` in `run_evaluator`, not
  from the response) still applies, so the redesign bias survives.
- **PROBLEM_STATEMENT trim relies on `_base`:** the no-shortcut contract now
  lives in the `_base` skill section, which `generate_skill_prompt` always
  emits. If `arc_src` matches no family and the skill prompt returns empty, the
  contract survives only in the goal blocks — confirm `GOAL`/`ADVERSARIAL_GOAL`
  restate it (they do, via the `valid` contract / anti-shortcut prose).

---

# Mode A escape hatch — stop forcing redesign once the heavy op is genuinely replaced (executed-path)

## Context

Mode A ("slow", `correct and 0 < fast_p < 0.8`) assumes the retained PyTorch
heavy op is the bottleneck, so it forces `direction="large"` and emits
redesign-only guidance — steering MCTS to keep calling the proposer for a
from-scratch rewrite. The failure we want to bound: a problem can sit in Mode A
indefinitely, forcing redesign after redesign even when redesign has already
been genuinely attempted and is still slow.

We add a gated **escape hatch**: stop forcing redesign once the agent has
*actually* replaced the heavy op in Triton (in the path that runs) and is still
slow — at that point fall back to the normal `GOAL` mode so tuning (small steps)
becomes reachable again.

### Two constraints (both required to exit Mode A)

1. **Current kernel replaced the heavy op in the executed path** — a `@triton.jit`
   custom implementation of the reference heavy op (Conv*/ConvTranspose*/Linear)
   is *called on the executed forward path*, replacing the `nn.*` call.
2. **A recent large_step also replaced it** — scanning the last 5 entries of
   `self.all_nodes` (global chronological order), at least one `large_step` node
   whose kernel also replaced the heavy op in its executed path.

Only when **both** hold does Mode A release to `GOAL`.

### Locked design decisions

- **Detector = executed-path only.** The eval harness (`src/eval.py`) never calls
  `.eval()`/`.train()`, so models run at PyTorch default `training=True`. A Triton
  op guarded behind `if not self.training:` / an `else:` branch is **dead code**
  and does **not** count as a replacement. The detector resolves the
  `if self.training:` guard and inspects only the live branch.
- **Both constraints required** to exit (conservative — stays in Mode A longer).
- **Accepted limitation:** the motivating case `2_15` will *remain* in Mode A,
  because its "replacement" kernels (steps 2/4/5/9) park the Triton conv in the
  never-executed `else` branch — they never replaced the op in the measured path.
  This is the *correct* verdict under executed-path semantics; this change is a
  **general anti-stuck mechanism**, not a `2_15`-specific patch.

## Files modified

| File | Change |
|------|--------|
| `agentprompt/Utils/detect.py` | New `replaced_heavy_op_in_executed_path(kernel_src, arch_src) -> bool`. Reuses `_find_model_class`, `_build_init_map`, and the `_HEAVY_CONV`/`_HEAVY_LINEAR` sets. Resolves the `if self.training:` guard in `ModelNew.forward`, walks only the live branch, returns True iff a `@triton.jit`/`tl.*` custom kernel call replaces the heavy op there (the heavy `nn.*`/`F.*`/`self.<heavy>` call is **absent** from the live branch while a triton kernel launch is present). Fail-safe to `False` on any parse error / missing `ModelNew.forward`. |
| `agentprompt/evaluator_prompt.py` | `generate_evaluator_prompt` gains optional `redesign_exhausted: bool=False`. In the mode-selection block (L227), when the slow gate matches **and** `redesign_exhausted` is True, choose `"default"` instead of `"slow"`. Return tuple unchanged `(prompt, mode)`. |
| `agent/actions.py` | `run_evaluator` gains optional `redesign_exhausted: bool=False`; forwards it to `generate_evaluator_prompt`. The `if mode == "slow": direction="large"` force is untouched (only fires when still in slow mode). 5-tuple return unchanged. |
| `agent/mcts.py` | New private helper `MCTS._redesign_exhausted(new_kernel) -> bool` that ANDs constraint #1 (detector on `new_kernel`) with constraint #2 (detector on any `large_step` in `self.all_nodes[-5:]`), using `self.ref_arch_src` as `arch_src`. Both `run_evaluator` call sites (`:517` large, `:587` small) compute it and pass `redesign_exhausted=`. |

**Unchanged:** the 4 loop call sites (`large_loop.py:153`, `small_loop.py:170/197/223`)
keep calling `run_evaluator` without the new arg → `redesign_exhausted=False` →
Mode A behavior there is exactly as today (these loops keep no per-step type, and
`large_loop` already ignores `direction`).

## Detector algorithm (`replaced_heavy_op_in_executed_path`)

1. Reference heavy op names: scan the reference `forward` (via `detect_shortcut_risk`
   plumbing / `_HEAVY`) for the heavy op names the reference uses (e.g.
   `{"ConvTranspose3d"}`).
2. Parse `kernel_src`; find `ModelNew` (`_find_model_class` pattern) and its
   `forward`. Build `init_map` (`_build_init_map`) so `self.conv_transpose`
   resolves to `ConvTranspose3d`.
3. **Select the live branch.** Walk `forward.body`. For an `If` whose test is
   `self.training` (or `not self.training`), keep only the live arm
   (`training=True` ⇒ the `if` body for `self.training`, the `else` body for
   `not self.training`). Statements outside any training guard are always live.
4. On the live statement stream detect (a) a triton kernel launch
   (`name[grid](...)` subscript-call, or a call to a module-level `@triton.jit`
   function / a `tl.dot`-bearing helper) and (b) absence of the reference heavy
   `nn.*`/`F.*`/`self.<heavy>` call. Return `True` iff a custom replacement is
   present **and** the heavy op call is absent from the live branch.
5. Fail safe: any parse error or no `ModelNew.forward` → `False` (never wrongly
   exit Mode A).

## Verification

Read-only / local (no GPU):

1. **Import + arity:** `python -c "import agent.actions, agent.mcts, agentprompt.evaluator_prompt, agentprompt.Utils.detect"`;
   `grep -rn "run_evaluator(" agent/` → 6 sites, all still unpack 5 values.
2. **Detector unit probes** (on real `2_15` artifacts):
   - `step_4.py`, `step_2.py`, `step_5.py`, `step_9.py` (Triton conv in the
     non-training/`else` branch) + `reference_src.py` → all **False** (dead code;
     executed path still calls `nn.ConvTranspose3d`).
   - `global_best_kernel_10.py` (wraps conv) → **False**.
   - Hand-written kernel calling a `@triton.jit` conv unconditionally in `forward`
     → **True**; same conv under `if self.training:` (live branch) → **True**.
3. **Mode-selection probe:** `generate_evaluator_prompt` with synthetic correct +
   `fast_p=0.4`: `redesign_exhausted=False` → `mode=="slow"` (SLOW_GOAL present,
   unchanged); `redesign_exhausted=True` → `mode=="default"` (GOAL present,
   SLOW_GOAL absent).
4. **mcts gate probe:** stub `all_nodes` with a `large_step` node whose kernel
   replaces the op (live branch) + a current kernel that also replaces it →
   `_redesign_exhausted` True; flip either → False (both-required semantics).
5. **`2_15` regression (documented):** detector across all `2_15` steps yields
   `_redesign_exhausted == False` everywhere ⇒ `2_15` stays in Mode A — expected
   and accepted, not a failure.
6. **Live smoke (optional, GPU):** one MCTS run on a problem where the agent
   genuinely replaces a Conv/Linear in the live path and stays <0.8× → confirm the
   evaluator switches from SLOW_GOAL to GOAL and a small step becomes reachable.

## Risks

- **Branch resolution is heuristic.** Only `self.training`-keyed `If` guards are
  resolved; exotic guards (a stored flag, `torch.is_grad_enabled()`) fall through
  as "always live" — the safe direction (worst case treats dead code as live →
  slightly more likely to exit; rare and acceptable).
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
prompts. Earlier sections of this doc only *designed* the executed-path detector;
this section is the executable fix.

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
| `src/eval.py` | Add a `TorchDispatchMode` subclass `_HeavyOpRecorder` + a module-level `_HEAVY_ATEN` allowlist (`aten::convolution`, `aten::_convolution`, `aten::conv_transpose3d`, `aten::addmm`, `aten::mm`, `aten::bmm`, `aten::linear`, …). In `run_and_check_correctness` (L801) wrap the **already-executed trial-0 forwards** — `model(*inputs)` (L853) and `model_new(*inputs)` (L858) — each in `with _HeavyOpRecorder() as rec:` (no extra passes). Set `metadata["heavy_op_executed_in_pytorch"] = bool(ref_heavy) and ref_heavy.issubset(new_heavy)` and `metadata["executed_heavy_ops"]`. Fully try/except-guarded → defaults `False` on any error. |
| `agentprompt/Utils/detect.py` | `replaced_heavy_op_built(kernel_src, arch_src)` — branch-agnostic: does `ModelNew` define/launch a `@triton.jit`/`tl.*` custom impl of the reference heavy op anywhere (not just the original `nn.*`/`F.*` call or an `F.conv_*` alias)? Used by the gate. `replaced_heavy_op_in_executed_path(kernel_src, arch_src)` — the live-branch detector designed earlier in this doc (resolves the `if self.training:` guard, inspects only the live arm; **False** for the 2_15 cheat). Both reuse `_find_model_class`/`_build_init_map`/`_HEAVY*`; fail-safe to `False`. Invariant: a kernel can't be both `in_executed_path=True` and a dead-branch rewrite. |
| `agent/actions.py` | In `run_evaluator` (L129) after the JSON parse (L153): `dead_branch = bool(metrics) and metrics.metadata.get("heavy_op_executed_in_pytorch") and replaced_heavy_op_built(kernel, ref_arch_src)`; if `dead_branch: valid = False`. Pass `dead_branch` into `generate_evaluator_prompt`. 5-tuple return + 6 call sites unchanged. |
| `agentprompt/evaluator_prompt.py` | Add `DEAD_BRANCH_ALERT` (sibling of `STRUCTURAL_ALERT`) + a `dead_branch_rewrite: bool=False` param to `generate_evaluator_prompt`; inject the alert **regardless of mode** (the cheat lands in Mode A, where `STRUCTURAL_ALERT` is skipped at L249). Inject near L249–255. |
| `agentprompt/skills/_base.md` (+ `proposer_prompt.py` ~L37, `tuner_prompt.py` ~L30/L69) | New rule reaching proposer/tuner/evaluator via `generate_skill_prompt`: the harness runs models at `training=True`; the heavy-op replacement must execute unconditionally; do **not** guard it behind `if self.training` / `if x.is_cuda` / `else:` fallbacks — such branches are dead code and count as *not* replacing the op. |

Gate mechanics: `calculate_score(metric, False) → (1,0,0)` (`agent/utils.py:124`), so
the cheat sorts like an incorrect kernel and cannot win as best. Remote eval needs
**no code change** — metadata rides back through `KernelExecResult(**result)`
(`src/eval.py:1044`); the remote server must run this updated code.

## Mode-A reconciliation

Wire `replaced_heavy_op_in_executed_path` into the Mode-A redesign accounting (the
escape hatch above) so a dead-branch "rewrite" is **not** counted as a genuine
redesign and the problem isn't prematurely released from redesign guidance.

## Verification

Read-only / local (no GPU):

1. `python -c "import agent.actions, agentprompt.evaluator_prompt, agentprompt.Utils.detect, src.eval"`;
   `grep -rn "run_evaluator(" agent/` → 6 sites, all unpack 5 values.
2. `detect.py` probes on real `2_15` artifacts: `replaced_heavy_op_built` →
   **True** for steps 2/4/5/6/9, **False** for 1/3/7; `replaced_heavy_op_in_executed_path`
   → **False** for all (live branch keeps the aten conv). Hand-written kernel
   calling a `@triton.jit` conv unconditionally → both **True**.
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
