# Harden the evaluator against high-ratio algebraic shortcuts

## Context

`agentprompt/evaluator_prompt.py` decides whether an LLM-generated Triton kernel does the same work as the PyTorch reference. The `<valid>true|false</valid>` decision currently has a strong bias toward `true`:

- The prompt at `evaluator_prompt.py:106-113` declares `true` the default and treats a missing/malformed tag as valid.
- `agent/actions.py:53-59` (`_extract_validity`) returns `None` on missing tag, and `agent/utils.py:110-128` (`calculate_score`) only collapses the score on an explicit `False`.
- No scrutiny scales with speedup. A kernel reporting `fast_p = 1.5` and one reporting `fast_p = 50` get the same prompt, even though the latter is almost certainly an algebraic shortcut (Conv/Linear pre-reduced into a downstream `sum/mean/AvgPool`, as forbidden by `agentprompt/skills/_base.md`).

We add two layers of defense without changing MCTS scoring or the proposer/tuner/reviser interfaces:

1. **Adversarial review at `fast_p ≥ 5`** — the prompt switches to a template whose default is `<valid>false</valid>` and which demands the model name the specific axis collapse before declaring validity.
2. **Static-analysis priming** — strip self-justifying comments/docstrings from the kernel before evaluation, and (when the reference contains a `heavy → reduction` chain) inject a Structural Alert section naming the expected intermediate shape and the specific shortcut pattern to look for.

The hard gate on `<valid>false</valid>` stays. The default-true behavior of `_extract_validity` stays (the adversarial prompt explicitly demands the tag, so a missing tag is a different problem from prompt bias).

## Design

### Trigger

```
ratio = run_info.runtime_stats.get("fast_p", 0) or 0   # handles None
adversarial = ratio >= 5.0
```

Compile/correctness failures have empty `runtime_stats` (`src/eval.py:625` only populates `fast_p` on the successful perf branch), so they naturally bypass adversarial mode and continue to use the existing `COMPILE_FAILURE` + `GOAL` flow. The Structural Alert is also skipped on `not correctness` to keep the failure-path prompt focused.

### Static analysis: reference-side only

The reference arch source lives in `task_params["arc_src"]` (confirmed populated in every entry path). We walk the AST of `Model.forward` and look for chains where a heavy op feeds into a linear reduction, possibly through pointwise ops:

- **Heavy ops:** `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.Linear`, `torch.matmul`, `F.conv2d`, `F.linear`.
- **Allowed transparent ops between:** activations (`relu`, `sigmoid`, `gelu`, …), `bias` add, `view`/`reshape`/`permute`, `clamp`. Anything else breaks the chain.
- **Reductions:** `sum`, `mean`, `nn.AvgPool*d`, `nn.AdaptiveAvgPool*d`, `F.avg_pool*d`, `F.adaptive_avg_pool*d`.

Shape derivation is **constructor-only** — we read `nn.Linear(in, out)` and `nn.Conv2d(in_channels, out_channels, kernel_size, …)` from `Model.__init__` without executing `get_inputs()`. For Conv we report `(N, out_channels, H_out, W_out)` and explicitly say H/W could not be derived statically. The taxonomy ("conv-output before mean reduction") is what matters; exact spatial dims don't change the alert wording.

### Comment stripping

Token-level pass (`tokenize.tokenize` on a `BytesIO` of the kernel) drops `COMMENT` tokens. A small AST pass then removes module/class/function docstrings (`ast.Expr` with `Constant(str)` at the head of the body) while leaving other top-level string literals (e.g. `cpp_source = """..."""` assignments, which are `ast.Assign`, not `ast.Expr`) intact.

### Adversarial prompt

Replace `GOAL` with `ADVERSARIAL_GOAL` when `adversarial` is true. The new template:

- Opens with explicit suspicion: "The measured speedup (≥5×) is in a range commonly produced by algebraic shortcuts. You are in adversarial review mode."
- Restates the validity contract from `_base.md` more aggressively.
- **Inverts the default:** "In adversarial mode, `<valid>true</valid>` is only emitted when you can name (a) which heavy-op output axis is materialized at full shape and (b) which arithmetic loop in the kernel performs the reference's full MAC count. Otherwise emit `<valid>false</valid>`."
- Carves out legitimate fusion: "A kernel that fuses the heavy op and the downstream reduction into one pass is still valid — the contract is violated only when the heavy op's work axis is collapsed before the heavy op runs."
- Demands a `<reasoning>` block before `<valid>` for traceability.

### Structural alert

When `detect_shortcut_risk(arc_src)` returns at least one chain, inject before `TASK_INSTRUCTION`:

```
## Structural alert

The reference contains a {heavy_op} → {reduction} chain.
Expected intermediate output of {heavy_op}: {shape_descriptor}
(then reduced to the kernel's reported output by {reduction}).

Common shortcut surface: pre-reducing {heavy_op}'s weight or input
along the axis that {reduction} later collapses, then running a smaller
GEMM/matvec. Legitimate fusion of {heavy_op} and {reduction} inside one
Triton kernel is allowed; the contract is violated only when the
heavy-op work axis is collapsed before the multiply-accumulate runs.

Verify the kernel's main accumulation loop iterates over the full
{heavy_op} output grid.
```

## Files to change

| File | Change |
|---|---|
| `agentprompt/_kernel_cleanup.py` (new) | `strip_comments_and_docstrings(src: str) -> str` |
| `agentprompt/skills/detect.py` | Add `detect_shortcut_risk(arch_src: str) -> list[ShortcutChain]` reusing existing AST infrastructure (`FAMILY_MAP`, `BUILTIN_IGNORE`, the walk from `detect_families`). |
| `agentprompt/evaluator_prompt.py` | Add `ADVERSARIAL_GOAL` and `STRUCTURAL_ALERT` template constants. In `generate_evaluator_prompt`: strip kernel comments before the format dict is built; compute `ratio`/`adversarial`; inject structural alerts after `HARDWARE_INFORMATION` and before `TASK_INSTRUCTION` (gated on `correctness`); choose `ADVERSARIAL_GOAL` vs `GOAL` based on `adversarial`. |
| `agentprompt/evaluator_prompt.py` `__main__` | Add a second smoke that constructs a `KernelExecResult` with `fast_p=12.0` to visually verify the adversarial prompt assembles correctly. |

No changes to `agent/actions.py`, `agent/mcts.py`, `agent/utils.py`, `src/eval.py`, the proposer, tuner, or reviser.

## Edge cases

- **No heavy→reduction chain in reference:** `detect_shortcut_risk` returns `[]`, alert omitted.
- **Compile failure:** `runtime_stats` empty → `ratio = 0` → adversarial off; existing `COMPILE_FAILURE` flow unchanged. Structural Alert also skipped on `not correctness`.
- **Missing/None `fast_p`:** `runtime_stats.get("fast_p", 0) or 0` handles both.
- **Ratios in `(1.0, 5.0)`:** unchanged.
- **`inf` ratio** triggers adversarial; **`nan`** does not (acceptable — usually mid-failure).
- **`arc_src` malformed** (e.g. JSON-encoded in FIT path): `ast.parse` is wrapped in try/except, returns `[]` gracefully.
- **Kernel with syntax errors that breaks `ast.parse`** during comment stripping: fall back to returning the original source unchanged.

## Risks

- **False negatives on honestly-fused fast kernels at `fast_p ≥ 5`.** Mitigated by the explicit fusion carve-out in `STRUCTURAL_ALERT` and by the `ADVERSARIAL_GOAL` requirement that the model name a specific axis collapse before emitting `<valid>false</valid>`. Worth monitoring rejection rates after deploy — check `outputs/` logs for kernels that ship `fast_p ≥ 5` with `valid=false` and audit a sample.

## Verification

1. **Unit-level sanity** (manual run, no test framework changes):
   - `python -m agentprompt.evaluator_prompt` — confirm `__main__` still prints a coherent prompt for `fast_p=1.0` and for the new `fast_p=12.0` smoke.
   - Construct a `Model` with `Conv2d → AvgPool2d`, run `detect_shortcut_risk` on its source — expect a chain returned. Repeat with `Conv2d → ReLU → BiasAdd` — expect `[]`.

2. **Comment stripping smoke:**
   - Feed a kernel containing `# Fused conv+mean shortcut — mathematically equivalent` and a triple-quoted module docstring — confirm both gone.
   - Feed a kernel containing `cpp_source = """..."""` — confirm survived.

3. **End-to-end against KernelBench level 2:**
   - Run the agent against `datasets/KernelBench/level2/14_Gemm_Divide_Sum_Scaling.py` (a known shortcut-prone problem). With the current evaluator the agent is documented to ship `<valid>true</valid>` on a shortcut kernel (see `DEBUG_level2_*.md`). After the change, confirm a high-ratio shortcut kernel gets `<valid>false</valid>` at least one time in the first few iterations.
   - Run against `datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py` (no reduction in the reference). Confirm Structural Alert is omitted and ratios in `(1.0, 5.0)` use the existing prompt unchanged.
