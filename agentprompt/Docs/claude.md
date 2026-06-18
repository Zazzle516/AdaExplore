# Plan — Tighten agent prompts: ban graph-level algebraic shortcuts, authorize custom Conv kernels

## Context

The TRT baseline run on level2 (`outputs/KB-l2_AdaExplore_50/Summary.xlsx`) surfaces two
opposite-direction failure modes that share a single root cause: the prompts in
`agentprompt/` define the agent's job in a way that mismatches what we actually want it
to optimize.

- **High-ratio rows (tid 13 / 14 / 18 / 42 / 44 / 50 / 51 / 80 / 83 / 98 — ratio 19×–700×).**
  The agent rewrites the reference graph using algebraic identities so the heavy op
  (matmul / conv-transpose) **doesn't run at runtime**. The math is hoisted into init-time
  precomputed weights, and forward becomes a tiny GEMM or matvec. Examples we read:
  - `outputs/KB-l2_AdaExplore_50/2_14`: `linear(x).sum(dim=1)` → `x · W.sum(dim=0)` (8000× FLOP cut).
  - `outputs/KB-l2_AdaExplore_50/2_44`: `mean(2,3) ∘ ConvTranspose2d` is folded into the
    weight; the conv-transpose itself never executes.
  TRT cannot follow these rewrites — it optimizes within tensor-algebra equivalence, not
  operator-elimination identities — so the ratio explodes.

- **Low-ratio rows (tid 3 / 15 / 34 / 36 / 38 / 49 / 60 / 72 / 77 / 79 — ratio 0.15–0.65).**
  The agent fuses the cheap epilogue with a Triton kernel and leaves the conv-transpose
  as `self.conv_transpose(x)`. TRT auto-tunes the conv tactic, fuses bias/epilogue,
  and runs under CUDA Graphs — so it wins. The agent has no instruction (and no
  examples) telling it that conv-class kernels are in scope.

The proposer/reviser/tuner prompts read end-to-end (proposer_prompt.py:41,
reviser_prompt.py:21, tuner_prompt.py:32, plus the reviser Goal at reviser_prompt.py:57)
**explicitly invite both behaviors**: each prompt encourages "restructuring computation
when mathematically equivalent" with the worked example "fold normalization into
preceding linear layers." That is the literal pattern producing the high-ratio rows.
None of the prompts authorize replacing `nn.Conv*` / `nn.ConvTranspose*` with a custom
kernel, so the low-ratio rows have no path to close.

This plan rewords those four locations to (1) ban graph-level shortcuts that elide the
heavy op while keeping ordinary, in-place fusions, and (2) explicitly authorize and
recommend custom Conv/ConvTranspose kernels when the torch conv dominates runtime.

## Scope

Prompt-text edits only. Four files:
- `agentprompt/proposer_prompt.py`
- `agentprompt/reviser_prompt.py`
- `agentprompt/tuner_prompt.py`
- `agentprompt/prompt_modules.py` *(new shared block — see Design)*

Out of scope (called out at the end):
- The tuner's `str_replace`-only edit mode is a structural constraint that prevents the
  agent from swapping a 3-line `self.conv(...)` call for a 100+ line custom kernel mid-
  trajectory. Prompt wording can't fix that. Flagged as a follow-up.
- The MCTS proposer/expansion logic in `agent/mcts.py` is unchanged.
- `KB_TRITON_PROMPT` in `agentprompt/benchmarks/KB_prompt.py` is already minimal; we
  layer the new rules in the upstream `PROBLEM_STATEMENT` blocks instead of duplicating
  them per benchmark.

## Design

### One shared rules block, three reuse sites

Add a single `OPTIMIZATION_RULES` constant in `agentprompt/prompt_modules.py` so the same
text is sent to proposer, reviser, and tuner without drift. The three call sites already
import from `prompt_modules` (proposer_prompt.py:2, reviser_prompt.py:2,
tuner_prompt.py:23) so no new wiring is needed — just append the block after
`PROBLEM_STATEMENT` and before the existing experience-guidance / hardware sections.

The block has two halves, mirroring the user's two requirements:

**Half 1 — what the agent must implement (ban graph-level algebraic shortcuts).**
Concretely the rule that catches all five tid 13/14/18/42/44 rewrites is: *every operator
in the reference forward must execute at runtime on the actual input tensor; you may not
precompute a reduction or a downstream-collapse identity into a constant at init time
so that forward skips the heavy op.* This deliberately allows ordinary fusion (`bias +
activation`, BN-into-conv at inference, online softmax, log-sum-exp shift) while
forbidding the specific class of rewrites we saw — `W' = W.sum(dim=0)`, `W' =
weight.sum(dim=(2,3))`, conv-transpose-folded-into-mean, etc.

**Half 2 — what the agent is allowed to write (custom Conv/ConvTranspose kernels).**
Spell out that `nn.Conv2d`, `nn.ConvTranspose2d`, and `nn.ConvTranspose3d` are valid
replacement targets, not fixed primitives. When the recorded baseline shows
`fast_p < 0.8` and the epilogue is already fused, the next attempt should propose a
custom conv (im2col-+-GEMM, or for transposed conv a scatter-add over the input × kernel
outer product). This addresses both the proposer (so the seed kernel can be conv-class)
and the reviser (so its guidance can recommend "write a custom conv" rather than
"toggle cudnn.benchmark," which is what reviser tid=87 step 4 actually emitted).

### Exact text to add to `prompt_modules.py`

```python
OPTIMIZATION_RULES = """## Optimization Rules

The reference architecture is a *contract*: every operator in the reference forward must
execute at runtime on the actual input tensor. Your job is to make those operators run
faster — not to make some of them disappear.

### Forbidden: graph-level algebraic shortcuts

Do **not** replace a heavy operator with a cheaper one by exploiting identities derived
from downstream reductions. Specifically forbidden:

- Folding a post-`Conv*`/`ConvTranspose*` global reduction (`mean`/`sum` over spatial
  dims, `mean`/`sum` over a depth/channel slice) into a per-element rewrite of the
  weight tensor at init time, so that forward executes a small GEMM instead of the
  convolution.
- Replacing `linear(x).sum(dim=...)` with `x @ weight.sum(dim=...)` (or any analogous
  identity that turns a `(B, in) × (in, out)` matmul into a `(B, in)` matvec by
  precomputing a column/row sum of the weight).
- Any rewrite that, on inputs of the documented shape, turns the work of a `Conv2d` /
  `ConvTranspose2d` / `ConvTranspose3d` / `matmul` / `Linear` into an operator with
  strictly fewer FLOPs by collapsing it with a downstream reduction.

These rewrites pass the loose `atol=rtol=5e-2` correctness check but are not the
optimization target — the comparison baseline is a strong inference engine that runs the
graph as written.

### Permitted: kernel-level fusion and standard constant-folding

You **may**:

- Fuse adjacent operators into a single Triton kernel (e.g., `bias + activation`,
  `LayerNorm + GELU`, `conv-bias + ReLU`).
- Reorder elementwise ops when the reordering does not change which heavy op runs
  (e.g., merge two scalar subtractions into one).
- Apply standard inference-time folds: BatchNorm-into-Conv (eval mode), scalar
  multipliers absorbed into adjacent affine ops, identity activations elided.
- Use numerical-stability rewrites: online softmax, log-sum-exp shift, RMS-norm via
  `rsqrt`.
- Cache parameter-derived tensors that are *the same shape* as the original parameter
  (not collapsed by a reduction).

### Authorized: custom Conv / ConvTranspose kernels

`nn.Conv2d`, `nn.ConvTranspose2d`, and `nn.ConvTranspose3d` are **not** off-limits.
When the torch convolution dominates runtime — symptom: `fast_p < 0.8` while the
epilogue is already fused — the right next step is to **replace the conv with a custom
Triton (or CUDA-cpp via `torch.utils.cpp_extension`) kernel**, not to keep tuning the
tail. Reasonable starting points:

- Conv2d: im2col-style indexing into a tiled GEMM, one program per `(N, OC tile, H tile,
  W tile)`.
- ConvTranspose2d/3d: scatter-add of the input × kernel outer product into the output
  grid, with stride / padding / output_padding handled in the index math.

Custom conv kernels are large but supported by the framework. Do not silently assume the
conv is the immovable part of the graph — it is exactly the part with the most headroom.

"""
```

### Where each prompt picks it up

| File | Current text to remove | Where to insert `OPTIMIZATION_RULES` |
|---|---|---|
| `agentprompt/proposer_prompt.py:41` | The sentence beginning *"You may also reorder mathematically equivalent operations..."* through *"...folding normalization parameters into preceding linear layers"* | Append `OPTIMIZATION_RULES` to `prompt` immediately after `PROBLEM_STATEMENT`, before `EXAMPLE_FORMATS`, in `generate_proposer_prompt` (proposer_prompt.py:88-93). |
| `agentprompt/reviser_prompt.py:21` | *"Beyond kernel-level tuning, also consider whether reordering mathematically equivalent operations could enable better fusion or memory access patterns."* | Append `OPTIMIZATION_RULES` to `prompt` after `PROBLEM_STATEMENT`, before `generate_experience_guidance_prompt(...)`, in `generate_reviser_prompt` (reviser_prompt.py:99-100). |
| `agentprompt/reviser_prompt.py:57` | The `### Goal` paragraph that lists *"reordering elementwise ops, folding normalization into linear layers"* as the recommended rewrites | Replace the example list with: *"Improvements may include low-level kernel optimizations (tiling, vectorization, occupancy), kernel-level fusion (combining adjacent operators into one Triton kernel), or — when the torch convolution dominates runtime — replacing `nn.Conv*` / `nn.ConvTranspose*` with a custom Triton kernel. Do not propose graph-level algebraic shortcuts that elide a heavy operator (see the Optimization Rules above)."* |
| `agentprompt/tuner_prompt.py:32` | *"Beyond low-level kernel tuning, you may also restructure the computation order when it is mathematically equivalent and can unlock better fusion or memory access patterns."* | Append `OPTIMIZATION_RULES` to `prompt` after `PROBLEM_STATEMENT`, before `generate_experience_guidance_prompt(...)`, in `generate_tuner_prompt` (tuner_prompt.py:210-211). Also soften the `Goal` at tuner_prompt.py:69 from *"Keep the overall model interface unchanged, but you may reorder or restructure internal operations when mathematically equivalent"* → *"Keep the overall model interface unchanged. Apply the Optimization Rules above — do not introduce algebraic shortcuts that eliminate the reference's heavy operators."* |

### Why all three prompts get the same block

The three agents form a feedback loop: the **proposer** produces the seed, the
**reviser** suggests improvements, the **tuner** applies them via `str_replace`. If only
the proposer were constrained, the reviser would still suggest a `weight.sum(...)`
shortcut in step 5 and the tuner would apply it. If only the tuner were constrained,
the proposer's seed would already contain the shortcut. Single source of truth in
`prompt_modules.py` keeps them aligned.

## Files to modify

| Path | Change |
|---|---|
| `agentprompt/prompt_modules.py` | Add the `OPTIMIZATION_RULES` constant at module scope (insert after the existing `HARDWARE_INFORMATION` block, before `generate_experience_guidance_prompt`). |
| `agentprompt/proposer_prompt.py` | Strip the algebraic-rewrite invitation from `PROBLEM_STATEMENT` (line 41), import `OPTIMIZATION_RULES` from `prompt_modules`, append it in `generate_proposer_prompt` (line 89). |
| `agentprompt/reviser_prompt.py` | Strip the rewrite line from `PROBLEM_STATEMENT` (line 21), update the `Goal` paragraph (line 57), import and append `OPTIMIZATION_RULES` in `generate_reviser_prompt` (line 99). |
| `agentprompt/tuner_prompt.py` | Strip the rewrite line from `PROBLEM_STATEMENT` (line 32), update the `Goal` paragraph (line 69), import and append `OPTIMIZATION_RULES` in `generate_tuner_prompt` (line 210). |

## What is intentionally NOT changed

- **`agentprompt/benchmarks/KB_prompt.py`.** The KB-specific template is already minimal
  ("Optimize the architecture named Model with custom Triton kernels!"). The new rules
  apply to all benchmarks (KB, FIT, TBG, SYN, MLSYS), so they belong upstream.
- **The `str_replace` constraint in the tuner** (tuner_prompt.py:69-77 and the
  `Output Format` section). Prompt wording cannot let a 3-line `self.conv(x)` site grow
  into a 200-line Triton kernel via local edits. A real fix would let the tuner emit a
  full-file replacement when the reviser's guidance asks for a conv kernel rewrite —
  this is a code change in `agent/mcts.py` plus the tuner's edit-extraction logic
  (`extract_edits` in `agent/utils.py`), not a prompt change. Flag this to the user as
  a follow-up; do not try to bend prompt text around a structural limit.
- **Few-shot examples under `agentprompt/examples/`.** `model_ex_add.py` and
  `model_new_ex_add_triton.py` are minimal elementwise demos — they don't show
  algebraic simplification (good or bad), so they don't need to change. Adding a
  conv-class example would be valuable but is a larger task; the prompt rules alone
  are sufficient to ban the bad pattern.
- **`MEMORY.md` / saved memories.** No persistent guidance to update; this is a project
  prompt change, not a Claude-collaboration preference.

## Verification

End-to-end check after editing the four files:

1. **Smoke-rendering.** Run `python agentprompt/proposer_prompt.py`,
   `python agentprompt/reviser_prompt.py`, and `python agentprompt/tuner_prompt.py` (each
   has a `__main__` that prints the rendered prompt). Confirm:
   - `## Optimization Rules` appears once in each rendered prompt.
   - The old "folding normalization into preceding linear layers" sentence no longer
     appears anywhere.
   - The reviser's `### Goal` mentions custom Conv kernels.
   - Format substitution still works (no `KeyError` on `{arc_src}`, `{custom_triton_kernels}`,
     `{previous_kernels_and_metrics}`, `{tuning_guidance}`).

2. **Re-run a high-ratio offender.** Pick **tid=14** (`Gemm_Divide_Sum_Scaling`, ratio
   177×). Run the same MCTS config that produced
   `outputs/KB-l2_AdaExplore_50/2_14/global_best_kernel_50.py`. Inspect the new
   `global_best_kernel_50.py`:
   - Pass: forward path computes `x @ self.weight.T` at runtime (or a tiled Triton
     matmul) and applies the trailing `sum(dim=1)` on the full `(B, hidden)` tensor.
   - Fail: forward still references a precomputed `w_reduced` / `w_sum` / `weight.sum`
     tensor as the only operand of the runtime matmul.

3. **Re-run a low-ratio offender.** Pick **tid=15** (`ConvTranspose3d_BatchNorm_Subtract`,
   ratio 0.32). Inspect the new `global_best_kernel_50.py`:
   - Pass: at least one MCTS step proposes a custom Triton conv-transpose kernel (the
     reviser's guidance text in `step_*_prompt.txt` should explicitly mention writing
     such a kernel when the conv dominates).
   - Acceptable: the search may still settle on `self.conv_transpose(x)` if the custom
     kernel doesn't beat cuDNN — but the *trajectory* should show the agent attempting
     the replacement, not silently leaving the conv as a torch primitive.

4. **Spot-check correctness still passes.** The new rules don't change what's correct,
   only what's allowed. Existing `(5/5)` correctness on tid=14, tid=15 should still
   hold for any kernel the agent produces.

5. **Check the reviser's guidance text changed.** Re-run tid=87 (`Conv2d_Subtract_Subtract_Mish`)
   and read the step-4 reviser guidance in `step_4_prompt.txt`. Old: "try
   `cudnn.benchmark` or `channels_last`." New target: "the conv dominates — propose a
   custom Triton conv2d kernel as the next attempt."
