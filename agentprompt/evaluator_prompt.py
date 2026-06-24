import re
import json
from agentprompt.prompt_modules import generate_experience_guidance_prompt
from agentprompt.Utils import generate_skill_prompt
from agentprompt.Utils.detect import detect_shortcut_risk, strip_comments_and_docstrings
from src.eval import KernelExecResult

def _extract_format_keys(template: str):
    """Extract format keys from a template string."""
    return set(re.findall(r'\{(\w+)\}', template))

PROBLEM_STATEMENT = """## Problem Statement

You evaluate the most recent custom Triton kernel and its measured performance, then emit guidance for the next iteration. Follow the Optimization Skills below.

"""

TASK_INSTRUCTION = """## Task Instruction

You are given the following architecture:

```
{arc_src}
```

The input shapes can be found in the input of the architecture, and the dtype is {dtype_str}.

An agent generated the following custom Triton kernels in the architecture:

```
{custom_triton_kernels}
```

The runtime metrics of the custom Triton kernels are:

```
{run_info}
```

The tuning metrics contain the following information:

* **Compiled**: whether the kernel is compiled successfully
* **Error Message**: the compilation or runtime error encountered by the kernel (if any)
* **Correctness**: whether the kernel is correct
* **Runtime**: the runtime of the kernel
* **Fast_p**: compared with the standard PyTorch implementation, how much speedup the customized kernel achieves, calculated as *standard time / custom time*.

"""

COMPILE_FAILURE = """### Compile failure

The kernel failed to compile. Traceback / error:

{compile_error}

Your <large_guidance> should diagnose the structural cause; your
<small_guidance> may suggest a minimal patch if one is obvious, or
restate that a redesign is needed.

"""

GOAL = """### Goal

Your objective is to decide whether the next iteration should **tune** the
current kernel or **redesign** it from scratch, for a kernel that is
correct-but-moderate, compile-failed, or numerically wrong. Produce concrete
guidance for both possibilities, judge which is more promising, and certify
that the kernel actually does the reference work.

Provide:

* **small_guidance** — 1-3 concrete tuning bullets (block sizes, num_warps,
  num_stages, layout, autotune configs, fusion opportunities within the
  existing kernel structure). Consumed by the tuner if a refinement step is
  chosen.
* **large_guidance** — 1-3 concrete design bullets (what a from-scratch rewrite
  should change about the kernel's strategy: im2col vs. direct, scatter-add vs.
  gather, fusion boundary, persistence). Consumed by the proposer if a redesign
  step is chosen.
* **direction** — `"large"` if the current kernel is structurally unable to
  reach the target and a from-scratch rewrite is more promising than continued
  tuning; `"small"` if continued tuning of the current kernel will close the
  gap. On compile failure, default to `"large"` unless the traceback points at a
  trivially-fixable issue (e.g. a typo, a wrong constexpr, a missing import).
* **valid** — `false` if the kernel reaches its measured performance via an
  algebraic shortcut — i.e. one or more operators in the reference forward have
  been collapsed at init time, folded into a downstream reduction, or replaced
  with a substitute that does materially less arithmetic than the reference (the
  "_base" skill section describes this contract). `true` in all other cases —
  including kernels that are slow, compile-failed, or numerically wrong;
  validity is about *whether the kernel is doing the reference work*, not whether
  it is doing it well. `true` is the default — only emit `false` when you are
  confident the contract was violated.

Respond with a SINGLE JSON object and nothing else — no prose outside it, no
markdown fences — using exactly these keys: "small_guidance" (string),
"large_guidance" (string), "direction" ("large" or "small"), "valid" (boolean).

"""

ADVERSARIAL_GOAL = """### Goal (adversarial review mode)

The measured speedup (>=5x) is in a range commonly produced by algebraic
shortcuts. **You are in adversarial review mode.** A kernel reaches this range
honestly only by genuinely faster scheduling of the *same* arithmetic; far more
often it reaches it by collapsing a heavy operator's work axis before the
multiply-accumulate runs (the contract violation described in the "_base" skill
section). Treat the kernel as guilty until the code proves otherwise.

The validity contract: every heavy op (Conv*, ConvTranspose*, Linear, matmul)
must materialize its full reference output shape at runtime, and the kernel must
perform the same asymptotic multiply-add count as the reference. Pre-reducing
weight or input along an axis that a downstream sum / mean / avg-pool later
collapses -- then running a smaller GEMM / matvec -- violates the contract even
though it is mathematically equivalent.

Provide:

* **reasoning** — walk the kernel's main accumulation loop. State (a) which
  heavy-op output axis is materialized at full shape, and (b) which arithmetic
  loop performs the reference's full MAC count. If you cannot point to both in
  the actual code, say so explicitly -- that is grounds for `valid: false`.
* **small_guidance** — 1-3 concrete tuning bullets (block sizes, num_warps,
  num_stages, layout, autotune configs, fusion opportunities within the existing
  kernel structure).
* **large_guidance** — 1-3 concrete design bullets (what a from-scratch rewrite
  should change about the kernel's strategy). If the kernel is a shortcut, the
  large guidance must direct the proposer back to materializing the full
  heavy-op output.
* **direction** — `"large"` or `"small"`.
* **valid** — `true` ONLY when your reasoning named both (a) the full-shape
  heavy-op output axis and (b) the full-MAC arithmetic loop. A kernel that fuses
  the heavy op and the downstream reduction into one pass is still valid -- the
  contract is violated only when the heavy op's work axis is collapsed *before*
  the heavy op runs. In every other case -- including when you are merely unsure
  -- emit `false`.

Respond with a SINGLE JSON object and nothing else — no prose outside it, no
markdown fences — using exactly these keys: "reasoning" (string),
"small_guidance" (string), "large_guidance" (string), "direction" ("large" or
"small"), "valid" (boolean).

"""

SLOW_GOAL = """### Goal (redesign mode)

The kernel is **correct but slow** (measured fast_p is between 0 and 0.8x — it
is slower than the PyTorch baseline). It retained the reference heavy operator
(Conv*, ConvTranspose*, or Linear) and only tuned around it. That retained
PyTorch operator *is* the bottleneck: tuning, fusion, and config changes cannot
close the gap while the heavy op runs as an opaque PyTorch call. The only way
forward is a from-scratch redesign that **replaces** (not wraps) the heavy
operator with a custom Triton kernel.

Provide a single **large_guidance** block: 1-3 concrete design bullets
prescribing a Triton strategy that *replaces* the heavy op. Use the per-op
strategy that matches the reference:

* **ConvTranspose*d** — scatter-add (or gather) formulation with explicit
  stride / padding / output-padding index arithmetic; do not call
  `nn.ConvTranspose*d`.
* **Conv*d** — tiled im2col + GEMM, or direct cross-correlation with an explicit
  reduction loop over the (C_in, K) window.
* **Linear / matmul** — split-K GEMM with `tl.dot`, tiled over M/N/K.

Do not produce tuning guidance, a direction decision, or a validity check — a
slow correct kernel is neither a shortcut nor a tuning candidate.

Respond with a SINGLE JSON object and nothing else — no prose outside it, no
markdown fences — using exactly this key: "large_guidance" (string).

"""

STRUCTURAL_ALERT = """## Structural alert

The reference contains a {heavy_op} -> {reduction} chain.
Expected intermediate output of {heavy_op}: {shape_descriptor}
(then reduced to the kernel's reported output by {reduction}).

Common shortcut surface: pre-reducing {heavy_op}'s weight or input
along the axis that {reduction} later collapses, then running a smaller
GEMM/matvec. Legitimate fusion of {heavy_op} and {reduction} inside one
Triton kernel is allowed; the contract is violated only when the
heavy-op work axis is collapsed before the multiply-accumulate runs.

Verify the kernel's main accumulation loop iterates over the full
{heavy_op} output grid.

"""

HEAVY_OP_NOT_REPLACED_ALERT = """## Heavy op not replaced (runtime-verified)

Runtime verification of the executed forward path found that the reference heavy
operator (Conv*, ConvTranspose*, or Linear) **still ran in PyTorch** during the
measured forward, and `ModelNew` defines **no** custom Triton/CUDA replacement for
it. The kernel only fuses the surrounding work (BatchNorm, reductions, pointwise
ops) and leaves the heavy op on the live PyTorch path, so the heavy op — the part
with the most headroom — was never actually replaced.

This kernel has been scored as **invalid** (the heavy op must be replaced, not
merely wrapped). Your guidance must require a custom Triton (or CUDA C++) kernel
that implements the heavy op itself and executes **unconditionally** on the
forward path. Fusing only the cheap surrounding ops does not count as replacing
the heavy op.

"""

NSIGHT_PROFILE = """## Nsight hardware profile (measured)

The kernel(s) this candidate actually launched were profiled with NVIDIA Nsight.
This is real per-operator hardware data for the executed forward path -- use it to
ground your diagnosis instead of reasoning from the single latency number alone.

```
{nsight_summary}
```

How to read it:
* **kernels** (from nsys): the GPU kernels this candidate launched -- name,
  instances, total/avg ms, and % of GPU time. The top entry dominates runtime.
* **memory_ops** (from nsys): host/device memcpy time -- large values signal a
  transfer bottleneck rather than a compute one.
* **ncu_kernels** (targeted hardware counters for the dominant kernel(s)):
  - **throughput / roofline:** `compute_throughput_pct` / `memory_throughput_pct`
    / `dram_throughput_pct` / `l2_throughput_pct` (% of peak), `duration`, and a
    `roofline_bound` classification (memory_bound / compute_bound / latency_bound).
  - **occupancy + what caps it:** `achieved_occupancy_pct` is the realized warp
    occupancy; `occ_limit_registers` / `occ_limit_shared_mem` / `occ_limit_warps`
    / `occ_limit_blocks` are the per-cause occupancy ceilings -- the *smallest*
    one is the binding constraint (e.g. a low `occ_limit_registers` means
    register pressure is capping occupancy, so cut `registers_per_thread`).
  - **launch config:** `grid_size`, `block_size`, `waves_per_sm`,
    `shared_mem_per_block`.
  - **cache + traffic:** `l1_hit_rate_pct` / `l2_hit_rate_pct` (low hit rates +
    high `dram_bytes_read` / `dram_bytes_write` point at a memory-traffic problem
    -- improve locality/coalescing or fuse to cut DRAM round-trips).

"""

def generate_evaluator_prompt(custom_triton_kernels: str=None, run_info=None, experience_guidance_path: str=None, task_params: dict=None, knowledge_1_threshold: int=3, heavy_op_not_replaced: bool=False, redesign_exhausted: bool=False):
    # Extract required parameters from task prompt template
    required_keys = _extract_format_keys(TASK_INSTRUCTION)

    # Build format dict: use task_params if provided, otherwise fall back to original parameters
    format_dict = {}
    if task_params is not None:
        format_dict.update(task_params)

    # Fall back to original parameters for missing keys
    for key in required_keys:
        if key not in format_dict:
            if key == 'custom_triton_kernels':
                format_dict[key] = custom_triton_kernels
            elif key == 'run_info':
                format_dict[key] = run_info
            else:
                raise ValueError(f"Missing required parameter: {key}")

    # Strip self-justifying comments / docstrings from the kernel before it is
    # shown to the evaluator -- an adversarial reviewer must reason from the
    # code, not from the kernel's own correctness claims.
    if isinstance(format_dict.get('custom_triton_kernels'), str):
        format_dict['custom_triton_kernels'] = strip_comments_and_docstrings(
            format_dict['custom_triton_kernels'])

    # Scale scrutiny with the measured speedup. fast_p is only populated on the
    # successful perf branch (src/eval.py), so compile/correctness failures have
    # empty runtime_stats and naturally bypass adversarial mode.
    ratio = 0
    correctness = False
    if isinstance(run_info, KernelExecResult):
        ratio = run_info.runtime_stats.get("fast_p", 0) or 0
        correctness = run_info.correctness
    # mode is the single source of truth for: skill step_type, STRUCTURAL_ALERT
    # gating, the goal-block append, and the Mode-A direction force in run_evaluator.
    # A kernel whose heavy op was never actually replaced on the live path (verified
    # at runtime: the reference heavy op still dispatched through PyTorch) is
    # structurally Mode A, so force "slow" regardless of measured speed -- this makes
    # the large-step (redesign) path deterministic rather than relying on the
    # unreplaced kernel coincidentally measuring slow. Everything downstream
    # (step_type="large", SLOW_GOAL, STRUCTURAL_ALERT skip, and the direction="large"
    # force in run_evaluator, which keys on mode=="slow") then applies automatically.
    #
    # Escape hatch: a kernel that genuinely replaced the heavy op (redesign_exhausted)
    # but is still slow has exhausted the redesign lever -- release Mode A to "default"
    # (GOAL) so small tuning steps become reachable. heavy_op_not_replaced still forces
    # "slow" and takes priority (the two are mutually exclusive by construction in
    # run_evaluator: genuine replacement requires the heavy op NOT to have run in
    # PyTorch, which is exactly what heavy_op_not_replaced asserts did happen).
    heavy_op_unreplaced = heavy_op_not_replaced
    if heavy_op_unreplaced:
        mode = "slow"          # Mode A (forced -- heavy op verified unreplaced)
    elif correctness and 0 < ratio < 0.8:
        # Correct but slow: redesign (Mode A), unless the heavy op was already
        # genuinely replaced -- then tune instead (Mode C).
        mode = "default" if redesign_exhausted else "slow"
    elif correctness and ratio >= 5.0:
        mode = "adversarial"   # Mode B
    else:
        mode = "default"       # Mode C (default + all compile/correctness failures)

    prompt = PROBLEM_STATEMENT
    # Skill step_type varies by mode: Mode A is redesign-only (Design content
    # only), Modes B/C see both Design and Tuning per detected family. _base
    # (the no-shortcut contract) is emitted for any step_type.
    prompt += generate_skill_prompt(
        task_params.get("arc_src"),
        step_type=("large" if mode == "slow" else "both"),
        task_params=task_params,
    )
    prompt += generate_experience_guidance_prompt(experience_guidance_path, threshold=knowledge_1_threshold)

    # Inject a Structural Alert (after the skills, before the task) when the
    # reference contains a heavy-op -> linear-reduction chain. Skipped on the
    # failure path and in Mode A: a slow correct kernel cannot be an algebraic
    # shortcut, so the anti-shortcut alert is irrelevant.
    if correctness and mode != "slow":
        for chain in detect_shortcut_risk(task_params.get("arc_src")):
            prompt += STRUCTURAL_ALERT.format(
                heavy_op=chain.heavy_op,
                reduction=chain.reduction,
                shape_descriptor=chain.shape_descriptor,
            )

    # Inject the heavy-op-not-replaced alert. heavy_op_not_replaced is forced into
    # Mode A ("slow") above, where STRUCTURAL_ALERT is skipped, so the corrective
    # feedback rides on this dedicated alert instead.
    if heavy_op_not_replaced:
        prompt += HEAVY_OP_NOT_REPLACED_ALERT

    prompt += TASK_INSTRUCTION.format(**format_dict)

    # Inject the Nsight hardware profile (after the metrics block) when present
    # and non-error. Mirrors the compilation_error_parsed pipeline: parsed data
    # lives in metadata["nsight"] and is rendered into a dedicated section.
    if isinstance(run_info, KernelExecResult):
        nsight = run_info.metadata.get("nsight")
        # Render only when there is actual profiling payload (kernels or ncu
        # counters); a pure-error dict (error / nsys_error only) is skipped.
        if isinstance(nsight, dict) and (
            nsight.get("kernels") or nsight.get("ncu_kernels")
        ):
            from src.nsight_profiler import format_nsight_summary
            prompt += NSIGHT_PROFILE.format(
                nsight_summary=format_nsight_summary(nsight)
            )

    # On compile or runtime failure, surface the traceback in a dedicated
    # section before the goal so the evaluator can diagnose the structural
    # cause. Fire on any failed run (compile failure, or compiled-but-incorrect
    # with a captured error such as the gcc-launcher build error), preferring
    # the structured parser output when present.
    if isinstance(run_info, KernelExecResult) and not run_info.correctness:
        parsed = run_info.metadata.get("compilation_error_parsed")
        if parsed:
            compile_error = json.dumps(parsed, indent=2, ensure_ascii=False)
        else:
            compile_error = (
                run_info.metadata.get("compilation_error")
                or run_info.metadata.get("runtime_error")
            )
        if compile_error:
            prompt += COMPILE_FAILURE.format(compile_error=compile_error)

    prompt += {"slow": SLOW_GOAL, "adversarial": ADVERSARIAL_GOAL}.get(mode, GOAL)
    return prompt, mode
