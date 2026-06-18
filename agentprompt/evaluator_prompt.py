import re
import json
from agentprompt.prompt_modules import generate_experience_guidance_prompt
from agentprompt.prompt_modules import generate_hardware_information_prompt
from agentprompt.Utils import generate_skill_prompt
from agentprompt.Utils.detect import detect_shortcut_risk, strip_comments_and_docstrings
from src.eval import KernelExecResult

def _extract_format_keys(template: str):
    """Extract format keys from a template string."""
    return set(re.findall(r'\{(\w+)\}', template))

PROBLEM_STATEMENT = """## Problem Statement

You evaluate the most recent custom Triton kernel and its measured performance, then emit guidance for the next iteration. Follow the Optimization Skills below — guidance should focus on kernel-level tuning, fusion, and (when the torch convolution dominates runtime) replacing it with a custom Triton kernel; never propose graph-level algebraic shortcuts that eliminate a heavy operator.

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

You evaluate the most recent custom Triton kernel and its measured
performance, then emit guidance for the next iteration. Produce exactly
four tagged blocks, in this order:

<small_guidance>
1-3 concrete tuning bullets — block sizes, num_warps, num_stages, layout,
autotune configs, fusion opportunities within the existing kernel
structure. These will be consumed by the tuner if a refinement step is
chosen.
</small_guidance>

<large_guidance>
1-3 concrete design bullets — what a from-scratch rewrite should change
about the kernel's strategy (im2col vs. direct, scatter-add vs. gather,
fusion boundary, persistence). These will be consumed by the proposer if a
redesign step is chosen.
</large_guidance>

<direction>large</direction>  if you believe the current kernel is
structurally unable to reach the target and a from-scratch rewrite is more
promising than continued tuning.

<direction>small</direction>  if you believe continued tuning of the
current kernel will close the gap.

<valid>false</valid>  if the kernel reaches its measured performance via an
algebraic shortcut — i.e. one or more operators in the reference forward have
been collapsed at init time, folded into a downstream reduction, or replaced
with a substitute that does materially less arithmetic than the reference.
The "_base" skill section describes this contract; emit `false` whenever it
is violated, even if compile + correctness checks pass and the speedup looks
real.

<valid>true</valid>  in all other cases — including kernels that are slow,
compile-failed, or numerically wrong. Validity is about *whether the kernel
is doing the reference work*, not whether it is doing it well.

Emit exactly one <direction> tag and exactly one <valid> tag. Both guidance
blocks are always required even if one is short. <valid>true</valid> is the
default — if you omit or malform the tag it will be treated as valid, so only
emit <valid>false</valid> when you are confident the contract was violated.
On compile failure, default to <direction>large</direction> unless the
traceback points at a trivially-fixable issue (e.g. a typo, a wrong
constexpr, a missing import).

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

Produce exactly four tagged blocks, in this order:

<reasoning>
Walk the kernel's main accumulation loop. State (a) which heavy-op output axis
is materialized at full shape, and (b) which arithmetic loop performs the
reference's full MAC count. If you cannot point to both in the actual code, say
so explicitly -- that is grounds for <valid>false</valid>.
</reasoning>

<small_guidance>
1-3 concrete tuning bullets -- block sizes, num_warps, num_stages, layout,
autotune configs, fusion opportunities within the existing kernel structure.
</small_guidance>

<large_guidance>
1-3 concrete design bullets -- what a from-scratch rewrite should change about
the kernel's strategy. If the kernel is a shortcut, the large guidance must
direct the proposer back to materializing the full heavy-op output.
</large_guidance>

<direction>large</direction> or <direction>small</direction> -- exactly one.

<valid>true</valid> is emitted ONLY when your <reasoning> named both (a) the
full-shape heavy-op output axis and (b) the full-MAC arithmetic loop. A kernel
that fuses the heavy op and the downstream reduction into one pass is still
valid -- the contract is violated only when the heavy op's work axis is
collapsed *before* the heavy op runs. In every other case -- including when you
are merely unsure -- emit <valid>false</valid>.

Emit exactly one <direction> tag and exactly one <valid> tag. The <reasoning>
block is required before <valid>.

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

def generate_evaluator_prompt(custom_triton_kernels: str=None, run_info=None, experience_guidance_path: str=None, task_params: dict=None, knowledge_1_threshold: int=3):
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
    adversarial = ratio >= 5.0

    prompt = PROBLEM_STATEMENT
    # Skill prompt loaded with step_type="both" so the evaluator sees both
    # Design and Tuning content per detected family.
    prompt += generate_skill_prompt(task_params.get("arc_src"), step_type="both", task_params=task_params)
    prompt += generate_experience_guidance_prompt(experience_guidance_path, threshold=knowledge_1_threshold)
    prompt += generate_hardware_information_prompt(task_params.get('gpu_name'), task_params.get('gpu_architecture'))

    # Inject a Structural Alert (after hardware info, before the task) when the
    # reference contains a heavy-op -> linear-reduction chain. Skipped on the
    # failure path to keep that prompt focused.
    if correctness:
        for chain in detect_shortcut_risk(task_params.get("arc_src")):
            prompt += STRUCTURAL_ALERT.format(
                heavy_op=chain.heavy_op,
                reduction=chain.reduction,
                shape_descriptor=chain.shape_descriptor,
            )

    prompt += TASK_INSTRUCTION.format(**format_dict)

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

    prompt += ADVERSARIAL_GOAL if adversarial else GOAL
    return prompt
