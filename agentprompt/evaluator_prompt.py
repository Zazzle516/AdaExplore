import re
from agentprompt.prompt_modules import generate_experience_guidance_prompt
from agentprompt.prompt_modules import generate_hardware_information_prompt
from agentprompt.skills import generate_skill_prompt
from src.utils import read_file
from src.eval import KernelExecResult
import os

def _extract_format_keys(template: str):
    """Extract format keys from a template string."""
    return set(re.findall(r'\{(\w+)\}', template))

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

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

    prompt = PROBLEM_STATEMENT
    # Skill prompt loaded with step_type="both" so the evaluator sees both
    # Design and Tuning content per detected family.
    prompt += generate_skill_prompt(task_params.get("arc_src"), step_type="both")
    prompt += generate_experience_guidance_prompt(experience_guidance_path, threshold=knowledge_1_threshold)
    prompt += generate_hardware_information_prompt(task_params.get('gpu_name'), task_params.get('gpu_architecture'))
    prompt += TASK_INSTRUCTION.format(**format_dict)

    # On compile failure, surface the traceback in a dedicated section before
    # the goal so the evaluator can diagnose the structural cause.
    if isinstance(run_info, KernelExecResult) and not run_info.compiled:
        compile_error = (
            run_info.metadata.get("compilation_error")
            or run_info.metadata.get("runtime_error")
            or "No traceback captured."
        )
        prompt += COMPILE_FAILURE.format(compile_error=compile_error)

    prompt += GOAL
    return prompt

if __name__ == "__main__":
    EXAMPLE_ARCH_SRC = read_file(os.path.join(REPO_TOP_PATH, "datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py"))
    # The same for display purpose
    EXAMPLE_NEW_ARCH_SRC = read_file(os.path.join(REPO_TOP_PATH, "datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py"))
    prompt = generate_evaluator_prompt(
        task_params={
            "arc_src": EXAMPLE_ARCH_SRC,
            "gpu_name": "NVIDIA A100",
            "gpu_architecture": "Ampere",
            "dtype_str": "float16",
        },
        custom_triton_kernels=EXAMPLE_NEW_ARCH_SRC,
        run_info=KernelExecResult(compiled=True, correctness=True, runtime=1.0, runtime_stats={"fast_p": 1.0}),
        experience_guidance_path=None,
    )
    print(prompt)
