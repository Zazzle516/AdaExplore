import argparse
import re
from typing import Optional
from agent.inference_server import query_inference_server
from agentprompt.proposer_prompt import (
    generate_proposer_prompt,
    generate_pool_prompt_dual,
)
from src.utils import extract_first_code
from src.eval import eval_kernel_against_ref, wrapped_eval_kernel_against_ref, KernelExecResult
from agentprompt.evaluator_prompt import generate_evaluator_prompt
from agentprompt.tuner_prompt import generate_tuner_prompt
from agent.utils import extract_edits, str_replace

def _use_performance_metric(args: argparse.Namespace) -> bool:
    test_source = str(getattr(args, "test_source", "KB")).upper()
    return test_source not in {"TBG"}

def _extract_tag(output: str, tag: str) -> str:
    """Return the last occurrence of <tag>...</tag> (DOTALL), stripped.

    Last-match: if the LLM drafts a tag then re-emits a corrected version, the
    final occurrence is the binding one.
    """
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", output, re.DOTALL)
    return matches[-1].strip() if matches else ""

def _extract_direction(output: str) -> Optional[str]:
    """Return the last <direction>large|small</direction>, or None if missing."""
    matches = re.findall(r"<direction>\s*(large|small)\s*</direction>", output)
    return matches[-1] if matches else None

def extract_proposal_kernel(output: str) -> str:
    """Pull the complete kernel file from a proposer response.

    1. Preferred: the last <kernel>...</kernel> block (per the output contract).
    2. Fallback: the last fenced block that defines class ModelNew.
    3. Last resort: the first fenced block (legacy extract_first_code) — saved
       as-is so it compile-fails visibly rather than silently picking a fragment.
    """
    tagged = _extract_tag(output, "kernel")
    if tagged:
        # tolerate a code fence nested inside the tags
        tagged = re.sub(r"^```(?:python|cpp)?\s*|\s*```$", "", tagged).strip()
        if "class ModelNew" in tagged:
            return tagged
    blocks = re.findall(r"```(?:python|cpp)?\s*(.*?)```", output, re.DOTALL)
    for block in reversed(blocks):
        if "class ModelNew" in block:
            return block.strip()
    return extract_first_code(output, ["python", "cpp"])

def _extract_validity(output: str) -> Optional[bool]:
    """Return the last <valid>true|false</valid> as a bool, or None if missing/malformed.

    Missing/malformed → None; downstream treats None as valid (default-true).
    """
    matches = re.findall(r"<valid>\s*(true|false)\s*</valid>", output, re.IGNORECASE)
    return matches[-1].lower() == "true" if matches else None

def run_evaluator(ref_arch_src: str, kernel: str, metrics: KernelExecResult, inference_server: str, args: argparse.Namespace):
    """Run the evaluator on a freshly produced kernel.

    Returns (small_guidance, large_guidance, direction, valid). Either guidance
    may be an empty string on parse failure; direction is None if the tag is
    missing or malformed; valid is None when the <valid> tag is absent or
    malformed (treated as valid downstream). Runs even when the kernel failed to
    compile — the prompt assembler injects the traceback into a dedicated
    section.
    """
    evaluator_prompt = generate_evaluator_prompt(
        task_params=args.task_params,
        custom_triton_kernels=kernel,
        run_info=metrics,
        experience_guidance_path=args.general_memory_path,
        knowledge_1_threshold=args.knowledge_1_threshold,
    )
    evaluator_output = query_inference_server(
        server=inference_server,
        model_name=args.model_name,
        prompt=evaluator_prompt,
        max_completion_tokens=args.max_completion_tokens,
    )
    small_guidance = _extract_tag(evaluator_output, "small_guidance")
    large_guidance = _extract_tag(evaluator_output, "large_guidance")
    direction = _extract_direction(evaluator_output)
    valid = _extract_validity(evaluator_output)
    return small_guidance, large_guidance, direction, valid

def single_small_step(ref_arch_src: str, inference_server: str, previous_kernels: list, previous_metrics: list, args: argparse.Namespace, tuning_guidance: str = ""):
    # Pure executor: the evaluator (run by the orchestrator) supplies
    # tuning_guidance; this function just applies the tuner edits.
    tuner_prompt = generate_tuner_prompt(
        task_params=args.task_params,
        previous_kernels=previous_kernels,
        previous_metrics=previous_metrics,
        tuning_guidance=tuning_guidance,
        experience_guidance_path=args.general_memory_path,
        knowledge_1_threshold=args.knowledge_1_threshold,
        filter_wrong_attempts=getattr(args, 'filter_wrong_attempts', False),
    )

    tuner_output = query_inference_server(
        server=inference_server,
        model_name=args.model_name,
        prompt=tuner_prompt,
        max_completion_tokens=args.max_completion_tokens,
    )
    tuned_kernel = previous_kernels[-1]
    edits = extract_edits(tuner_output)
    for old_str, new_str in edits:
        tuned_kernel = str_replace(tuned_kernel, old_str, new_str)

    measure_performance = _use_performance_metric(args)
    tuned_metrics = wrapped_eval_kernel_against_ref(
        ref_arch_src,
        tuned_kernel,
        measure_performance=measure_performance,
        num_correct_trials=5,
        num_perf_trials=100 if measure_performance else 1,
        backend="triton",
        dtype_str=args.dtype_str,
        # Use the assigned GPU id (single int), not the list of available GPUs
        device=args.gpu_id,
        use_remote_eval=getattr(args, 'use_remote_eval', False),
        remote_eval_url=getattr(args, 'remote_eval_url', "http://127.0.0.1:12017"),
        test_source=getattr(args, 'test_source', 'KB'),
        level=getattr(args, 'level', None),
        problem_id=getattr(args, 'problem_id', None),
        gpu_name=getattr(args, 'gpu_name', None),
    )

    logs = {
        "tuning_guidance": tuning_guidance,
        "tuner_prompt": tuner_prompt,
        "tuner_output": tuner_output,
        "tuned_kernel": tuned_kernel,
        "tuned_metrics": tuned_metrics,
        "prompt": tuner_prompt,
    }
    return tuned_kernel, tuned_metrics, logs

def single_large_step(
    ref_arch_src: str,
    inference_server: str,
    kernel_pool: list,
    metrics_pool: list,
    args: argparse.Namespace,
    *,
    large_guidance: Optional[str] = None,
    context_ids: list[int] | None = None,
    elite_kernel_pool: list | None = None,
    elite_metrics_pool: list | None = None,
    elite_context_ids: list[int] | None = None,
):
    # previously experienced kernels and metrics
    pool_prompt = generate_pool_prompt_dual(
        kernel_pool=kernel_pool,
        metrics_pool=metrics_pool,
        kernel_pool_ids=context_ids,
        elite_kernel_pool=elite_kernel_pool,
        elite_metrics_pool=elite_metrics_pool,
        elite_pool_ids=elite_context_ids,
    )
    # Proposer Agent
    proposer_prompt = generate_proposer_prompt(
        task=args.test_source,
        task_params=args.task_params,
        experience_guidance_path=args.general_memory_path,
        pool_prompt=pool_prompt,
        knowledge_1_threshold=args.knowledge_1_threshold,
        large_guidance=large_guidance,
    )
    proposer_output = query_inference_server(
        server=inference_server,
        model_name=args.model_name,
        prompt=proposer_prompt,
        max_completion_tokens=args.max_completion_tokens,
    )
    proposal_kernel = extract_proposal_kernel(proposer_output)
    measure_performance = _use_performance_metric(args)
    proposal_metrics = wrapped_eval_kernel_against_ref(
        ref_arch_src,
        proposal_kernel,
        measure_performance=measure_performance,
        num_correct_trials=5,
        num_perf_trials=100 if measure_performance else 1,
        backend="triton",
        dtype_str=args.dtype_str,
        # Use the assigned GPU id (single int), not the list of available GPUs
        device=args.gpu_id,
        use_remote_eval=getattr(args, 'use_remote_eval', False),
        remote_eval_url=getattr(args, 'remote_eval_url', "http://127.0.0.1:12017"),
        test_source=getattr(args, 'test_source', 'KB'),
        level=getattr(args, 'level', None),
        problem_id=getattr(args, 'problem_id', None),
        gpu_name=getattr(args, 'gpu_name', None),
    )
    # For large step, prompt is just the proposer_prompt
    prompt = proposer_prompt
    
    logs = {
        "proposer_prompt": proposer_prompt,
        "proposal_kernel": proposal_kernel,
        "proposal_metrics": proposal_metrics,
        "prompt": prompt,
    }
    return proposal_kernel, proposal_metrics, logs

def dummy_small_step(ref_arch_src: str, inference_server: str, previous_kernels: list, previous_metrics: list, args: argparse.Namespace):
    """
    Dummy small step that does nothing, returns empty kernel and zero metrics.
    """
    from src.eval import KernelExecResult
    
    empty_kernel = ""
    zero_metrics = KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={},
        runtime=-1.0,
        runtime_stats={}
    )
    logs = {
        "dummy": True,
        "action": "dummy_small_step",
        "prompt": "",
    }
    return empty_kernel, zero_metrics, logs

def dummy_large_step(ref_arch_src: str, inference_server: str, kernel_pool: list, metrics_pool: list, args: argparse.Namespace):
    """
    Dummy large step that does nothing, returns empty kernel and zero metrics.
    """
    from src.eval import KernelExecResult
    
    empty_kernel = ""
    zero_metrics = KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={},
        runtime=-1.0,
        runtime_stats={}
    )
    logs = {
        "dummy": True,
        "action": "dummy_large_step",
        "prompt": "",
    }
    return empty_kernel, zero_metrics, logs
