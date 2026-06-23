"""Standalone driver that NVIDIA Nsight (nsys/ncu) wraps as a whole process.

The Nsight profilers attach to an *entire process*, so to profile a single
candidate kernel we need a minimal Python entry point that:

1. loads the reference architecture + the candidate ``ModelNew`` exactly the
   way ``src/eval.py`` does (same loaders, same seed, same dtype), and
2. runs warmup followed by ``--num_iters`` forward passes, each bracketed by
   ``torch.cuda.synchronize()``.

It performs **no timing of its own** -- the profiler records the GPU work. The
only thing printed to stdout is a small JSON status line so the orchestrator can
tell whether the forward pass actually ran (profilers do not propagate Python
exit codes cleanly in every case).

Usage (mirrors ``eval_kernel_against_ref`` arguments):

    python tool_scripts/nsight_runner.py \
        --kernel_path /tmp/cand.py \
        --test_source KB --level 2 --problem_id 1 \
        --device 0 --dtype fp32 --backend triton --num_iters 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make the repo root importable when invoked as a bare script under a profiler.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from agent.utils import load_test_source  # noqa: E402
from src.eval import (  # noqa: E402
    load_custom_model,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
    set_seed,
)


def _to_dtype(tensors, dtype_str: str, device):
    """Move tensors to ``device`` and cast to the requested dtype (mirrors eval)."""
    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Invalid data type: {dtype_str}")
    out = []
    for x in tensors:
        if isinstance(x, torch.Tensor):
            out.append(x.cuda(device=device).to(dtype=dtype))
        else:
            out.append(x)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Nsight forward-pass driver")
    parser.add_argument("--kernel_path", type=str, required=True,
                        help="Path to the candidate kernel .py file (defines ModelNew)")
    parser.add_argument("--test_source", type=str, default="KB",
                        choices=["KB", "SYN", "FIT", "MLSYS", "TBG"])
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--problem_id", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--backend", type=str, default="triton",
                        choices=["triton", "cuda"])
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available, cannot run Nsight runner"
    torch.cuda.set_device(args.device)
    device = args.device
    is_triton = args.backend == "triton"

    # Load reference arch (for Model / get_init_inputs / get_inputs) the same way eval does.
    _name, ref_arch_src = load_test_source(
        args.test_source, args.level, args.problem_id, suffix=args.suffix
    )
    context: dict = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )

    set_seed(args.seed)
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    # Load the candidate ModelNew identically to eval (tempfile import for Triton).
    with open(args.kernel_path, "r") as f:
        custom_model_src = f.read()

    tempfile_handle = None
    if is_triton:
        ModelNew, tempfile_handle = load_custom_model_with_tempfile(
            custom_model_src, entry_point="ModelNew"
        )
    else:
        ModelNew = load_custom_model(custom_model_src, context)

    with torch.no_grad():
        set_seed(args.seed)
        model_new = ModelNew(*init_inputs).cuda(device=device)
        torch.cuda.synchronize(device=device)

        set_seed(args.seed)
        inputs = _to_dtype(get_inputs(), args.dtype, device)
        torch.cuda.synchronize(device=device)

        # Warmup -- excluded from the region of interest as much as possible.
        for _ in range(args.num_warmup):
            model_new(*inputs)
        torch.cuda.synchronize(device=device)

        # Measured region: the profiler records the GPU kernels launched here.
        for _ in range(args.num_iters):
            model_new(*inputs)
            torch.cuda.synchronize(device=device)

    # Best-effort cleanup of the Triton tempfile.
    if tempfile_handle is not None:
        try:
            os.unlink(tempfile_handle.name)
        except OSError:
            pass

    print(json.dumps({"status": "ok", "iters": args.num_iters}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
