# AdaExplore

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.16625-b31b1b.svg)](https://arxiv.org/abs/2604.16625)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://stiglidu.github.io/AdaExplore/)

> **Weihua Du, Jingming Zhuo, Yixin Dong, Andre He, Weiwei Sun, Zeyu Zheng, Manupa Karunaratne, Ivan Fox, Tim Dettmers, Tianqi Chen, Yiming Yang, Sean Welleck**  
> ["AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation " (2026)](https://arxiv.org/abs/2604.16625)

AdaExplore is a research codebase for **LLM-driven GPU kernel engineering**. The framework is built around two complementary stages:

- **Adapt**: synthesize training tasks, collect execution failures, and distill recurring error patterns into a reusable cross-task skill memory.
- **Explore**: optimize each target kernel with a diversity-preserving tree search that alternates between local refinement and larger structural regeneration.

This repository also includes the evaluation harness, synthetic task generation pipeline, an optional remote evaluation service for multi-GPU or cluster setups, and a TensorRT FP32 baseline so reported speedups are measured against a strong inference engine rather than torch eager.

<p align="center">
  <img src="assets/overview.png" alt="AdaExplore overview" />
</p>

## Overview

AdaExplore addresses kernel generation with a two-stage workflow. The **Adapt** stage improves correctness by turning repeated compile/runtime failures on synthesized tasks into reusable constraint rules, which are then injected back into future prompts as skill memory. The **Explore** stage improves optimization quality by organizing candidate kernels as a search tree and balancing short-horizon local edits with broader structural reconstruction.

In practice, this repository is a simplified open-source release of AdaExplore, where:
- `synthesis/` generates the playground used for adaptation,
- `skill_memory/` builds and updates the reusable memory,
- `agent/` runs the search procedure used for benchmark or task-time optimization,
- `agentprompt/` composes the proposer / evaluator prompts (operator-aware skills, anti-shortcut contract, three review modes),
- `src/` hosts the evaluation harness, the Triton error parser, the heavy-op runtime checker, and the Nsight profiling pipeline,
- `TRT_Baseline/` builds and times the TensorRT FP32 reference baseline.

## Installation

### 1. Create a Python environment and install dependencies

```bash
conda env create -f environment.yml
conda activate adaexplore
```

This creates a Python 3.12 environment named `adaexplore` and installs the pinned dependencies from `requirements.txt`. Verified on Linux x86_64 with Ampere-class GPUs and a CUDA 12.4 user-space driver; the bundled `torch==2.5.0` wheel ships its own CUDA 12.4 runtime, so only the NVIDIA host driver needs to support CUDA 12.4.

### 2. Configure model access

Set the API credentials for the backend you want to use:

```bash
# OpenAI
export OPENAI_API_KEY=...

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT=...
export AZURE_API_KEY=...

# Anthropic
export ANTHROPIC_API_KEY=...
```

### 3. Start the evaluation service

Most provided configs use the remote evaluation service on port `12017`.

```bash
bash online_judge/start_server.sh
```

You can also launch it directly:

```bash
python -m uvicorn online_judge.app_with_queue:app --host 0.0.0.0 --port 12017
```

## Adapt: Failure-Driven Skill Acquisition

The adaptation stage corresponds to the first half of the paper: AdaExplore synthesizes diverse kernel-style tasks, runs the agent on them, and summarizes repeated failures into a cross-task memory of rules such as invalid Triton usage patterns. In this repo, that workflow is implemented by `synthesis/` for task generation and `skill_memory/` for extracting or updating the memory file.

We also provide a pre-generated skill memory at `results/memory/general_memory_v1_200.txt`, so you can use it directly as the starting point for exploration runs. To run this stage, you can directly use the pre-generated dataset in `datasets/KernelBench_syn/syn_v1`.

(Optional) If you want to rebuild the synthetic dataset from scratch, generate the tasks and materialize them into the same KernelBench-style folder:

```bash
python synthesis/generate_data.py \
  --server_type azure \
  --model_name gpt-5-mini \
  --prompt_style composite \
  --input_levels 1 \
  --num_generations 200 \
  --num_examples_per_request 3 \
  --temperature 1.0

python synthesis/rename.py \
  --source_path outputs/data_generation/<generated_data_dir_from_previous_step> \
  --data_path datasets/KernelBench_syn/<your_dataset_name> \
  --force
```

Run the agent on the synthesized set with online memory updates enabled:

```bash
python agent/agent_entry.py --config config/SYN-v1/config_SYN-v1_none_MCTS.yaml
```

This config uses `memory_update: true`, so failed generations are distilled into `outputs/SYN-v1_example_run/general_memory.txt` as the run progresses. If you want to refresh the bundled memory from historical logs under `outputs/...`, you can also build it offline with:

```bash
python skill_memory/skill_memory.py \
  --log-dir outputs/example_run \
  --knowledge-store-path outputs/example_run/general_memory.txt \
  --server openai \
  --model-name gpt-5-mini \
  --max-logs 3000 \
  --seed 42
```

## Explore: Diversity-Preserving Kernel Search

The exploration stage corresponds to the second half of the paper: AdaExplore uses the skill memory collected during adaptation to guide search, while a tree-based optimizer preserves multiple candidate branches and alternates between local edits and structural reconstruction. In this repo, that behavior is implemented through the `MCTS` agent in `agent/agent_entry.py` and the benchmark configs under `config/`.

Each search step runs a **propose → evaluate → tune** loop. A *proposer* writes a fresh kernel; the candidate is run through the evaluation harness; an LLM *evaluator* then diagnoses the result and returns structured guidance (`small_guidance`, `large_guidance`, a `direction` of `large`/`small`, and a `valid` flag) that steers the next expansion toward either a local tune (`small`) or a from-scratch redesign (`large`).

The single *evaluator* (`agentprompt/evaluator_prompt.py`) replaces the former separate *reviser* agent: diagnosis, validity certification, and next-step guidance are now produced in one pass by `run_evaluator` (`agent/actions.py`), called from `agent/mcts.py` and the large/small loops. Each `MCTS` tree node now carries the evaluator's output inline (`small_guidance`, `large_guidance`, `evaluator_direction`, `evaluator_valid`) and the exact prompt sent to it, persisted per step as `step_*_evaluator_prompt.txt` for inspection. The tuner consumes this guidance directly ("Guidance from Evaluator Agent").

### Operator-aware skill prompts

Prompt content is no longer one monolithic block. The reusable optimization guidance now lives as per-operator **skills** under `agentprompt/skills/*.md` (`conv`, `linear`, `norm`, `activation`, `pooling`, `reduction`, plus `_base` and `_default`). At prompt-assembly time, `agentprompt/Utils` parses the reference architecture's source (`detect_families`, an AST walk over `nn.*` / `F.*` calls), selects only the matching skill files, and substitutes the live hardware parameters (GPU name, architecture, dtype) into the text. Each operator skill is split into a `## Design` section (used for redesign / large steps) and a `## Tuning` section (used for local tuning); `_base` carries the always-on safety contract.

As part of this move, the standalone "Hardware Information" blocks were removed from the benchmark template (`agentprompt/benchmarks/KB_prompt.py`) and the tuner prompt; hardware facts are now injected through the skill system's placeholder substitution (`agentprompt/Utils/hardware.py`) from each config's `gpu_name` / `gpu_architecture` / `dtype_str`, keeping a single source of truth.

### Three prompt-composition modes

The evaluator (`agentprompt/evaluator_prompt.py`) selects one of three review modes per candidate, which jointly controls the skill altitude, the structural alerts, and the next-step direction:

- **Mode A — redesign (`slow`)**: the kernel is correct but slow (`0 < fast_p < 0.8`), or its heavy op was never actually replaced on the live path. Only the `## Design` skill content is shown, the direction is forced to `large`, and the search is pushed toward a structural rewrite.
- **Mode B — adversarial (`fast_p ≥ 5`)**: a large measured speedup is exactly the regime that algebraic shortcuts produce, so the prompt flips its default to `<valid>false</valid>` and demands the evaluator name both the full-shape heavy-op axis and the accumulation loop performing the full MAC count before certifying the kernel.
- **Mode C — default**: ordinary tuning / all compile and correctness failures. Both `## Design` and `## Tuning` skill content is shown so small tuning steps stay reachable.

### Algebraic-shortcut & heavy-op detection

A recurring failure mode is the agent "winning" by *eliminating* the heavy operator instead of speeding it up — e.g. folding a post-conv `mean`/`sum` into the weight at init time so a tiny GEMM replaces the convolution. These pass the loose `atol=rtol=5e-2` correctness check but are not the optimization target. AdaExplore defends against this on two fronts:

- **Static detection** (`agentprompt/Utils/detect.py`, `detect_shortcut_risk`): AST analysis of the reference detects `heavy-op → linear-reduction` chains and injects a *Structural Alert* naming the expected intermediate shape and the shortcut pattern to look for.
- **Runtime detection** (`src/heavyop_checker.py`): a `TorchDispatchMode` records which `aten` heavy ops actually dispatched through PyTorch during the timed forward. Because Triton kernel launches do not go through `__torch_dispatch__`, a heavy `aten` op showing up means the reference op truly ran in PyTorch — catching kernels that park their "rewrite" in dead code (e.g. an `if self.training:` branch that never executes). Such a kernel is forced into Mode A and flagged as not replacing the heavy op.

The shared anti-shortcut contract — every reference operator must materialize its full output shape and perform the same asymptotic multiply-add count, while ordinary kernel-level fusion and standard inference-time folds remain allowed, and custom `Conv`/`ConvTranspose` kernels are explicitly authorized — lives in `agentprompt/skills/_base.md` and is injected into every prompt.

### Structured failure feedback

When a candidate fails, the harness no longer dumps a raw multi-kilobyte traceback into the evaluator prompt. `src/triton_error_parser.py` runs a regex pass over the captured Triton traceback, sorts it into named categories (`syntax`, `autotune`, `gcc`, `ptx`, `tl_constexpr`, `triton_compile`), extracts a one-line `summary` and `last_frame`, truncates per field, and persists the structured result both into the result metadata and into a sibling `step_*_compile_error.json`. The evaluator prompt renders this compact JSON instead of the verbatim dump.

### Nsight profiling

For every *correct* candidate, `src/nsight_profiler.py` profiles the kernel in stages: a cheap `nsys` trace ranks every launched kernel by GPU time, framework/library noise (`at::`, `cudnn`, `cutlass`, …) is filtered out, and a targeted `ncu` pass (a default stage) replays only the dominant candidate kernel(s) with a rich explicit `--metrics` bundle plus NVIDIA's section **rule engine** (SpeedOfLight, Occupancy, MemoryWorkloadAnalysis, SchedulerStats). The roofline-aware expert verdicts (and their estimated speedups) are surfaced directly in the evaluator prompt, so the agent optimizes the bottleneck the hardware reports rather than chasing the wrong lever. The whole pipeline is best-effort: any failure degrades to an `*_error` field and never raises into the eval path. `ncu` needs root — supply `nsight_ncu_sudo` on a non-root host.

## TensorRT FP32 Baseline

`TRT_Baseline/` builds a strong stock-inference reference so that "the kernel is faster than torch eager" is not mistaken for "the kernel is good." It exports each KernelBench `Model` to ONNX (`convertONNX/convert_all.py`), builds and times a TensorRT FP32 engine via `trtexec` (`build_and_time.py`), verifies numeric parity against PyTorch (`atol=rtol=5e-2`), and writes per-problem timings to `results/timing/RTX-4090-D/baseline_time_trt.json` alongside the existing torch baseline. All parameters (precision, workspace cap, tactic sources, opset, trials) are driven by `TRT_Baseline/config.yaml`; no engine or timing cache is used, so every run rebuilds from scratch. This pass covers KernelBench Level 1 + Level 2 (200 files).

## Reproducing Results

The two configs below correspond directly to the **AdaExplore, 50-step, gpt-5-mini** rows of Table 1 in the paper:

| Config | Benchmark | Problems | Steps / problem | Model |
|---|---|---|---|---|
| `config/KB-l2/config_KB-l2_AdaExplore_50.yaml` | KernelBench Level 2 | 100 (`test_list_2.txt`) | 50 | `gpt-5-mini` (OpenAI) |
| `config/KB-l3/config_KB-l3_AdaExplore_50.yaml` | KernelBench Level 3 | 50 (`test_list_3.txt`) | 50 | `gpt-5-mini` (OpenAI) |

A Level 1 config (`config/KB-l1/config_KB-l1_AdaExplore_50.yaml`, driven by `test_list_1.txt`) is also provided.

Both configs already point to the bundled cross-task skill memory at `results/memory/general_memory_v1_200.txt`, so you can launch the reproduction directly without rerunning the Adapt stage.

> **Note — shipped config defaults vs. paper setup.** The table above describes the canonical paper configuration (`gpt-5-mini`, OpenAI, Ampere, remote judge). The KB-l2 config in this branch is currently checked in with local-development defaults instead: `server_type: claude` / `model_name: claude-opus-4-7`, `use_remote_eval: false`, `gpu_name: RTX-4090-D` / `gpu_architecture: Ada`, `nsight_ncu_sudo` set for the always-on ncu stage, and tuned search knobs (`knowledge_1_threshold: 5`, `small_step_limit: 5`). Edit `server_type` / `model_name` / `use_remote_eval` / `gpu_*` back to the paper values to reproduce Table 1.

**Hardware.** The paper experiments use A6000 with a fixed frequency (1500MHz) at fp32.

**Concurrency.** Each config runs `num_processes` agent workers in parallel, all submitting evaluations to the same online judge.

**Summarizing a finished run.** After a config completes, use `tool_scripts/stats.py` to recover the Table 1 numbers:

```bash
python tool_scripts/stats.py --log_folder outputs/KB-l2_AdaExplore_50 --step 50
python tool_scripts/stats.py --log_folder outputs/KB-l3_AdaExplore_50 --step 50
```

This reports correctness rate and aggregate speedup statistics (mean / median / fast_p threshold accuracies). If you want to re-measure performance with more trials before reporting, run `tool_scripts/re_evaluate.py` first (see the Evaluation section below).

**Cost / runtime expectations.** A full 50-step Level 2 run hits the OpenAI API on the order of $10^4$ requests; budget several hours of wall-clock time on a single Ampere GPU and a non-trivial token bill. Both can be reduced for smoke testing by editing the test list (e.g. `2 1` to evaluate only level-2 problem 1) or by lowering `total_steps`.

## Evaluation

AdaExplore uses a lightweight evaluation harness built around `online_judge/app_with_queue.py` and `src/eval_subprocess_runner.py`. The online judge exposes an API service with per-GPU bounded concurrency and queueing, so many agent workers can submit evaluations at once without oversubscribing a device; requests are routed to the least busy GPU and executed in isolated subprocesses to keep crashes or illegal memory accesses from taking down the main service.

For result quality, the harness separates correctness from performance measurement. Correctness is checked over one or more randomized trials and uses a small numerical tolerance (`atol=rtol=5e-2`) to avoid rejecting kernels over insignificant floating-point noise, while performance is measured after warmup across multiple runs and summarized into runtime statistics after removing outliers. On the same correct path the harness also records the heavy-op dispatch trace and (best-effort) the Nsight profile described above.

**Fresh random inputs per run.** Every `get_inputs()` in the dataset (`datasets/KernelBench/level{1-4}`, `datasets/KernelBench_syn/syn_v1`) now begins with `torch.seed()`, which reseeds from OS entropy. The harness previously called `torch.manual_seed(trial_seed)` immediately before `get_inputs()`, so test inputs were identical across runs and a kernel could overfit to one fixed input draw; the reseed makes each run draw new inputs. The reseed line is inserted in bulk by `tool_scripts/randomize_get_inputs.py`.

For post-processing, `tool_scripts/re_evaluate.py` can re-run the saved best kernels in a log directory, for example `python tool_scripts/re_evaluate.py --log_folder outputs/<run> --num_correct_trials 5 --num_perf_trials 100 --use_remote_eval`.
To summarize a completed run, use `tool_scripts/stats.py`, e.g. `python tool_scripts/stats.py --log_folder outputs/<run> --step 50`, which reports accuracy and aggregate speedup statistics.

Additional helper scripts under `tool_scripts/`:
- `fill_summary.py` / `fill_summary_l2.py` — build `outputs/KB-l{level}_AdaExplore_50/Summary.xlsx` with one row per test (name, TRT baseline ms, agent result, ratio, fail flag, implementation), joining the agent results against `baseline_time_trt.json`.
- `nsight_runner.py` — standalone whole-process entry point that nsys/ncu attach to: it loads the reference arch and candidate `ModelNew` exactly as `src/eval.py` does, runs warmup + forward passes, and does no timing of its own (the profiler records the GPU work).
- `eval_one_kernel.py` — run a single saved kernel through the harness for ad-hoc inspection.

