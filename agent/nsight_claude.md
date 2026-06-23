# Plan: Integrate NVIDIA Nsight profiling into the AdaExplore optimization loop

## Context

AdaExplore optimizes GPU kernels with an MCTS loop: a proposer/tuner writes Triton/CUDA
kernels, `src/eval.py` measures correctness + wall-clock latency (`fast_p` speedup vs a
PyTorch/TRT baseline), and an **evaluator** LLM turns those metrics into `small_guidance` /
`large_guidance` that steer the next round.

Today the only performance signal the evaluator sees is a single scalar latency (`fast_p`)
from `torch.cuda.Event`. It has **no visibility into *why* a kernel is slow** — occupancy,
memory throughput, warp-stall reasons, launch config, per-op timeline. As `Note.md`/`backup.md`
document, this causes the evaluator to misdiagnose bottlenecks (e.g. "use cuBLAS for the heavy
GEMM" is never suggested because nothing tells it the kernel is compute-bound at 40% of peak).

**Goal:** capture detailed per-operator execution data with NVIDIA Nsight and feed a compact,
parsed summary into the evaluator prompt, so next-round guidance is grounded in real hardware
counters rather than a single latency number.

### Environment facts (verified)
- `nsys` 2024.6 and `ncu` 2025.1 are **already installed** at `/usr/local/cuda/bin/`. No install step needed.
- GPU: RTX 4090, driver 580.126.20, CUDA 12.8.
- **`RmProfilingAdminOnly: 1` and `perf_event_paranoid: 4`** → `ncu` hardware counters require **root/sudo**; `nsys` tracing runs **without root**. This drives the nsys-default / ncu-opt-in design.

### Decisions
- **nsys runs by default** for every correct kernel (no-root). **ncu is opt-in** via `nsight_ncu` (needs sudo; password supplied via `nsight_ncu_sudo`).
- **Profile only the currently executed kernel** — the candidate `ModelNew` forward under evaluation. Do not profile the reference/baseline, and do not return a cross-process top-N list; return exactly the GPU kernel(s) this candidate launches.
- **Local eval path first** (`src/eval.py`), validated with `use_remote_eval=false`. Remote judge wiring is out of scope for v1.

### Existing pattern to mirror
`backup.md` specifies `src/triton_error_parser.py`: a self-contained parser whose output is
stashed in `KernelExecResult.metadata` (`metadata["compilation_error_parsed"]`, set at
`src/eval.py:947`) and later rendered into a dedicated evaluator-prompt section. This Nsight
work follows the identical parse → metadata → prompt-section pipeline.

---

## Implementation

### 1. New standalone runner: `tool_scripts/nsight_runner.py`
nsys/ncu profile a **whole process**, so we need a minimal driver they can wrap.
- CLI: `--kernel_path --test_source --level --problem_id --device --dtype --backend --num_iters`.
- Reuses existing loaders: `agent.utils.load_test_source` + the same temp-file model-loading
  helper used in `src/eval.py` (`load_custom_model_with_tempfile`) to build `ModelNew` and inputs
  exactly as eval does (consistency with the measured path).
- Does warmup then runs the forward `--num_iters` times, with `torch.cuda.synchronize()`. No
  timing logic of its own — the profiler records it. Keep iters small (e.g. 5) because ncu replay is slow.

### 2. New parser/orchestrator module: `src/nsight_profiler.py`
Mirrors `src/triton_error_parser.py` in spirit (self-contained, never raises into the eval path).
- `profile_kernel(kernel_path, *, run_args, device, ncu=False, ncu_sudo="", timeout=...) -> dict`:
  - **nsys path** (always): `nsys profile -o <tmp> --force-overwrite true python tool_scripts/nsight_runner.py ...`,
    then `nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --format json <tmp>.nsys-rep`.
    Parse JSON → **the kernel(s) the candidate actually launched** (its own Triton/CUDA kernel, not a
    cross-process top-N): name, instances, total/avg ms, % of GPU time, grid/block, plus memcpy time.
    (Use `nsys stats` JSON, not raw sqlite, to avoid a schema dependency.)
  - **ncu path** (opt-in, when `ncu=True`): `ncu --set basic --csv --target-processes all python tool_scripts/nsight_runner.py ...`
    on a **single iteration**. Parse CSV → for the candidate's kernel: SM/achieved occupancy, DRAM
    throughput %, compute(SM) throughput %, L2 hit %, registers/thread, dominant warp-stall reason,
    roofline bound (memory vs compute). ncu needs root here → invoke via `sudo -S` feeding the
    `ncu_sudo` password on stdin (fall back to direct call if already root). On permission failure,
    set `{"ncu_error": "..."}` and continue (nsys data still useful).
  - Returns a dict describing **only the currently executed kernel**.
  - Wrap everything in try/except + subprocess timeout; on any failure return `{"error": "..."}` so
    profiling never breaks evaluation.

### 3. Hook into the local eval path: `src/eval.py`
- Extend `eval_kernel_against_ref` (def at `src/eval.py:359`) signature with
  `nsight_ncu: bool = False, nsight_ncu_sudo: str = ""`.
- After performance stats are attached (right after `src/eval.py:637`
  `kernel_exec_result.runtime_stats = runtime_stats`), inside the `correctness` block, add:
  ```python
  if kernel_exec_result.correctness:
      from src.nsight_profiler import profile_kernel
      kernel_exec_result.metadata["nsight"] = profile_kernel(
          ..., ncu=nsight_ncu, ncu_sudo=nsight_ncu_sudo)
  ```
  nsys always runs for a correct kernel; ncu only when `nsight_ncu=True`. Gated, best-effort,
  never raises. Needs the kernel source on disk → reuse the tempfile already created during model
  loading, or write `custom_model_src` to a temp `.py` for the runner.
- Thread the same flags through `wrapped_eval_kernel_against_ref` (def at `src/eval.py:1262`) so
  callers can pass them; remote branch ignores them in v1 (documented).

### 4. Config + plumbing
- Add to `config/KB-l2/config_KB-l2_AdaExplore_50.yaml` (and document in the config loader)
  **exactly two fields**: `nsight_ncu` (bool, enable the ncu hardware-counter pass) and
  `nsight_ncu_sudo` (string, the sudo password used to run ncu as root). nsys profiling of the
  current kernel runs automatically for every correct kernel; these two only gate/enable ncu.
- Pass `args.nsight_ncu` / `args.nsight_ncu_sudo` from the agent call sites that invoke eval. The
  proposer/tuner evaluation goes through `wrapped_eval_kernel_against_ref` in `agent/actions.py`
  (`single_large_step`/`single_small_step`); forward the flags there.

### 5. Inject into the evaluator prompt: `agentprompt/evaluator_prompt.py`
- Add a `NSIGHT_PROFILE` template section (mirror `STRUCTURAL_ALERT`, evaluator_prompt.py ~L174).
- In `generate_evaluator_prompt`, read `run_info.metadata.get("nsight")`; if present and non-error,
  render via `format_nsight_summary(...)` (a plain `json.dumps` dump of the dict) and append the
  section after the metrics block.

---

## Critical files
- **New** `tool_scripts/nsight_runner.py` — process the profiler wraps.
- **New** `src/nsight_profiler.py` — orchestrate nsys/ncu + parse + format (mirrors `src/triton_error_parser.py`).
- `src/eval.py` — hook after L637; extend `eval_kernel_against_ref` (L359) and `wrapped_eval_kernel_against_ref` (L1262).
- `agentprompt/evaluator_prompt.py` — new prompt section + render in `generate_evaluator_prompt`.
- `agent/actions.py` — forward profiling flags through eval calls.
- `config/KB-l2/config_KB-l2_AdaExplore_50.yaml` + config loader — new flags.
- `agent/nsight_claude.md` — design documentation (this file).

## Modification locations

### New files
| File | Contents |
|------|----------|
| `tool_scripts/nsight_runner.py` | Standalone driver the profiler wraps. CLI `--kernel_path --test_source --level --problem_id --device --dtype --backend --num_iters`; warmup + forward ×N with `torch.cuda.synchronize()`, no timing of its own. Reuses `agent.utils.load_test_source` + `load_custom_model_with_tempfile`. |
| `src/nsight_profiler.py` | `profile_kernel(kernel_path, *, run_args, device, ncu=False, ncu_sudo="", timeout=...) -> dict`. nsys path always; ncu path opt-in via `sudo -S`. try/except + subprocess timeout, never raises. |

### Edits to existing files
| File | Location | Change |
|------|----------|--------|
| `src/eval.py` | **L359** `eval_kernel_against_ref` | Extend signature: `nsight_ncu: bool = False, nsight_ncu_sudo: str = ""`. |
| `src/eval.py` | **after L637** (`kernel_exec_result.runtime_stats = runtime_stats`, inside `correctness` block) | Add gated `profile_kernel(...)` writing `kernel_exec_result.metadata["nsight"]`. Reuse model-loading tempfile or write `custom_model_src` to temp `.py`. |
| `src/eval.py` | **L1262** `wrapped_eval_kernel_against_ref` | Thread the same two flags through; remote branch ignores them in v1. |
| `agentprompt/evaluator_prompt.py` | **L174** (next to `STRUCTURAL_ALERT`) | Add `NSIGHT_PROFILE` template section. |
| `agentprompt/evaluator_prompt.py` | **L208** `generate_evaluator_prompt` | Read `run_info.metadata.get("nsight")`; if present and non-error, render via `format_nsight_summary(...)` and append after the metrics block. |
| `agent/actions.py` | **L222** `single_small_step` call to `wrapped_eval_kernel_against_ref` | Forward `args.nsight_ncu` / `args.nsight_ncu_sudo`. |
| `agent/actions.py` | **L289** `single_large_step` call to `wrapped_eval_kernel_against_ref` | Forward `args.nsight_ncu` / `args.nsight_ncu_sudo`. |
| `config/KB-l2/config_KB-l2_AdaExplore_50.yaml` + config loader | — | Add `nsight_ncu` (bool) and `nsight_ncu_sudo` (string); document in loader. |
| `tool_scripts/eval_one_kernel.py` | — | Add `--nsight_ncu` / `--nsight_ncu_sudo` flags (used by verification step 5). |

## Reused utilities (avoid new code)
- `agent.utils.load_test_source` — load reference arch for the runner.
- `load_custom_model_with_tempfile` (src/eval.py) — build `ModelNew` identically to eval.
- `KernelExecResult.metadata` (`src/format.py`) — carrier for nsight data (same channel as `compilation_error_parsed`).
- `STRUCTURAL_ALERT` injection pattern in `evaluator_prompt.py` — template for the new section.

---

## Verification (end-to-end)
1. **Tools sanity**: `nsys --version`, `ncu --version` (already confirmed present).
2. **Runner alone**: `python tool_scripts/nsight_runner.py --test_source KB --level 2 --problem_id 1 ...`
   completes a forward pass without profiling. Then wrap with `nsys profile ...` and confirm a `.nsys-rep` is produced.
3. **Parser unit check**: call `src.nsight_profiler.profile_kernel(...)` on a known-good kernel
   (e.g. a `global_best` kernel under `outputs/`) → returns a populated dict; assert nsys path works without sudo.
4. **ncu gating**: with `nsight_ncu=true` and the `nsight_ncu_sudo` password set → confirm
   occupancy/throughput fields populate; with a wrong/empty password → confirm graceful
   `{"ncu_error": ...}` while the nsys data is still present.
5. **eval integration**: `python tool_scripts/eval_one_kernel.py <kernel> --level 2 --problem_id 1`
   (extend it with `--nsight_ncu` / `--nsight_ncu_sudo` flags) → `result.metadata["nsight"]`
   populated for a correct kernel with the current kernel's data; nsys-only by default, ncu when enabled.
6. **Prompt check**: run `generate_evaluator_prompt` on a metrics object carrying `metadata["nsight"]`
   → confirm the NSIGHT_PROFILE section renders and is absent when no nsight data exists.
7. **Loop smoke test**: one short MCTS run on level-2 problem 1 with `use_remote_eval: false`,
   `total_steps` small → confirm per-step metrics JSON in
   `outputs/KB-l2_AdaExplore_50/<pid>/` carry nsight data and the saved evaluator prompt shows the section.

## Out of scope (v1)
- Remote judge (`online_judge/app_with_queue.py`) profiling — local path only.
- Persisting `.nsys-rep`/`.ncu-rep` artifacts long-term (use temp files; optionally add a `nsight_keep_reports` flag later).
- Per-kernel ncu on every iteration by default (expensive); ncu stays opt-in.

---

## Open problem: Nsight guidance is thin (observed 2026-06-23, run `outputs/KB-l2_AdaExplore_50/2_15`)

**Symptom.** After the data path was fixed (see below), the evaluator's `small_guidance` /
`large_guidance` does cite real counters — e.g. step 3: *"conv_transpose3d kernel is compute-bound
at 69% with 254 registers/thread and only 16% occupancy — register pressure is killing
throughput"* — but the guidance feels under-leveraged relative to the data available. The model
re-derives the same diagnosis from raw numbers every step and sometimes anchors on the wrong kernel.

**Prerequisite fixes already landed (this run validated them):**
- **Config loader bool bug** (`agent/utils.py` `load_config_from_yaml`): a `store_true` flag whose
  default is `True` (e.g. `use_remote_eval`) could never be turned off from YAML — `use_remote_eval:
  false` was silently dropped, so the run used the **remote** path where ncu is not wired. Fixed to
  coerce YAML bools in both directions (`setattr(args, key, bool(value))`). This is why an earlier
  inspection saw nsys-only data with no ncu.
- **ncu CSV metric mapping** (`src/nsight_profiler.py` `_NCU_METRICS`): was keyed on internal metric
  ids (`sm__throughput.avg...`) but `ncu --set basic --csv` emits *display* names (`Compute (SM)
  Throughput`, `Achieved Occupancy`, `Registers Per Thread`, ...). Re-keyed to display names; added
  `memory_throughput_pct` (SOL memory) and switched `_roofline_bound` to classify on SOL memory.
- Confirmed: every **correct** kernel now carries fully-populated `ncu_kernels` (steps 1,2,3,7,8,10);
  `correctness=False` steps (4,5,6,9) have no nsight by design (the hook only profiles correct kernels).

**Root causes of the thin guidance (NOT a data-plumbing issue — data is present and cited):**

1. **Promised noise-filtering was never implemented.** The module comment
   (`src/nsight_profiler.py:44-45`) claims it "keeps the candidate's own kernels; drops
   framework/library noise," but there is **no filtering code** — every framework kernel leaks into
   both `kernels` and `ncu_kernels`. Measured dilution in this run:
   - step 8: **10 of 11** profiled kernels are <1% of GPU time (mostly `void at::native::
     vectorized_elementwise_kernel` / `reduce_kernel`), each carrying a full 6-counter ncu block.
   - step 7: 8 of 10 are <1%. step 3: 6 of 10. The one kernel that is 85–99% of runtime is buried.
   The candidate's own kernel is trivially separable: custom/Triton kernels have clean names
   (`conv_transpose3d_kernel`), framework ones are `void at::… / void cudnn::… / void cutlass::…`.

2. **Raw JSON dump, no pre-digested verdict.** `NSIGHT_PROFILE` pastes the dict + a generic "how to
   read it." Nothing surfaces the one-line bottleneck (*"dominant kernel compute-bound at 69% SM, 254
   reg/thread, occupancy capped at 16% → register-pressure-limited"*). The model must re-derive it
   each step, and on a noisy list it can latch onto the wrong kernel.

**Candidate fixes (not yet decided):**
- **A. Filter to candidate kernel(s):** drop sub-1%-GPU-time framework kernels (`void at::/cudnn::/
  cutlass::`) from both lists — implements the filtering the code already advertises; trims ~10 noise
  kernels to 1–3. (Keep a small floor, e.g. always keep top-N by time even if named like framework,
  so a candidate that legitimately calls a library kernel isn't dropped.)
- **B. Append a derived diagnosis line** per dominant kernel (roofline bound + the limiting counter)
  so the evaluator starts from an interpreted signal rather than raw numbers.
- **C. Minimal:** just rank by GPU time and cap to top-N (already sorted; add the cap).

A and B are complementary and the recommended pair; C is the low-effort fallback.
