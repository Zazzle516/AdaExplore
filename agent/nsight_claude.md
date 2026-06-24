# Nsight profiling pipeline: nsys → filter → targeted ncu

## Pipeline (target design)

Profile each iteration's **correct** kernel in three stages, so ncu's expensive hardware-counter
pass only ever touches the kernel that actually matters:

```
nsys profile (cheap trace, no replay)        →  every kernel the candidate launched, ranked by GPU time
  └─ filter: pick the dominant candidate kernel(s)  (drop framework/library noise)
       └─ ncu -k <that kernel>, rich --metrics  (targeted replay, deep counters)
            └─ assembled summary dict → evaluator prompt
```

This **inverts** today's flow. Currently ncu replays *every* kernel with a thin `--set basic` and
the noise is discarded afterward. Instead, let the cheap nsys pass say what matters, then point ncu
at only that — which is both cheaper (1–3 kernels, not ~11) and richer (a large explicit metric set
becomes affordable). ncu is a **default stage**, not a user toggle.

All code lives in `src/nsight_profiler.py` unless noted. Everything is best-effort and must **never**
raise into the eval path (mirror the existing `# noqa: BLE001` + subprocess-timeout pattern); any
failure surfaces as `error` / `nsys_error` / `ncu_error` and the rest of the dict still stands.

## Why (background)

- **Framework noise buries the real kernel.** A correct candidate also launches many tiny
  PyTorch/cuDNN/CUTLASS helper kernels. In run `outputs/KB-l2_AdaExplore_50/2_15`, step 8 had **10 of
  11** kernels each <1% of GPU time (e.g. `void at::native::vectorized_elementwise_kernel`,
  `reduce_kernel`), each carrying a full ncu block, while the kernel at 85–99% of runtime sat buried
  in the middle of the JSON. Candidate kernels are name-separable: custom/Triton kernels have clean
  names (`conv_transpose3d_kernel`, `triton_...`); framework ones are `void at::… / void cudnn::… /
  void cutlass::…`.
- **ncu fields are thin and report *that* not *why*.** `--set basic` gives 6 counters — it shows
  occupancy is 16% but never whether registers, shared memory, or block count is the cap.
- **Prereq fixes already landed** (kept so the history isn't lost): the config-loader now coerces
  YAML bools both ways (`agent/utils.py:load_config_from_yaml`), and the ncu CSV metric mapping was
  corrected (`_NCU_METRICS`) with `_roofline_bound` classifying on SOL memory throughput.

## Current state → target

| Aspect | Today | Target |
|---|---|---|
| nsys pass | always (correct kernels) | unchanged |
| ncu pass | opt-in via `nsight_ncu` flag | **default stage** (field removed) |
| noise filtering | none (comment promises it, no code) | candidate-only |
| ncu metrics | 6 (`--set basic`) | rich explicit `--metrics` bundle |
| ncu scope | every kernel | `-k` dominant candidate only |

---

## Implementation

### Stage 1 — select the candidate kernel(s) from the nsys ranking

`_is_framework_kernel(name)` classifies a kernel name as library/framework noise vs.
candidate-authored. Framework kernels start with `void ` and/or carry a C++ namespace marker;
candidate kernels (Triton / custom CUDA from `load_inline`) have neither.

```python
_FRAMEWORK_MARKERS = ("at::", "cudnn", "cutlass", "cublas", "cub::", "thrust::", "void <unnamed>")

def _is_framework_kernel(name) -> bool:
    """True for library/framework kernels (PyTorch/cuDNN/CUTLASS/...) vs. candidate-authored ones."""
    if not name or not isinstance(name, str):
        return False
    low = name.lower()
    if any(m in low for m in _FRAMEWORK_MARKERS):
        return True
    # Template-instantiated C++ library kernels are emitted as `void ns::kernel<...>(...)`.
    return low.startswith("void ") and "::" in name
```

`_candidate_kernel_names(...)` returns the keep-set **before** ncu runs, from the nsys kernel list
(already sorted desc by `time_pct` at line ~108). Keep a kernel if it is not framework, or is above a
GPU-time floor, or is within the top-N — the `keep_top_n` floor guarantees ≥1 name even when a
candidate legitimately calls a library GEMM as its main op. A missing `time_pct` (dropped when `None`
in `_normalize_kernel_row`) counts as `0.0`.

```python
def _candidate_kernel_names(kernels, *, time_floor: float = 1.0, keep_top_n: int = 3) -> list[str]:
    """Names of candidate-authored / dominant kernels, derived from the nsys ranking."""
    names = []
    for i, k in enumerate(kernels or []):
        tp = k.get("time_pct")
        tp = tp if isinstance(tp, (int, float)) else 0.0
        name = k.get("name")
        if name and ((not _is_framework_kernel(name)) or tp >= time_floor or i < keep_top_n):
            names.append(name)
    return names
```

### Stage 2 — targeted, richer ncu

**Re-key `_NCU_METRICS` on internal metric IDs and expand the bundle.** With explicit `--metrics`,
ncu's CSV "Metric Name" column prints the internal IDs (not display names), so keys move to IDs — a
deliberate, *consistent* reversal of the earlier display-name fix now that `--set basic` is gone. The
mapped value names stay stable so `_roofline_bound` and the prompt keep working
(`compute_throughput_pct` / `memory_throughput_pct` / `dram_throughput_pct` /
`achieved_occupancy_pct` / `registers_per_thread` all still produced).

```python
_NCU_METRICS = {
    # throughput / roofline (kept — _roofline_bound depends on these three)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":                 "compute_throughput_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":               "dram_throughput_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed":                "l2_throughput_pct",
    "gpu__time_duration.sum":                                           "duration",
    # occupancy + WHY it is capped (new — answers the run_2_15 gap)
    "sm__warps_active.avg.pct_of_peak_sustained_active":                "achieved_occupancy_pct",
    "launch__registers_per_thread":                                     "registers_per_thread",
    "launch__occupancy_limit_registers":                                "occ_limit_registers",
    "launch__occupancy_limit_shared_mem":                               "occ_limit_shared_mem",
    "launch__occupancy_limit_warps":                                    "occ_limit_warps",
    "launch__occupancy_limit_blocks":                                   "occ_limit_blocks",
    # launch config (new)
    "launch__grid_size":                                                "grid_size",
    "launch__block_size":                                               "block_size",
    "launch__waves_per_multiprocessor":                                 "waves_per_sm",
    "launch__shared_mem_per_block_static":                              "shared_mem_per_block",
    # cache behavior (new)
    "l1tex__t_sector_hit_rate.pct":                                     "l1_hit_rate_pct",
    "lts__t_sector_hit_rate.pct":                                       "l2_hit_rate_pct",
    # dram traffic (new — feeds arithmetic-intensity / roofline)
    "dram__bytes_read.sum":                                             "dram_bytes_read",
    "dram__bytes_write.sum":                                            "dram_bytes_write",
}
```
> **Verify the IDs against the installed ncu** (`ncu --query-metrics` / `--list-metrics`) — a few vary
> by version. `_parse_ncu_csv` maps unknown metrics to nothing, so a stray ID is harmless, but a
> *renamed* one silently goes missing; double-check the roofline-critical three.

**Make `_run_ncu` targeted + metric-driven.** Replace `--set basic` with `--metrics <ids>`; accept
`candidate_names` and, when non-empty, add a kernel-name filter (`--kernel-name-base demangled`
because the nsys "Name" is demangled, so the regex must match that form). Targeting only the candidate
kernels is what makes the larger metric list affordable.

```python
base_cmd = [
    _NCU,
    "--metrics", ",".join(_NCU_METRICS.keys()),
    "--csv",
    "--target-processes", "all",
]
if candidate_names:
    base_cmd += [
        "--kernel-name-base", "demangled",
        "-k", "regex:" + "|".join(re.escape(n) for n in candidate_names),
    ]
base_cmd += _runner_cmd(kernel_path, run_args, num_iters=1)
```
`_parse_ncu_csv` needs no structural change — it already pivots on the "Metric Name" column and looks
up `_NCU_METRICS.get(metric)`; that column now carries the internal IDs the re-keyed dict expects.
(Add `import re`.) **Fallback:** if the targeted run yields no `ncu_kernels` (regex matched nothing),
retry **once with the same `--metrics` but no `-k`** — identical metrics keep parsing consistent, and
Stage 3's reconcile then trims framework noise from the unfiltered result. No empty profile.

### Stage 3 — assemble in `profile_kernel`

```python
    result.update(_run_nsys(kernel_path, run_args, num_iters, timeout))
    candidate_names = _candidate_kernel_names(result.get("kernels") or [])

    # ncu is a default stage — always attempt it (guarded by _NCU existence + try/except).
    if os.path.exists(_NCU):
        try:
            result.update(_run_ncu(kernel_path, run_args, ncu_sudo, timeout, candidate_names))
        except subprocess.TimeoutExpired:
            result["ncu_error"] = f"ncu timed out after {timeout}s"
        except Exception as e:  # noqa: BLE001 -- never break eval
            result["ncu_error"] = f"ncu exception: {e}"
    else:
        result["ncu_error"] = f"ncu not found at {_NCU}"

    if result.get("kernels"):
        try:
            _filter_noise(result)   # trims the nsys display list + reconciles ncu_kernels by name
        except Exception:  # noqa: BLE001 -- filtering never breaks eval
            pass
    return result
```

`_filter_noise(result, *, time_floor=1.0, keep_top_n=3)` mutates in place: filter `result["kernels"]`
by `_is_framework_kernel` / `time_floor` / `keep_top_n`, reconcile `result["ncu_kernels"]` to the kept
names (keep an ncu entry if its name is kept **or** it is not framework — guards nsys/ncu
name-mangling mismatches), and record `result["noise_kernels_dropped"] = <count>` (no silent
truncation). With targeted ncu it usually has nothing to drop on the ncu side; it still matters for
the unfiltered fallback path.

### Make ncu a default stage (remove the `nsight_ncu` toggle)

ncu currently is gated by a `nsight_ncu` boolean threaded CLI/YAML → `profile_kernel(ncu=...)`. Make it
unconditional and stop exposing it. **Rule for every file:** delete `nsight_ncu` plumbing; make the
ncu pass unconditional; **keep every `nsight_ncu_sudo` line** (ncu still needs root). Do *not* add a
default-on `store_true` (`default=True` on `store_true` recreates the config-loader bool bug noted
above). Surface verified with `grep -rn nsight_ncu` (outputs/ snapshots excluded):

- **`src/nsight_profiler.py`** — drop the `ncu: bool = False` param from `profile_kernel` (line ~342;
  keep `ncu_sudo`); the Stage-3 snippet above already runs ncu unconditionally. Update the module
  docstring (lines ~14-18): "ncu (opt-in, needs root)" → "ncu (default stage, needs root)".
- **`src/eval.py`** — delete the `nsight_ncu: bool = False` line from `eval_kernel_against_ref` (377),
  `_local_subprocess_eval` (1221), `wrapped_eval_kernel_against_ref` (1341); drop `ncu=nsight_ncu`
  from the `profile_kernel` call (678); drop `'nsight_ncu': nsight_ncu` from the subprocess JSON
  (1251); drop `nsight_ncu=nsight_ncu` from the forward (1454). Keep each `nsight_ncu_sudo` line
  (378/1222/1342/679/1252/1455). Update the docstring (384-385) and comments (647, 1408).
- **`src/eval_subprocess_runner.py`** — remove `nsight_ncu = args.get('nsight_ncu', False)` (47) and
  the forward (74); keep lines 48 and 75.
- **`agent/actions.py`** — remove `nsight_ncu=getattr(args, 'nsight_ncu', False)` at both call sites
  (238, 307); keep the `nsight_ncu_sudo=getattr(...)` lines (239, 308).
- **`agent/agent_entry.py`** — delete the `--nsight_ncu` argument (279); keep `--nsight_ncu_sudo`
  (280); update the section comment (277-278).
- **`tool_scripts/eval_one_kernel.py`** — delete the `--nsight_ncu` argument (24-25) and the
  `nsight_ncu=args.nsight_ncu` forward (49); keep `--nsight_ncu_sudo` (26-27) and its forward (50).
- **`config/KB-l2/config_KB-l2_AdaExplore_50.yaml`** — remove `nsight_ncu: true` (17); keep
  `nsight_ncu_sudo` (18). **`config/KB-l1/config_KB-l1_AdaExplore_50.yaml`** — no `nsight_ncu` key;
  add `nsight_ncu_sudo` if run on a non-root host. Stale `nsight_ncu:` keys are harmless after the arg
  is removed (`load_config_from_yaml` skips keys with no matching attr via the `hasattr` guard,
  `agent/utils.py:387`) but clean them. Auto-saved `outputs/**/config.yaml` are generated — ignore.

### Document the new fields in the evaluator prompt

Extend the `NSIGHT_PROFILE` "how to read it" block (`agentprompt/evaluator_prompt.py:218-226`) so the
richer counters arrive documented, not as bare keys — call out `occ_limit_registers /
occ_limit_shared_mem / occ_limit_warps / occ_limit_blocks` as "what is capping occupancy" and the
`l1_hit_rate_pct` / `l2_hit_rate_pct` / `dram_bytes_*` group. This is the only change outside
`src/nsight_profiler.py` besides the eval-chain plumbing.

## Consequences

- **Root/sudo:** ncu now runs for every correct kernel. On a non-root host with no `nsight_ncu_sudo`,
  each correct kernel hits a graceful `ncu_error` (nsys data still stands) — configure
  `nsight_ncu_sudo`; that is why the field is retained.
- **Cost:** ncu replays kernels, so eval slows down — targeting 1–3 kernels via `-k` instead of ~11
  keeps total replays well below the old `--set basic`-over-everything cost. Watch the per-subprocess
  `timeout`.
- **Remote path unaffected:** `nsight_ncu` / `nsight_ncu_sudo` were never forwarded to remote judges
  (`src/eval.py:1408`), so this changes only the local path.

## Deferred

- **Pre-digested one-line verdict.** A deterministic `verdict` string (e.g. *"dominant kernel
  `conv_transpose3d_kernel` = 85% GPU time; compute-bound 69% SM, 16% occupancy capped by registers
  → register-pressure-limited"*) computed in Python and surfaced prominently in `NSIGHT_PROFILE`, so
  the evaluator reads the conclusion instead of re-deriving it each step. The new `occ_limit_*`
  counters make this much stronger; build it once this pipeline lands.

## Verification

1. **Selection/filter check:** feed a synthetic nsys kernel list mixing framework entries (`void
   at::native::...` at `time_pct=0.3`) with a candidate (`conv_transpose3d_kernel` at `time_pct=85`);
   assert `_candidate_kernel_names` returns the candidate + top-N and excludes sub-floor framework
   names, and `_filter_noise` sets `noise_kernels_dropped`.
2. **ncu argv check:** assert the built command contains `--metrics ...`, `--kernel-name-base
   demangled`, and `-k regex:conv_transpose3d_kernel` when candidate names are supplied — and omits
   `-k` in the fallback shape.
3. **End-to-end:** run a known-correct kernel through `profile_kernel` (mirror `src/eval.py`
   ~644-689); confirm `ncu_kernels` holds only the dominant kernel(s), each carrying the new fields
   (`occ_limit_registers`, `l2_hit_rate_pct`, `dram_bytes_read`, …), and that for the run_2_15 case
   `occ_limit_registers` flags registers as the occupancy cap.
4. **Default-stage check:** `grep -rn "nsight_ncu" --include=*.py --include=*.yaml .` returns only
   `nsight_ncu_sudo`; `python tool_scripts/eval_one_kernel.py --help` shows `--nsight_ncu_sudo` but no
   `--nsight_ncu`; running it on a correct kernel produces `ncu_kernels` with no enable flag passed.
5. **Prompt spot-check:** render the evaluator prompt (`NSIGHT_PROFILE` injection,
   `agentprompt/evaluator_prompt.py:323-336`) and confirm the JSON block is concise (1–3 kernels) and
   carries the richer counters.

---

# Follow-up: surface ncu's own rule-engine verdicts (replaces the static "How to read it")

## Context

The targeted-ncu pipeline above landed and works: the per-kernel counters reach the evaluator
prompt accurately. But on the `outputs/KB-l2_AdaExplore_50/2_15` run the agent's `fast_p` plateaued
(~0.25, still ~4× slower than cuDNN) because it kept chasing the **wrong lever**. The cause is the
prompt itself: the hand-written *"How to read it:"* block in `agentprompt/evaluator_prompt.py` is
roofline-blind — it says *"low `occ_limit_registers` → cut `registers_per_thread` to raise
occupancy"* with no caveat. For a kernel already at 95% memory SOL (`roofline_bound: memory_bound`),
raising occupancy buys nothing, yet the static text steered the agent there anyway.

Fix: stop hand-authoring the interpretation. NVIDIA Nsight Compute ships an expert **rule engine**
that emits per-kernel, roofline-aware verdicts (e.g. *"theoretical occupancy 25% limited by
registers, Est. Local Speedup: 75%"* — or, for a memory-bound kernel, *"Memory is more heavily
utilized…"*). We surface ncu's own text and delete our static advice. This realizes (and supersedes)
the **Deferred "pre-digested verdict"** idea above: rather than computing the verdict in Python, we
use ncu's authored verdict directly.

### Verified facts (drive the design)
- The nsight dict (`KernelExecResult.metadata["nsight"]`) is **purely informational**. Only two
  consumers touch it: `src/eval.py` stores it, and `agentprompt/evaluator_prompt.py` (~L336-346)
  gates on `nsight.get("kernels") or nsight.get("ncu_kernels")` then JSON-dumps the whole dict via
  `format_nsight_summary`. No MCTS reward / skill-memory code reads any inner field. **Restructuring
  ncu output is safe.**
- A **single** ncu invocation combining `--metrics <internal IDs>` + `--section <name>` + `--csv`
  returns both our existing numeric metric rows **and** rule rows. The CSV gains columns:
  `Section Name`, `Rule Name`, `Rule Type` (OPT/WRN/INF), `Rule Description` (the expert prose),
  `Estimated Speedup Type` (local/global), `Estimated Speedup` (e.g. `75`). Rule rows carry a
  `Kernel Name` but a **blank `Metric Name`**, so they skip the existing metric pivot naturally.
  → This is **additive**: keep `_NCU_METRICS` and `_parse_ncu_csv`'s pivot; layer rules on top.
- Installed ncu is **2025.1.1**. Decided section set: **SpeedOfLight, Occupancy,
  MemoryWorkloadAnalysis, SchedulerStats** — all four reliably emit roofline-aware rules with
  Estimated Speedup. WarpStateStats / SourceCounters are **excluded**: their rules need source-level
  PC sampling (warning `smsp__pcsamp_sample_count could not be found` → rules suppressed), so they
  would cost extra replay passes for no verdict text.

## Changes

### 1. `src/nsight_profiler.py` — add `_NCU_SECTIONS` constant
After `_NCU_METRICS` (~L331):
```python
# ncu's section rule engine emits per-kernel, roofline-aware verdicts (OPT/WRN
# text + estimated speedup). These four reliably produce rules without needing
# source-level PC sampling. Adding sections multiplies ncu replay passes, so this
# list is the single cost/timeout knob (drop SchedulerStats first if needed).
_NCU_SECTIONS = ("SpeedOfLight", "Occupancy", "MemoryWorkloadAnalysis", "SchedulerStats")
```

### 2. `src/nsight_profiler.py` — `_run_ncu` / `_build_cmd`
In `_build_cmd` (~L347-361), after the `--metrics metrics_arg` entry, append a `--section <name>`
pair for each `_NCU_SECTIONS` entry. Leave everything else intact: `--csv`, `--target-processes
all`, the targeted `--kernel-name-base demangled -k regex:...`, the unfiltered fallback retry, the
sudo/root branch, and `num_iters=1`. The fallback rebuilds via `_build_cmd`, so it inherits the
sections automatically. Keeping `--metrics` preserves every existing stable numeric field.

### 3. `src/nsight_profiler.py` — `_parse_ncu_csv` collect rule rows
The header detector (~L407) still matches (same `"Kernel Name"` header with sections added).
Restructure the row loop (~L411-425):
- A row is a **rule row** when `Metric Name` is blank/absent and `Rule Description` (or `Rule Name`)
  is populated. The current `if not kname or not metric: continue` wrongly drops these — change it
  so a populated `kname` with a rule is kept even when `metric` is blank.
- Metric rows: unchanged — pivot `Metric Name` through `_NCU_METRICS.get` → stable key.
- Rule rows: append to `entry.setdefault("rules", [])` as:
  ```python
  {"section": <Section Name>|None, "type": <Rule Type>|None,
   "name": <Rule Name>|None, "desc": <Rule Description>|None,
   "speedup_pct": float(<Estimated Speedup>) | None}
  ```
  Coerce `speedup_pct` in its own try/except; all columns via `row.get(...)` so older ncu degrades to
  `None`, never raises. **Drop `Rule Type == "INF"`** to cut noise (keep OPT + WRN).
- `_roofline_bound` stays unchanged — cheap deterministic cross-check / fallback when a rule is
  suppressed.

### 4. `src/nsight_profiler.py` — `format_nsight_summary`
No change. The `rules` arrays serialize fine; INF-drop + 4-section subset keep the payload modest.

### 5. `agentprompt/evaluator_prompt.py` — NSIGHT_PROFILE template (~L208-238)
**Delete the entire "How to read it:" block** (the field-by-field interpretive advice, including the
wrong "cut registers" rule). Keep the short lead-in paragraph and the `{nsight_summary}` JSON block.
Add one concise line directing the evaluator to the embedded `rules`: ncu's own roofline-aware expert
verdicts — treat them as authoritative and prioritize the fix with the highest `Estimated Speedup` /
`speedup_pct`. The injection site and its gate (~L336-346) are unchanged.

## Verification
1. **Static argv:** built command contains both `--metrics ...` and the four `--section` flags;
   fallback shape still omits `-k`.
2. **Parse unit check:** feed a captured CSV with one metric row + one OPT rule row + one INF row;
   assert numeric stable fields un-regressed, `rules` has the OPT entry with `speedup_pct` parsed,
   INF excluded.
3. **End-to-end:** run `profile_kernel` on an existing `outputs/KB-l2_AdaExplore_50/2_15/step_*.py`
   (note: `global_best_kernel_10.py` no longer exists; use a present `step_NN.py`), watch ncu's pass
   count against the 600s timeout, confirm `ncu_kernels[].rules` carries the SOLBottleneck/Occupancy
   verdicts with speedups.
4. **Prompt render:** rendered NSIGHT_PROFILE shows the rule text and the old "cut registers"
   sentence is gone.

## Risks
- **Timeout/cost (primary):** sections multiply replay passes (~14 observed for 5 sections); four
  sections on 1–3 targeted kernels should fit 600s. `_NCU_SECTIONS` is the single knob — drop
  SchedulerStats first if timeouts appear.
- **CSV columns vary by ncu version:** older ncu may omit `Rule Type`/`Estimated Speedup`; `.get` +
  per-field try/except degrade gracefully.
- **Empty `rules`** for a kernel at its roofline ceiling is legitimate; `roofline_bound` + counters
  remain the fallback signal.
- All new paths sit under the existing `try/except` + `# noqa: BLE001`, so nothing raises into eval.
