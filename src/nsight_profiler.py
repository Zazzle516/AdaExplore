"""Orchestrate NVIDIA Nsight (nsys/ncu) profiling of a single candidate kernel.

This mirrors :mod:`src.triton_error_parser` in spirit: it is self-contained,
wraps everything in try/except + subprocess timeouts, and **never raises into the
eval path** -- on any failure it returns a dict with an ``error`` (or
``ncu_error``) key so profiling can never break evaluation.

Two profiling passes:

* **nsys** (always, no root required) -- traces the whole process and reports the
  GPU kernel(s) the candidate launched (name, instances, total/avg ms, % of GPU
  time) plus memcpy time. Parsed from ``nsys stats`` JSON, not raw sqlite, to
  avoid a schema dependency.
* **ncu** (default stage, needs root) -- hardware counters for the candidate's
  dominant kernel(s) only (selected from the nsys ranking, so ncu's expensive
  replay never touches framework noise): achieved occupancy *and what caps it*
  (register/shared-mem/warp/block occupancy limits), compute/memory/DRAM/L2
  throughput %, registers/thread, launch config, cache hit rates, DRAM traffic,
  and a roofline bound. Invoked via ``sudo -S`` feeding the password on stdin
  (falls back to a direct call when already root). On permission failure the
  result carries ``ncu_error`` and the nsys data still stands.

The single public entry point is :func:`profile_kernel`.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile

__all__ = ["profile_kernel", "format_nsight_summary"]

# Absolute paths -- these live under CUDA's bin dir and may not be on PATH for
# the subprocess environment eval runs in.
_NSYS = "/usr/local/cuda/bin/nsys"
_NCU = "/usr/local/cuda/bin/ncu"
_RUNNER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tool_scripts",
    "nsight_runner.py",
)

# Keep the candidate's own kernels; drop framework/library noise so the summary
# describes the kernel(s) this candidate actually authored/launched.
_DEFAULT_TIMEOUT = 600


# --------------------------------------------------------------------------- #
# Stage 1: candidate-kernel selection (framework-noise classification)
# --------------------------------------------------------------------------- #
# A correct candidate also launches many tiny PyTorch/cuDNN/CUTLASS helper
# kernels. Candidate-authored kernels (Triton / custom CUDA from load_inline)
# have clean names; framework ones start with `void ` and/or carry a C++
# namespace marker. We use this to (a) pick what ncu should replay and (b) trim
# the nsys display list afterward.
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


def _candidate_kernel_names(kernels, *, time_floor: float = 1.0, keep_top_n: int = 3) -> list[str]:
    """Names of candidate-authored / dominant kernels, derived from the nsys ranking.

    Keep a kernel if it is not framework, or is above a GPU-time floor, or is
    within the top-N -- the ``keep_top_n`` floor guarantees >=1 name even when a
    candidate legitimately calls a library GEMM as its main op. The list is
    already sorted desc by ``time_pct``; a missing ``time_pct`` counts as 0.0.
    """
    names = []
    for i, k in enumerate(kernels or []):
        tp = k.get("time_pct")
        tp = tp if isinstance(tp, (int, float)) else 0.0
        name = k.get("name")
        if name and ((not _is_framework_kernel(name)) or tp >= time_floor or i < keep_top_n):
            names.append(name)
    return names


def _filter_noise(result: dict, *, time_floor: float = 1.0, keep_top_n: int = 3) -> None:
    """Mutate ``result`` in place: trim the nsys display list + reconcile ncu_kernels.

    Filters ``result["kernels"]`` by framework classification / time floor / top-N,
    reconciles ``result["ncu_kernels"]`` to the kept names (keeping an ncu entry if
    its name is kept **or** it is not framework -- guards nsys/ncu name-mangling
    mismatches), and records ``result["noise_kernels_dropped"]`` (no silent
    truncation).
    """
    kernels = result.get("kernels") or []
    kept = []
    for i, k in enumerate(kernels):
        tp = k.get("time_pct")
        tp = tp if isinstance(tp, (int, float)) else 0.0
        name = k.get("name")
        if (not _is_framework_kernel(name)) or tp >= time_floor or i < keep_top_n:
            kept.append(k)
    result["noise_kernels_dropped"] = len(kernels) - len(kept)
    result["kernels"] = kept

    kept_names = {k.get("name") for k in kept}
    ncu_kernels = result.get("ncu_kernels")
    if ncu_kernels:
        result["ncu_kernels"] = [
            k for k in ncu_kernels
            if k.get("name") in kept_names or not _is_framework_kernel(k.get("name"))
        ]


def _runner_cmd(kernel_path: str, run_args: dict, num_iters: int) -> list[str]:
    """Build the ``python tool_scripts/nsight_runner.py ...`` argv shared by both passes."""
    cmd = [
        sys.executable,
        _RUNNER,
        "--kernel_path", kernel_path,
        "--test_source", str(run_args.get("test_source", "KB")),
        "--level", str(run_args.get("level")),
        "--problem_id", str(run_args.get("problem_id")),
        "--device", str(run_args.get("device", 0)),
        "--dtype", str(run_args.get("dtype_str", "fp32")),
        "--backend", str(run_args.get("backend", "triton")),
        "--num_iters", str(num_iters),
    ]
    if run_args.get("suffix"):
        cmd += ["--suffix", str(run_args["suffix"])]
    return cmd


# --------------------------------------------------------------------------- #
# nsys path
# --------------------------------------------------------------------------- #
def _parse_nsys_report(report_path: str, timeout: int) -> dict:
    """Run ``nsys stats`` and parse the kernel + memcpy summaries into a dict."""
    proc = subprocess.run(
        [
            _NSYS, "stats",
            "--report", "cuda_gpu_kern_sum",
            "--report", "cuda_gpu_mem_time_sum",
            "--format", "json",
            report_path,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return {"nsys_error": f"nsys stats failed: {proc.stderr.strip()[:500]}"}

    # `nsys stats` with multiple --report and --format json emits one JSON array
    # per report, concatenated. Split on the array boundaries and parse each.
    blocks = _split_json_arrays(proc.stdout)
    kernels: list[dict] = []
    memops: list[dict] = []
    for block in blocks:
        if not block:
            continue
        # Heuristic: kernel report rows have a "Name" with a grid/kernel; mem
        # report rows have an "Operation" key. Sort by the columns present.
        sample = block[0]
        keys = {k.lower() for k in sample.keys()}
        if "operation" in keys:
            memops.extend(block)
        else:
            kernels.extend(block)

    parsed_kernels = [_normalize_kernel_row(r) for r in kernels]
    parsed_mem = [_normalize_mem_row(r) for r in memops]
    # Highest GPU-time share first -- the candidate's dominant kernel leads.
    parsed_kernels.sort(key=lambda k: k.get("time_pct", 0.0), reverse=True)
    return {"kernels": parsed_kernels, "memory_ops": parsed_mem}


def _split_json_arrays(text: str) -> list[list]:
    """Parse a stream that may contain several concatenated top-level JSON arrays.

    ``nsys stats`` interleaves non-JSON status lines ("Generating SQLite file...",
    "Processing [...]") before/between each report's JSON array, so we scan for
    ``[`` boundaries and attempt to decode an array at each one.
    """
    arrays: list[list] = []
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        # Advance to the next array start; bail if there are none left.
        start = text.find("[", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            # Not a valid array here (e.g. a stray '[' in a status line) -- skip it.
            idx = start + 1
            continue
        if isinstance(obj, list):
            arrays.append(obj)
        idx = end
    return arrays


def _num(row: dict, *names):
    """Fetch the first present key (case-insensitive) and coerce to float."""
    lowered = {k.lower(): v for k, v in row.items()}
    for name in names:
        v = lowered.get(name.lower())
        if v is None:
            continue
        try:
            return float(str(v).replace(",", ""))
        except (TypeError, ValueError):
            return v
    return None


def _str(row: dict, *names):
    lowered = {k.lower(): v for k, v in row.items()}
    for name in names:
        v = lowered.get(name.lower())
        if v is not None:
            return str(v)
    return None


def _normalize_kernel_row(row: dict) -> dict:
    """Map an nsys cuda_gpu_kern_sum row to stable field names (ns -> ms)."""
    total_ns = _num(row, "Total Time (ns)", "Total Time")
    avg_ns = _num(row, "Avg (ns)", "Average (ns)", "Avg")
    out = {
        "name": _str(row, "Name", "Kernel Name"),
        "instances": _num(row, "Instances", "Count", "Num Calls"),
        "time_pct": _num(row, "Time (%)", "Time(%)", "% Time"),
        "total_ms": (total_ns / 1e6) if isinstance(total_ns, (int, float)) else None,
        "avg_ms": (avg_ns / 1e6) if isinstance(avg_ns, (int, float)) else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def _normalize_mem_row(row: dict) -> dict:
    total_ns = _num(row, "Total Time (ns)", "Total Time")
    out = {
        "operation": _str(row, "Operation", "Name"),
        "time_pct": _num(row, "Time (%)", "Time(%)", "% Time"),
        "total_ms": (total_ns / 1e6) if isinstance(total_ns, (int, float)) else None,
        "count": _num(row, "Count", "Instances", "Num Calls"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _run_nsys(kernel_path: str, run_args: dict, num_iters: int, timeout: int) -> dict:
    """Run the nsys profiling pass and return parsed kernel/memcpy data."""
    with tempfile.TemporaryDirectory(prefix="nsight_nsys_") as tmpdir:
        out_base = os.path.join(tmpdir, "profile")
        cmd = [
            _NSYS, "profile",
            "-o", out_base,
            "--force-overwrite", "true",
        ] + _runner_cmd(kernel_path, run_args, num_iters)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        report_path = out_base + ".nsys-rep"
        if not os.path.exists(report_path):
            # Some nsys versions emit .qdrep
            alt = out_base + ".qdrep"
            if os.path.exists(alt):
                report_path = alt
            else:
                return {
                    "nsys_error": (
                        "nsys produced no report. "
                        f"rc={proc.returncode} stderr={proc.stderr.strip()[:500]}"
                    )
                }
        return _parse_nsys_report(report_path, timeout=timeout)


# --------------------------------------------------------------------------- #
# ncu path (default stage, needs root)
# --------------------------------------------------------------------------- #
# ncu --csv emits one row per (kernel, metric) with columns including
# "Kernel Name", "Metric Name", "Metric Value". We pivot the metrics we care
# about per kernel.
#
# With explicit `--metrics`, ncu's "Metric Name" column prints the *internal*
# metric ids (not the human display names `--set basic` showed), so the keys
# below are those ids. The mapped value names stay stable so `_roofline_bound`
# and the evaluator prompt keep working. `gpu__compute_memory_throughput` is the
# GPU Speed-Of-Light memory number (max over the memory subsystems);
# `dram__throughput` is the DRAM-only sub-metric -- we keep both so the roofline
# classification isn't fooled by a kernel that saturates L1/L2/shared while DRAM
# stays idle. Targeting only the dominant candidate kernel(s) via `-k` is what
# makes this larger bundle affordable. NB: the `launch__*` ids live in ncu's
# `launch` metric collection; they coexist fine in one `--metrics` request.
_NCU_METRICS = {
    # throughput / roofline (kept -- _roofline_bound depends on these three)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":                 "compute_throughput_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":               "dram_throughput_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed":                "l2_throughput_pct",
    "gpu__time_duration.sum":                                           "duration",
    # occupancy + WHY it is capped (new -- answers the run_2_15 gap)
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
    # dram traffic (new -- feeds arithmetic-intensity / roofline)
    "dram__bytes_read.sum":                                             "dram_bytes_read",
    "dram__bytes_write.sum":                                            "dram_bytes_write",
}

# ncu's section rule engine emits per-kernel, roofline-aware verdicts (OPT/WRN
# text + estimated speedup). These four reliably produce rules without needing
# source-level PC sampling. Adding sections multiplies ncu replay passes, so this
# list is the single cost/timeout knob (drop SchedulerStats first if needed).
_NCU_SECTIONS = ("SpeedOfLight", "Occupancy", "MemoryWorkloadAnalysis", "SchedulerStats")


def _run_ncu(
    kernel_path: str, run_args: dict, ncu_sudo: str, timeout: int,
    candidate_names: list = None,
) -> dict:
    """Run the ncu hardware-counter pass (single iteration), targeting candidates. Needs root.

    When ``candidate_names`` is non-empty, restrict ncu to those kernels via a
    demangled-name regex so its expensive replay never touches framework noise.
    If the targeted run yields no kernels (regex matched nothing), retry once
    with the same ``--metrics`` but no ``-k`` so we never produce an empty profile.
    """
    metrics_arg = ",".join(_NCU_METRICS.keys())

    def _build_cmd(use_filter: bool) -> list:
        base = [
            _NCU,
            "--metrics", metrics_arg,
            "--csv",
            "--target-processes", "all",
        ]
        for section in _NCU_SECTIONS:
            base += ["--section", section]
        if use_filter and candidate_names:
            # The nsys "Name" is demangled, so the regex must match the demangled form.
            base += [
                "--kernel-name-base", "demangled",
                "-k", "regex:" + "|".join(re.escape(n) for n in candidate_names),
            ]
        base += _runner_cmd(kernel_path, run_args, num_iters=1)
        return base

    is_root = hasattr(os, "geteuid") and os.geteuid() == 0

    def _invoke(use_filter: bool):
        base_cmd = _build_cmd(use_filter)
        if is_root:
            cmd = base_cmd
            stdin_input = None
        else:
            # sudo -S reads the password from stdin; -p "" suppresses the prompt.
            cmd = ["sudo", "-S", "-p", ""] + base_cmd
            stdin_input = (ncu_sudo or "") + "\n"
        return subprocess.run(
            cmd,
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    proc = _invoke(use_filter=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        lowered = msg.lower()
        if "password" in lowered or "sudo" in lowered or "permission" in lowered or "ERR_NVGPUCTRPERM" in msg:
            return {"ncu_error": f"ncu permission/sudo failure: {msg[:400]}"}
        return {"ncu_error": f"ncu failed (rc={proc.returncode}): {msg[:400]}"}

    parsed = _parse_ncu_csv(proc.stdout)
    # Fallback: targeted run matched nothing -> retry once unfiltered (same
    # metrics keep parsing consistent; Stage 3 reconcile trims framework noise).
    if candidate_names and not parsed.get("ncu_kernels"):
        retry = _invoke(use_filter=False)
        if retry.returncode == 0:
            parsed = _parse_ncu_csv(retry.stdout)
    return parsed


def _parse_ncu_csv(csv_text: str) -> dict:
    """Pivot ncu --csv (long format) into per-kernel metric dicts."""
    # ncu prints non-CSV preamble lines (e.g. "==PROF== ...") before the header;
    # locate the header row that contains the metric columns.
    lines = csv_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if '"Kernel Name"' in line or "Kernel Name" in line and "Metric Name" in line:
            header_idx = i
            break
    if header_idx is None:
        return {"ncu_error": "ncu CSV header not found in output"}

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    per_kernel: dict[str, dict] = {}
    order: list[str] = []
    for row in reader:
        kname = row.get("Kernel Name") or row.get("Demangled Name")
        metric = row.get("Metric Name")
        value = row.get("Metric Value")
        if not kname:
            continue
        if kname not in per_kernel:
            per_kernel[kname] = {"name": kname}
            order.append(kname)
        if metric:
            # Metric row: pivot the long-format metric into a stable key.
            mapped = _NCU_METRICS.get(metric)
            if mapped:
                try:
                    per_kernel[kname][mapped] = float(str(value).replace(",", ""))
                except (TypeError, ValueError):
                    per_kernel[kname][mapped] = value
            continue
        # Rule row: blank Metric Name but a populated rule description. These are
        # ncu's own roofline-aware expert verdicts. Drop INF (informational) rows
        # to cut noise; keep OPT (optimization) + WRN (warning).
        rule_desc = row.get("Rule Description")
        rule_name = row.get("Rule Name")
        rule_type = row.get("Rule Type")
        if not (rule_desc or rule_name):
            continue
        if rule_type == "INF":
            continue
        speedup_pct = None
        try:
            raw_speedup = row.get("Estimated Speedup")
            if raw_speedup not in (None, ""):
                speedup_pct = float(str(raw_speedup).replace(",", ""))
        except (TypeError, ValueError):
            speedup_pct = None
        per_kernel[kname].setdefault("rules", []).append({
            "section": row.get("Section Name") or None,
            "type": rule_type or None,
            "name": rule_name or None,
            "desc": rule_desc or None,
            "speedup_pct": speedup_pct,
        })

    kernels = [per_kernel[k] for k in order]
    for k in kernels:
        k["roofline_bound"] = _roofline_bound(k)
    return {"ncu_kernels": kernels}


def _roofline_bound(k: dict) -> str:
    """Classify a kernel as memory- or compute-bound from throughput percentages."""
    # Use the SOL memory throughput (max over the memory subsystems) as the
    # memory side, falling back to DRAM-only when it is the sole signal -- a
    # kernel can saturate L1/L2/shared while DRAM stays near idle, and the SOL
    # number is what ncu's own roofline guidance keys on.
    mem = k.get("memory_throughput_pct")
    if not isinstance(mem, (int, float)):
        mem = k.get("dram_throughput_pct")
    comp = k.get("compute_throughput_pct")
    if not isinstance(mem, (int, float)) or not isinstance(comp, (int, float)):
        return "unknown"
    if max(mem, comp) < 40:
        return "latency_bound"
    return "memory_bound" if mem >= comp else "compute_bound"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def profile_kernel(
    kernel_path: str,
    *,
    run_args: dict,
    device=0,
    ncu_sudo: str = "",
    num_iters: int = 5,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """Profile the candidate kernel and return a JSON-serializable summary dict.

    Args:
        kernel_path: Path to the candidate .py file (defines ``ModelNew``).
        run_args: dict carrying ``test_source``, ``level``, ``problem_id``,
            ``backend``, ``dtype_str`` (and optional ``suffix``) -- the same
            values eval used so the profiled path matches the measured path.
        device: CUDA device index.
        ncu_sudo: sudo password fed to ``sudo -S`` for the (root-only) ncu pass.
        num_iters: forward iterations for the nsys pass (kept small).
        timeout: per-subprocess timeout in seconds.

    Returns:
        A dict describing only the currently executed kernel. Always best-effort:
        any failure surfaces as ``error`` / ``nsys_error`` / ``ncu_error`` rather
        than raising. ncu is a default stage (no toggle): it always runs when the
        binary exists, targeting the dominant candidate kernel(s) from the nsys
        ranking.
    """
    run_args = dict(run_args or {})
    run_args.setdefault("device", device)
    result: dict = {}
    try:
        if not os.path.exists(kernel_path):
            return {"error": f"kernel_path does not exist: {kernel_path}"}
        if not os.path.exists(_NSYS):
            return {"error": f"nsys not found at {_NSYS}"}

        result.update(_run_nsys(kernel_path, run_args, num_iters, timeout))
        candidate_names = _candidate_kernel_names(result.get("kernels") or [])

        # ncu is a default stage -- always attempt it (guarded by _NCU existence + try/except).
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
    except subprocess.TimeoutExpired:
        result.setdefault("error", f"nsys timed out after {timeout}s")
    except Exception as e:  # noqa: BLE001 -- never break eval
        result.setdefault("error", f"profile_kernel exception: {e}")
    return result


def format_nsight_summary(nsight: dict, *, indent: int = 2) -> str:
    """Render the nsight dict for the evaluator prompt (plain JSON dump)."""
    try:
        return json.dumps(nsight, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(nsight)
