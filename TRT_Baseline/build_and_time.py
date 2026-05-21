#!/usr/bin/env python3
"""Stage 3: build TRT engines via `trtexec`, do a Python-runtime numeric
parity check against torch eager, and aggregate per-file timings into
results/timing/<hw>/baseline_time_trt.json.

Reads TRT_Baseline/config.yaml and TRT_Baseline/convertONNX/manifest.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
TRTEXEC = "/opt/tensorrt/bin/trtexec"


# --------------------------------------------------------------------------- #
# Config / filter / module-loading helpers (mirror Stage 2)
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_filter(s: str | None) -> tuple[int, str] | None:
    if not s:
        return None
    s = s.strip().strip("/")
    parts = s.split("/")
    if len(parts) != 2 or not parts[0].startswith("level"):
        raise ValueError(f"--filter must look like 'level2/1', got {s!r}")
    return int(parts[0][len("level"):]), parts[1]


def matches_filter(level: int, key_file: str, flt) -> bool:
    if flt is None:
        return True
    flt_level, flt_id = flt
    if level != flt_level:
        return False
    stem = Path(key_file).stem
    return stem.startswith(f"{flt_id}_") or stem == flt_id


def import_kb_module(py_path: Path):
    spec = importlib.util.spec_from_file_location(f"kb_{py_path.stem}", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# trtexec subprocess
# --------------------------------------------------------------------------- #
def resolve_workspace_mib(cfg: dict) -> int:
    cap = int(cfg["workspace"]["cap_bytes"])
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        free = cap
    use = min(free, cap)
    # Keep a small headroom so the rest of the process (driver, our buffers) fits.
    use = max(use - (256 << 20), cap >> 1)
    return max(use // (1 << 20), 256)


def build_trtexec_cmd(onnx_path: Path, plan_path: Path, times_path: Path,
                       cfg: dict) -> list[str]:
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
    ]
    if not cfg.get("tf32", False):
        cmd.append("--noTF32")

    cmd.append(f"--memPoolSize=workspace:{resolve_workspace_mib(cfg)}")
    cmd.append(f"--builderOptimizationLevel={int(cfg['builder']['optimization_level'])}")

    tactics = cfg["builder"].get("tactic_sources") or []
    if tactics:
        cmd.append("--tacticSources=" + ",".join(f"+{t}" for t in tactics))

    cmd.append(f"--warmUp={int(cfg['timing']['warmup_ms'])}")
    cmd.append(f"--iterations={int(cfg['timing']['trials'])}")
    cmd.append("--avgRuns=1")

    use_graph = cfg["timing"].get("use_cuda_graphs", False)
    if use_graph is True or (isinstance(use_graph, str) and use_graph.lower() == "auto"
                              and cfg["shapes"]["mode"] == "static"):
        cmd.append("--useCudaGraph")

    cmd.append("--useSpinWait")
    cmd.append(f"--exportTimes={times_path}")
    return cmd


def run_trtexec(cmd: list[str], log_path: Path, timeout_s: int = 1800):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    log_path.write_text("CMD: " + " ".join(cmd) + "\n\n"
                        + "===== STDOUT =====\n" + (proc.stdout or "")
                        + "\n===== STDERR =====\n" + (proc.stderr or ""))
    return proc


def parse_export_times(times_path: Path) -> list[float]:
    """trtexec --exportTimes emits a JSON list of {latencyMs, computeMs, ...}."""
    data = json.loads(times_path.read_text())
    if isinstance(data, dict):
        # Some trtexec versions wrap the list under a key.
        for k in ("trace", "iterations", "data"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    out = []
    for entry in data:
        if isinstance(entry, dict):
            if "latencyMs" in entry:
                out.append(float(entry["latencyMs"]))
            elif "latency" in entry:
                out.append(float(entry["latency"]))
    return out


# --------------------------------------------------------------------------- #
# Python-runtime numeric parity
# --------------------------------------------------------------------------- #
def torch_dtype_from_trt(trt, ttype):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.UINT8: torch.uint8,
        trt.DataType.INT64: torch.int64,
    }.get(ttype, torch.float32)


def numeric_check(plan_path: Path, py_path: Path, cfg: dict) -> tuple[bool, str]:
    """Return (ok, message)."""
    import tensorrt as trt
    seed = int(cfg["onnx"]["random_seed"])
    atol = float(cfg["tolerance"]["atol"])
    rtol = float(cfg["tolerance"]["rtol"])

    # 1) torch reference (same construct path as Stage 2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    M = import_kb_module(py_path)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        model = M.Model(*M.get_init_inputs()).cuda().eval()
        inputs = [t.cuda().contiguous() if isinstance(t, torch.Tensor) else t
                  for t in M.get_inputs()]
        ref = model(*inputs)
    ref_outs = ref if isinstance(ref, (list, tuple)) else (ref,)

    # 2) TRT runtime
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        return False, "deserialize_cuda_engine returned None"
    ctx = engine.create_execution_context()

    # Map IO names to torch buffers. Inputs come from M.get_inputs() in order.
    n_io = engine.num_io_tensors
    io_names = [engine.get_tensor_name(i) for i in range(n_io)]
    is_input = [engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                for name in io_names]

    input_names = [n for n, inp in zip(io_names, is_input) if inp]
    output_names = [n for n, inp in zip(io_names, is_input) if not inp]

    if len(input_names) != len(inputs):
        return False, (f"engine input count {len(input_names)} != torch inputs "
                       f"{len(inputs)}")

    stream = torch.cuda.Stream()
    out_bufs = []
    with torch.cuda.stream(stream):
        for name, t in zip(input_names, inputs):
            if not isinstance(t, torch.Tensor):
                return False, f"input {name} is not a tensor"
            t = t.contiguous()
            shp = tuple(t.shape)
            try:
                ctx.set_input_shape(name, shp)
            except Exception:
                pass
            ctx.set_tensor_address(name, int(t.data_ptr()))

        for name in output_names:
            shp = tuple(ctx.get_tensor_shape(name))
            if any(d < 0 for d in shp):
                return False, f"output {name} has dynamic shape {shp}"
            dtype = torch_dtype_from_trt(trt, engine.get_tensor_dtype(name))
            buf = torch.empty(shp, dtype=dtype, device="cuda").contiguous()
            out_bufs.append((name, buf))
            ctx.set_tensor_address(name, int(buf.data_ptr()))

        ok = ctx.execute_async_v3(stream.cuda_stream)
        if not ok:
            return False, "execute_async_v3 returned False"
    stream.synchronize()

    if len(out_bufs) != len(ref_outs):
        return False, (f"engine output count {len(out_bufs)} != torch outputs "
                       f"{len(ref_outs)}")

    for (name, trt_out), ref_t in zip(out_bufs, ref_outs):
        if trt_out.shape != ref_t.shape:
            return False, (f"output {name} shape {tuple(trt_out.shape)} != torch "
                           f"{tuple(ref_t.shape)}")
        a = trt_out.to(torch.float32)
        b = ref_t.to(torch.float32)
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a - b).abs().max().item()
            return False, f"output {name} max|diff|={diff:.4g} exceeds tolerance"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def stats_from_latencies(lat_ms: list[float]) -> dict:
    if not lat_ms:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "num_trials": 0}
    return {
        "mean": float(statistics.fmean(lat_ms)),
        "std": float(statistics.pstdev(lat_ms)) if len(lat_ms) > 1 else 0.0,
        "min": float(min(lat_ms)),
        "max": float(max(lat_ms)),
        "num_trials": len(lat_ms),
    }


def load_existing_results(out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--filter", default=None,
                    help="Restrict to a single file like 'level2/1'")
    ap.add_argument("--manifest", default=None, type=Path,
                    help="Override manifest path (default: convertONNX/manifest.json)")
    ap.add_argument("--skip-numeric", action="store_true",
                    help="Skip Python-runtime numeric parity check")
    args = ap.parse_args()

    if not Path(TRTEXEC).is_file():
        print(f"[error] trtexec not found at {TRTEXEC}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    flt = parse_filter(args.filter)
    manifest_path = args.manifest or (THIS_DIR / "convertONNX" / "manifest.json")
    if not manifest_path.exists():
        print(f"[error] manifest missing: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())

    out_path = (REPO_ROOT / cfg["output_path"]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_existing_results(out_path)

    hardware_label = cfg["hardware"]
    device_label = cfg["device"]
    try:
        hw_full = torch.cuda.get_device_name(0)
    except Exception:
        hw_full = hardware_label
    workspace_bytes = int(cfg["workspace"]["cap_bytes"])
    tf32 = bool(cfg.get("tf32", False))

    total = ok_count = build_fail = numeric_fail = export_fail = 0
    for key, entry in sorted(manifest.items()):
        # key like "level2/1_Conv2D_ReLU_BiasAdd.py"
        try:
            level_str, fname = key.split("/", 1)
            level = int(level_str[len("level"):])
        except Exception:
            continue
        if not matches_filter(level, fname, flt):
            continue
        total += 1
        level_key = f"level{level}"
        results.setdefault(level_key, {})

        common = {
            "hardware": hw_full,
            "device": device_label,
            "engine": "trt-10.x",
            "path": "convertONNX",
            "tf32": tf32,
            "workspace_bytes": workspace_bytes,
        }

        if entry.get("status") != "ok":
            results[level_key][fname] = {
                **{"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "num_trials": 0},
                **common,
                "status": "onnx_export_failed",
            }
            export_fail += 1
            print(f"[{key}] skip: onnx_export_failed")
            out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
            continue

        onnx_path = (REPO_ROOT / entry["onnx_path"]).resolve()
        stem = onnx_path.stem
        eng_dir = THIS_DIR / "engines" / level_key
        eng_dir.mkdir(parents=True, exist_ok=True)
        plan_path = eng_dir / f"{stem}.plan"
        times_path = eng_dir / f"{stem}.times.json"
        log_path = eng_dir / f"{stem}.trtexec.log"

        # Always rebuild — engine_cache: false.
        if plan_path.exists():
            plan_path.unlink()
        if times_path.exists():
            times_path.unlink()

        cmd = build_trtexec_cmd(onnx_path, plan_path, times_path, cfg)
        print(f"[{key}] trtexec build+time ...", flush=True)
        try:
            proc = run_trtexec(cmd, log_path)
        except subprocess.TimeoutExpired:
            results[level_key][fname] = {
                **{"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "num_trials": 0},
                **common,
                "status": "trt_build_failed",
                "error": "trtexec timeout",
            }
            build_fail += 1
            print(f"  -> trt_build_failed (timeout)")
            out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
            continue

        if proc.returncode != 0 or not plan_path.is_file() or not times_path.is_file():
            err_tail = "\n".join((proc.stderr or "").splitlines()[-40:])
            results[level_key][fname] = {
                **{"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "num_trials": 0},
                **common,
                "status": "trt_build_failed",
                "error": f"rc={proc.returncode}; tail: {err_tail[-400:]}",
            }
            entry_err = entry.setdefault("trt_error", err_tail[-1000:])
            build_fail += 1
            print(f"  -> trt_build_failed (rc={proc.returncode})")
            out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
            continue

        # Numeric parity (best-effort; mismatch records but engine + times kept).
        numeric_status = "ok"
        numeric_msg = "skipped" if args.skip_numeric else "ok"
        if not args.skip_numeric:
            try:
                py_path = (REPO_ROOT / entry["source_py"]).resolve()
                ok_n, msg = numeric_check(plan_path, py_path, cfg)
                if not ok_n:
                    numeric_status = "numeric_mismatch"
                    numeric_msg = msg
            except Exception as e:
                numeric_status = "numeric_mismatch"
                numeric_msg = f"{type(e).__name__}: {e}"
                traceback.print_exc(limit=4)
            finally:
                torch.cuda.empty_cache()

        latencies = parse_export_times(times_path)
        s = stats_from_latencies(latencies)
        record = {**s, **common}
        if numeric_status == "ok":
            record["status"] = "ok"
            ok_count += 1
            print(f"  -> ok  mean={s['mean']:.4f}ms n={s['num_trials']}")
        else:
            record["status"] = "numeric_mismatch"
            record["numeric_error"] = numeric_msg
            numeric_fail += 1
            print(f"  -> numeric_mismatch ({numeric_msg})")
        results[level_key][fname] = record
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    print(f"\n[build_and_time] total={total} ok={ok_count} "
          f"export_fail={export_fail} build_fail={build_fail} "
          f"numeric_fail={numeric_fail}")
    print(f"[build_and_time] output: {out_path}")


if __name__ == "__main__":
    main()
