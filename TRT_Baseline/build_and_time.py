#!/usr/bin/env python3
"""Stage 3: build TRT engines via `trtexec`, do a Python-runtime numeric
parity check against torch eager, and aggregate per-file timings into
results/timing/<hw>/baseline_time_trt.json.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent

TRTEXEC = "/opt/tensorrt/bin/trtexec"
if not Path(TRTEXEC).is_file():
    TRTEXEC = shutil.which("trtexec") or TRTEXEC

ZERO_STATS = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "num_trials": 0}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_filter(s):
    if not s:
        return None
    level_str, ident = s.strip("/").split("/")
    return int(level_str.removeprefix("level")), ident


def matches_filter(level, fname, flt):
    if flt is None:
        return True
    flt_level, flt_id = flt
    stem = Path(fname).stem
    return level == flt_level and (stem == flt_id or stem.startswith(f"{flt_id}_"))


def import_kb_module(py_path: Path):
    spec = importlib.util.spec_from_file_location(f"kb_{py_path.stem}", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def free_gpu():
    # Drop stale kb_* imports + release torch's caching allocator. Without this
    # the next trtexec subprocess sees shrinking VRAM and OOMs after a few files.
    for n in [n for n in sys.modules if n.startswith("kb_")]:
        sys.modules.pop(n, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# --------------------------------------------------------------------------- #
# trtexec
# --------------------------------------------------------------------------- #
def workspace_mib(cfg) -> int:
    """min(free, cap) - 1 GiB headroom. Old version's `cap // 2` floor caused
    OOM on a fragmented 24 GiB card."""
    cap = int(cfg["workspace"]["cap_bytes"])
    free, _ = torch.cuda.mem_get_info()
    return max((min(free, cap) - (1 << 30)) // (1 << 20), 1024)


def trtexec_cmd(onnx, plan, times, cfg):
    cmd = [TRTEXEC, f"--onnx={onnx}", f"--saveEngine={plan}",
           f"--memPoolSize=workspace:{workspace_mib(cfg)}",
           f"--builderOptimizationLevel={cfg['builder']['optimization_level']}",
           f"--warmUp={cfg['timing']['warmup_ms']}",
           f"--iterations={cfg['timing']['trials']}",
           f"--exportTimes={times}",
           "--avgRuns=1", "--useSpinWait"]
    if not cfg.get("tf32", False):
        cmd.append("--noTF32")
    if cfg["builder"].get("tactic_sources"):
        cmd.append("--tacticSources=" + ",".join(
            f"+{t}" for t in cfg["builder"]["tactic_sources"]))
    use_graph = cfg["timing"].get("use_cuda_graphs", False)
    if use_graph is True or (str(use_graph).lower() == "auto"
                             and cfg["shapes"]["mode"] == "static"):
        cmd.append("--useCudaGraph")
    return cmd


def parse_export_times(path: Path) -> list[float]:
    """Read kernel-only times (`computeMs`) from a trtexec --exportTimes JSON.

    `latencyMs` would include H2D + compute + D2H, but the AdaExplore torch
    baseline times only the model forward pass on already-on-GPU tensors,
    so we match that contract by reporting `computeMs`.
    """
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for k in ("trace", "iterations", "data"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
    return [float(e.get("computeMs", e.get("compute", 0.0))) for e in data]


def stats(lat_ms):
    if not lat_ms:
        return dict(ZERO_STATS)
    return {
        "mean": float(statistics.fmean(lat_ms)),
        "std": float(statistics.pstdev(lat_ms)) if len(lat_ms) > 1 else 0.0,
        "min": float(min(lat_ms)),
        "max": float(max(lat_ms)),
        "num_trials": len(lat_ms),
    }


# --------------------------------------------------------------------------- #
# Numeric parity (best-effort)
# --------------------------------------------------------------------------- #
TRT_TO_TORCH = {
    "FLOAT": torch.float32, "HALF": torch.float16, "INT8": torch.int8,
    "INT32": torch.int32, "BOOL": torch.bool, "UINT8": torch.uint8,
    "INT64": torch.int64,
}


def numeric_check(plan_path: Path, py_path: Path, cfg) -> tuple[bool, str]:
    import tensorrt as trt
    seed = cfg["onnx"]["random_seed"]
    atol = cfg["tolerance"]["atol"]
    rtol = cfg["tolerance"]["rtol"]

    # 1) torch ref pass: produce inputs (kept on GPU as bytes) and CPU ref outs,
    #    then drop the torch model + activations before allocating TRT bufs.
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    M = import_kb_module(py_path)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    try:
        with torch.no_grad():
            model = M.Model(*M.get_init_inputs()).cuda().eval()
            inputs_gpu = [t.cuda().contiguous() for t in M.get_inputs()]
            ref = model(*inputs_gpu)
        ref_outs_cpu = [t.detach().float().cpu()
                        for t in (ref if isinstance(ref, (list, tuple)) else (ref,))]
        # Keep raw input bytes on GPU but drop the autograd-free torch graph.
        input_bytes = [t.detach().clone() for t in inputs_gpu]
    except Exception as e:
        free_gpu()
        return False, f"torch_ref_failed: {e}"
    del model, inputs_gpu, ref
    free_gpu()

    # 2) TRT pass.
    out_cpu: list = []
    try:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
        ctx = engine.create_execution_context()

        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        ins = [n for n in names
               if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outs = [n for n in names
                if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        for n, t in zip(ins, input_bytes):
            ctx.set_tensor_address(n, int(t.data_ptr()))
        out_bufs = []
        for n in outs:
            shp = tuple(ctx.get_tensor_shape(n))
            dtype = TRT_TO_TORCH.get(engine.get_tensor_dtype(n).name, torch.float32)
            buf = torch.empty(shp, dtype=dtype, device="cuda")
            out_bufs.append(buf)
            ctx.set_tensor_address(n, int(buf.data_ptr()))

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        out_cpu = [b.detach().float().cpu() for b in out_bufs]
        del out_bufs, ctx, engine, runtime, input_bytes
    finally:
        free_gpu()

    # 3) Compare on CPU — avoids extra GPU allocations the size of outputs.
    for trt_out, ref_t in zip(out_cpu, ref_outs_cpu):
        if trt_out.shape != ref_t.shape:
            return False, f"shape mismatch trt={tuple(trt_out.shape)} ref={tuple(ref_t.shape)}"
        if not torch.allclose(trt_out, ref_t, atol=atol, rtol=rtol):
            return False, f"max|diff|={(trt_out - ref_t).abs().max().item():.4g}"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--manifest", type=Path,
                    default=THIS_DIR / "convertONNX" / "manifest.json")
    ap.add_argument("--skip-numeric", action="store_true")
    args = ap.parse_args()

    if not Path(TRTEXEC).is_file():
        sys.exit(f"[error] trtexec not found at {TRTEXEC}")

    cfg = yaml.safe_load(args.config.read_text())
    flt = parse_filter(args.filter)
    manifest = json.loads(args.manifest.read_text())

    out_path = (REPO_ROOT / cfg["output_path"]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    common = {
        "hardware": torch.cuda.get_device_name(0),
        "device": cfg["device"],
        "engine": "trt-10.x",
        "path": "convertONNX",
        "tf32": bool(cfg.get("tf32", False)),
        "workspace_bytes": int(cfg["workspace"]["cap_bytes"]),
    }

    def write(level_key, fname, status, **extra):
        results.setdefault(level_key, {})[fname] = {
            **ZERO_STATS, **common, **extra, "status": status,
        }
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    counts = {"total": 0, "ok": 0, "build_fail": 0, "numeric_fail": 0,
              "export_fail": 0}

    for key, entry in sorted(manifest.items()):
        level_str, fname = key.split("/", 1)
        level = int(level_str.removeprefix("level"))
        if not matches_filter(level, fname, flt):
            continue
        counts["total"] += 1
        level_key = f"level{level}"

        if entry.get("status") != "ok":
            write(level_key, fname, "onnx_export_failed")
            counts["export_fail"] += 1
            print(f"[{key}] skip: onnx_export_failed")
            continue

        onnx_path = (REPO_ROOT / entry["onnx_path"]).resolve()
        eng_dir = THIS_DIR / "engines" / level_key
        eng_dir.mkdir(parents=True, exist_ok=True)
        plan_path = eng_dir / f"{onnx_path.stem}.plan"
        times_path = eng_dir / f"{onnx_path.stem}.times.json"
        log_path = eng_dir / f"{onnx_path.stem}.trtexec.log"
        plan_path.unlink(missing_ok=True)
        times_path.unlink(missing_ok=True)

        free_gpu()
        cmd = trtexec_cmd(onnx_path, plan_path, times_path, cfg)
        print(f"[{key}] trtexec build+time ...", flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            write(level_key, fname, "trt_build_failed", error="trtexec timeout")
            counts["build_fail"] += 1
            print("  -> trt_build_failed (timeout)")
            continue
        log_path.write_text("CMD: " + " ".join(cmd) + "\n\n=== STDOUT ===\n"
                             + proc.stdout + "\n=== STDERR ===\n" + proc.stderr)

        if proc.returncode != 0 or not plan_path.is_file() or not times_path.is_file():
            err_tail = "\n".join(proc.stderr.splitlines()[-40:])
            write(level_key, fname, "trt_build_failed",
                  error=f"rc={proc.returncode}; tail: {err_tail[-400:]}")
            counts["build_fail"] += 1
            print(f"  -> trt_build_failed (rc={proc.returncode})")
            continue

        numeric_status, numeric_msg = "ok", "skipped" if args.skip_numeric else "ok"
        if not args.skip_numeric:
            py_path = (REPO_ROOT / entry["source_py"]).resolve()
            try:
                ok_n, msg = numeric_check(plan_path, py_path, cfg)
            except Exception as e:
                ok_n, msg = False, f"numeric_check_exception: {type(e).__name__}: {e}"
                free_gpu()
            if not ok_n:
                numeric_status, numeric_msg = "numeric_mismatch", msg

        s = stats(parse_export_times(times_path))
        if numeric_status == "ok":
            write(level_key, fname, "ok", **s)
            counts["ok"] += 1
            print(f"  -> ok  mean={s['mean']:.4f}ms n={s['num_trials']}")
        else:
            write(level_key, fname, "numeric_mismatch",
                  numeric_error=numeric_msg, **s)
            counts["numeric_fail"] += 1
            print(f"  -> numeric_mismatch ({numeric_msg})")

    print(f"\n[build_and_time] {counts}\n[build_and_time] output: {out_path}")


if __name__ == "__main__":
    main()
