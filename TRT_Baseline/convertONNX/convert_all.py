#!/usr/bin/env python3
"""Stage 2: export every KernelBench level1+level2 Model to ONNX.

For each `*.py` in `dataset_root/level{N}`: dynamically import the file,
seed the RNG, instantiate `Model`, run `torch.onnx.export`, and record the
result in `manifest.json`.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def import_kb_module(py_path: Path, mod_name: str):
    # Each KernelBench file defines `class Model`; load under a unique name so
    # they don't collide in sys.modules.
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build spec for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cleanup(mod_name: str, *objs) -> None:
    # Without this, weights + traced graphs from prior files accumulate on the
    # GPU and starve later exports (saw ~14-18 GiB pinned before crashes on a
    # 24 GiB card). Drop refs, evict the module, then force GC + empty_cache.
    for o in objs:
        del o
    sys.modules.pop(mod_name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def export_one(py_path: Path, onnx_path: Path, cfg: dict) -> dict:
    """Export one KernelBench file. Returns a manifest entry dict."""
    entry = {
        "level": None,
        "file": py_path.name,
        "status": "pending",
        "onnx_path": str(onnx_path.relative_to(REPO_ROOT)),
        "error": None,
    }
    seed = int(cfg["onnx"]["random_seed"])
    opset = int(cfg["onnx"]["opset_version"])
    simplify = bool(cfg["onnx"]["simplify"])
    mod_name = f"kb_{py_path.stem}"

    M = model = inputs = out = None
    try:
        M = import_kb_module(py_path, mod_name)
        for attr in ("Model", "get_inputs", "get_init_inputs"):
            if not hasattr(M, attr):
                raise AttributeError(f"missing {attr}")

        # Seed *immediately* before construction — 33 L2 modules use
        # nn.Parameter(torch.randn(...)), so weight values depend on RNG state
        # at __init__ time. Reproducibility across runs requires this.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            model = M.Model(*M.get_init_inputs()).cuda().eval()
            inputs = tuple(
                t.cuda() if isinstance(t, torch.Tensor) else t
                for t in M.get_inputs()
            )

            # One forward pass purely to discover output arity for naming.
            # Result is freed before export — torch.onnx.export traces its own.
            out = model(*inputs)
        n_out = len(out) if isinstance(out, (list, tuple)) else 1
        del out
        out = None

        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            inputs,
            str(onnx_path),
            opset_version=opset,
            input_names=[f"in_{i}" for i in range(len(inputs))],
            output_names=[f"out_{i}" for i in range(n_out)],
            dynamic_axes=None,
            do_constant_folding=True,
        )

        if simplify:
            # Best-effort: simplifier failure shouldn't fail the export.
            try:
                import onnx
                import onnxsim

                simplified, ok = onnxsim.simplify(onnx.load(str(onnx_path)))
                if ok:
                    onnx.save(simplified, str(onnx_path))
                else:
                    entry["simplify_warning"] = "onnxsim returned ok=False"
            except Exception as e:
                entry["simplify_warning"] = f"{type(e).__name__}: {e}"

        entry["status"] = "ok"
    except Exception as e:
        entry["status"] = "onnx_export_failed"
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc(limit=8)
    finally:
        cleanup(mod_name, M, model, inputs, out)
    return entry


def parse_filter(s: str | None) -> tuple[int, str] | None:
    """`--filter level2/1` → (2, '1'). Matches '1_*' files in level2."""
    if not s:
        return None
    parts = s.strip().strip("/").split("/")
    if len(parts) != 2 or not parts[0].startswith("level"):
        raise ValueError(f"--filter must look like 'level2/1', got {s!r}")
    return int(parts[0][len("level"):]), parts[1]


def matches_filter(level: int, py_path: Path, flt) -> bool:
    if flt is None:
        return True
    flt_level, flt_id = flt
    if level != flt_level:
        return False
    return py_path.stem.startswith(f"{flt_id}_") or py_path.stem == flt_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--filter", default=None,
                    help="Restrict to one file, e.g. 'level2/1'")
    args = ap.parse_args()

    cfg = load_config(args.config)
    levels = list(cfg["levels"])
    dataset_root = (REPO_ROOT / cfg["dataset_root"]).resolve()
    onnx_root = THIS_DIR / "onnx_out"
    flt = parse_filter(args.filter)

    # When filtering, merge into the existing manifest so single-file retries
    # don't wipe results from a prior full sweep.
    manifest_path = THIS_DIR / "manifest.json"
    manifest: dict = {}
    if flt is not None and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    total = ok = fail = 0
    for level in levels:
        level_dir = dataset_root / f"level{level}"
        if not level_dir.is_dir():
            print(f"[warn] missing dataset dir: {level_dir}", file=sys.stderr)
            continue
        out_level_dir = onnx_root / f"level{level}"
        for py_path in sorted(level_dir.glob("*.py")):
            if not matches_filter(level, py_path, flt):
                continue
            total += 1
            onnx_path = out_level_dir / f"{py_path.stem}.onnx"
            print(f"[level{level}] {py_path.name} ...", flush=True)

            entry = export_one(py_path, onnx_path, cfg)
            entry["level"] = level
            entry["source_py"] = str(py_path.relative_to(REPO_ROOT))
            manifest[f"level{level}/{py_path.name}"] = entry

            if entry["status"] == "ok":
                ok += 1
                print(f"  -> ok  ({onnx_path.relative_to(REPO_ROOT)})")
            else:
                fail += 1
                print(f"  -> {entry['status']}: {entry['error']}")

            # Persist after every file so a crash mid-sweep is recoverable.
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"\n[convert_all] total={total} ok={ok} failed={fail}")
    print(f"[convert_all] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
