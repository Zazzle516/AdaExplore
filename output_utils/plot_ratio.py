"""Plot iteration vs. speedup ratio for AdaExplore test cases.

Reads step_<n>_metrics.json files under outputs/KB-l{level}_AdaExplore_*/{level}_{pid}/
and plots one PNG per test case where:
    x-axis = iteration (step) number
    y-axis = ratio (runtime_stats.fast_p, i.e. baseline_time / runtime)

Failed / non-correct steps are shown as gaps (NaN) so the line breaks instead of
collapsing to zero.

Usage:
    # Batch mode (default): plot every test case under outputs/
    python output_utils/plot_ratio.py

    # Restrict to one level:
    python output_utils/plot_ratio.py --level 1

    # Single test case (matches the {level}_{pid} folder name):
    python output_utils/plot_ratio.py --case 1_1

    # Custom input / output directories:
    python output_utils/plot_ratio.py --input-dir outputs --output-dir output_utils/plots
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_DIR = os.path.join(ROOT, "outputs")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "output_utils", "plots")

STEP_RE = re.compile(r"^step_(\d+)_metrics\.json$")
RUN_DIR_RE = re.compile(r"^KB-l(\d+)_AdaExplore_\d+$")
CASE_DIR_RE = re.compile(r"^(\d+)_(\d+)$")


def collect_steps(case_dir):
    """Return a sorted list of (iteration, ratio_or_nan) tuples for one test case."""
    points = []
    for fn in os.listdir(case_dir):
        m = STEP_RE.match(fn)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(case_dir, fn)
        try:
            with open(path, "r") as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            points.append((step, float("nan")))
            continue

        compiled = metrics.get("compiled", False)
        correct = metrics.get("correctness", False)
        stats = metrics.get("runtime_stats") or {}
        ratio = stats.get("fast_p")

        if compiled and correct and isinstance(ratio, (int, float)) and ratio > 0:
            points.append((step, float(ratio)))
        else:
            points.append((step, float("nan")))

    points.sort(key=lambda p: p[0])
    return points


def plot_case(case_label, points, out_path):
    import matplotlib.pyplot as plt

    valid = [(x, y) for x, y in points if y == y]  # drop NaNs so adjacent points connect
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=4, color="#1f77b4")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="baseline (ratio=1.0)")

    ax.set_title(f"Speedup ratio vs. iteration — {case_label}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Ratio (baseline / runtime)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def find_case_dir(input_dir, case, level=None):
    """Locate the case subdir given a `{level}_{pid}` label."""
    m = CASE_DIR_RE.match(case)
    if not m:
        raise ValueError(f"--case must look like '<level>_<pid>', got: {case!r}")
    case_level = int(m.group(1))
    if level is not None and case_level != level:
        raise ValueError(f"--case {case} does not belong to --level {level}")

    candidates = []
    for run in os.listdir(input_dir):
        rm = RUN_DIR_RE.match(run)
        if not rm:
            continue
        if int(rm.group(1)) != case_level:
            continue
        path = os.path.join(input_dir, run, case)
        if os.path.isdir(path):
            candidates.append((run, path))

    if not candidates:
        raise FileNotFoundError(f"No directory matching {case} found under {input_dir}")
    return candidates


def iter_all_cases(input_dir, level=None):
    """Yield (run_name, case_label, case_dir) for every test case directory."""
    for run in sorted(os.listdir(input_dir)):
        rm = RUN_DIR_RE.match(run)
        if not rm:
            continue
        run_level = int(rm.group(1))
        if level is not None and run_level != level:
            continue
        run_dir = os.path.join(input_dir, run)
        if not os.path.isdir(run_dir):
            continue
        for case in sorted(os.listdir(run_dir), key=_case_sort_key):
            if not CASE_DIR_RE.match(case):
                continue
            case_dir = os.path.join(run_dir, case)
            if os.path.isdir(case_dir):
                yield run, case, case_dir


def _case_sort_key(name):
    m = CASE_DIR_RE.match(name)
    if not m:
        return (10**9, name)
    return (int(m.group(1)), int(m.group(2)))


def render(run_name, case_label, case_dir, output_dir):
    points = collect_steps(case_dir)
    if not points:
        print(f"[skip] {run_name}/{case_label}: no step_*_metrics.json files")
        return False
    out_path = os.path.join(output_dir, run_name, f"{case_label}.png")
    plot_case(f"{run_name} :: {case_label}", points, out_path)
    valid = sum(1 for _, y in points if y == y)  # NaN != NaN
    print(f"[ok]   {run_name}/{case_label}: {valid}/{len(points)} valid steps -> {out_path}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help=f"Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--level", type=int, choices=(1, 2), default=None,
                        help="Restrict batch mode to a single KernelBench level.")
    parser.add_argument("--case", default=None,
                        help="Single test case label, e.g. '1_1'. Overrides batch mode.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.input_dir):
        print(f"Input dir not found: {args.input_dir}", file=sys.stderr)
        return 2

    rendered = 0
    if args.case is not None:
        try:
            targets = find_case_dir(args.input_dir, args.case, args.level)
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        for run, case_dir in targets:
            if render(run, args.case, case_dir, args.output_dir):
                rendered += 1
    else:
        for run, case_label, case_dir in iter_all_cases(args.input_dir, args.level):
            if render(run, case_label, case_dir, args.output_dir):
                rendered += 1

    print(f"\nDone. {rendered} figure(s) written under {args.output_dir}")
    return 0 if rendered else 1


if __name__ == "__main__":
    sys.exit(main())
