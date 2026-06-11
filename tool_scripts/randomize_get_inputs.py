"""Insert a non-deterministic torch reseed at the top of every get_inputs() in datasets/.

The eval harness in src/eval.py calls torch.manual_seed(trial_seed) right before
invoking get_inputs(), so test inputs were identical across runs. Reseeding inside
get_inputs() with torch.seed() pulls entropy from the OS so each run draws fresh
random inputs.
"""
import ast
import glob
import re
import sys

MARKER = "torch.seed()  # reseed: fresh random inputs per run"

DATASET_GLOBS = [
    "/home/agiuser/AdaExplore/datasets/KernelBench/level1/*.py",
    "/home/agiuser/AdaExplore/datasets/KernelBench/level2/*.py",
    "/home/agiuser/AdaExplore/datasets/KernelBench/level3/*.py",
    "/home/agiuser/AdaExplore/datasets/KernelBench/level4/*.py",
    "/home/agiuser/AdaExplore/datasets/KernelBench_syn/syn_v1/*.py",
]


def patch_source(src: str) -> tuple[str, bool]:
    if MARKER in src:
        return src, False

    tree = ast.parse(src)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            target = node
            break
    if target is None:
        return src, False

    body = target.body
    if not body:
        return src, False

    first = body[0]
    insert_after_node = None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        insert_after_node = first

    lines = src.splitlines(keepends=True)

    if insert_after_node is not None:
        insert_lineno = insert_after_node.end_lineno
    else:
        insert_lineno = target.body[0].lineno - 1

    indent = " " * (target.body[0].col_offset)
    new_line = f"{indent}{MARKER}\n"

    new_lines = lines[:insert_lineno] + [new_line] + lines[insert_lineno:]
    return "".join(new_lines), True


def main() -> int:
    changed = 0
    skipped = 0
    errors = []
    for pattern in DATASET_GLOBS:
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, "r") as fh:
                    src = fh.read()
                new_src, did_change = patch_source(src)
                if did_change:
                    with open(path, "w") as fh:
                        fh.write(new_src)
                    changed += 1
                else:
                    skipped += 1
            except Exception as exc:
                errors.append((path, repr(exc)))

    print(f"patched: {changed}")
    print(f"skipped: {skipped}")
    if errors:
        print(f"errors: {len(errors)}")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
