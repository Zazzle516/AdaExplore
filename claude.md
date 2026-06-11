# Triton Compile-Error Regex Extractor

## Context

Now, when a Triton kernel fails to compile, `src/eval.py::eval_kernel_against_ref()` (lines 442–462) stores the full `traceback.format_exc()` string verbatim into `KernelExecResult.metadata["compilation_error"]`. That string is later spliced into the evaluator prompt by `agentprompt/evaluator_prompt.py` (lines 148–154) via the `COMPILE_FAILURE` template — un-truncated, un-categorized.

This is fine for short Python `SyntaxError`s (~1 KB in our sample), but Triton tracebacks can blow up: PTX assembler dumps, gcc/`ptxas` stderr, generated-source listings with `^^^` carets, autotuner retry traces over many configs, and chained `tl.constexpr` failures can stretch into the tens of KB. Stuffing that into the evaluator prompt wastes tokens and buries the actionable signal.

The goal: when the failing implementation is Triton-based (i.e., `backend == "triton"` in `eval.py`), run a regex pass over the captured traceback, sort the relevant bits into named categories (syntax, autotune, gcc, ptx/ptxas, `tl.constexpr`, generic Triton CompilationError, plus a `last_frame` and a one-line `summary`), truncate per-field, and persist the structured result both into `metadata` (for the evaluator) and into a sibling JSON file (for inspection). The evaluator prompt then renders the structured fields instead of the raw traceback.

## Approach

### 1. New module: `src/triton_error_parser.py`

Single public entry point:

```python
def parse_triton_compile_error(
    traceback_str: str,
    *,
    error_class_name: str | None = None,
    per_field_max: int = 1500,
    summary_max: int = 240,
) -> dict
```

Returns a plain dict (JSON-serializable):

```python
{
    "error_class": "<exception class, e.g. triton.compiler.errors.CompilationError>",
    "summary": "<one-line headline pulled from the last `<ErrName>: <msg>` line>",
    "last_frame": "<trailing `  File ...` frame + the line that raised>",
    "categories": {
        "syntax":          "<excerpt or empty>",
        "autotune":        "...",
        "gcc":             "...",
        "ptx":             "...",
        "tl_constexpr":    "...",
        "triton_compile":  "...",
    },
    "truncated": ["syntax", "autotune", ...],   # which fields were cut
    "raw_length": <int>,
}
```

Implementation notes:

- One compiled `re` pattern per category. Anchors validated against the real samples under `outputs/KB-l2_AdaExplore_50/`:
  - `syntax`: `r"^\s*File \".+\", line \d+\s*\n.*\n\s*\^+\s*\nSyntaxError: .+$"` (multiline, capture the block ending in `SyntaxError:`)
  - `autotune`: `r"autotuner\.py"`, `r"triton\.runtime\.autotuner"`, plus the recurring `ValueError: '.+' is not in list` we see across `2_1/step_0`, `2_2/step_0`, etc.
  - `gcc`: `r"gcc(?:-\d+)?: .*error:"`, `r"\bg\+\+ .* error:"`
  - `ptx`: `r"ptxas .*error"`, `r"PTX assembly aborted"`, `r"nvcc .* error:"`
  - `tl_constexpr`: `r"tl\.constexpr"`, `r"constexpr "`, paired with the conventional `IncompatibleTypeError`/`CompilationError` line that follows
  - `triton_compile`: `r"triton\.compiler\.errors\.CompilationError"` plus its `at \d+:\d+:` source-arrow block
- Each category captures a window (the matched line plus ±N surrounding lines) and truncates to `per_field_max` (default 1500) chars with a `"...(+K chars truncated)"` suffix. Names of truncated fields go into `truncated`.
- `last_frame` reuses `extract_last_error` from `src/eval.py:592` rather than duplicating it.
- `summary` = the last line matching `r"^[A-Za-z_][\w.]*Error: .+$"` (truncated to `summary_max`); falls back to `error_class_name + ": <first 120 chars of last frame>"`.
- Categories with no match are dropped from the dict (do not emit empty strings).
- Module has no Triton import — pure regex on a string, so it's cheap and testable in isolation.

### 2. Hook into the capture path in `src/eval.py`

In the `except` branch at lines 456–462, after `metadata["compilation_error"] = full_error` and *only when `is_triton`* (already in scope at line 384):

```python
if is_triton:
    from src.triton_error_parser import parse_triton_compile_error
    metadata["compilation_error_parsed"] = parse_triton_compile_error(
        full_error,
        error_class_name=metadata["compilation_error_name"],
    )
```

The full traceback stays in `metadata["compilation_error"]` for now (don't break any existing consumer that reads it); evaluator switches to the parsed dict. A follow-up can drop the raw field after a sweep.

Apply the same hook in the model-load `RuntimeError` branch (STAGE 2, lines ~479–496) for the instantiation failure path — Triton autotune errors often surface there rather than at import.

**Also apply the hook in the STAGE 3 correctness-check branch (`except Exception` at lines ~515–522).** This is essential, not optional: Triton compiles the gcc launcher (`__triton_launcher...so`) **lazily on first kernel launch**, so launcher build failures (`subprocess.CalledProcessError` from `/usr/bin/gcc`) surface during the correctness check, *not* at compile or load time. This is precisely the `gcc` category the parser targets, yet it lands in the one stage that — in the original plan — was left uninstrumented. Observed empirically in `outputs/KB-l2_AdaExplore_50/2_9/step_{1,2,3}`: those steps have `compiled=true, correctness=false` with a gcc `runtime_error` but **no** `compilation_error_parsed`, so no `step_*_compile_error.json` was written and the evaluator prompt fell back to the raw metadata repr. Mirror the same block after `runtime_error`/`runtime_error_name` are set:

```python
if is_triton:
    from src.triton_error_parser import parse_triton_compile_error
    metadata["compilation_error_parsed"] = parse_triton_compile_error(
        full_error,
        error_class_name=metadata["runtime_error_name"],
    )
```

### 3. Write the dedicated JSON file in `agent/mcts.py`

Around the existing `step_{step_idx}_metrics.json` write (mcts.py ~lines 765–795), when `node.metrics is not None and "compilation_error_parsed" in node.metrics.metadata`, also dump:

```python
with open(os.path.join(self.log_path, f"step_{step_idx}_compile_error.json"), "w") as f:
    json.dump(node.metrics.metadata["compilation_error_parsed"], f, indent=2)
```

Gate **only** on the parsed key's presence — do *not* add `and not node.metrics.compiled`. The gcc-launcher failure path sets the parsed dict while `compiled=True`, so a `not compiled` gate would silently skip exactly the case we care about most.

Naming follows the existing `step_{step_idx}_*` convention; no new directory needed.

### 4. Update `agentprompt/evaluator_prompt.py`

The existing `COMPILE_FAILURE` template (lines 58–67) and its single `{compile_error}` substitution stay unchanged. Only the value computed at lines 148–154 changes: when `metadata["compilation_error_parsed"]` is present, dump the parsed dict as JSON and feed it into the same slot. Empty categories are already dropped by the parser, so the JSON naturally reflects only what was matched.

```python
if isinstance(run_info, KernelExecResult) and not run_info.correctness:
    parsed = run_info.metadata.get("compilation_error_parsed")
    if parsed:
        compile_error = json.dumps(parsed, indent=2, ensure_ascii=False)
    else:
        compile_error = (
            run_info.metadata.get("compilation_error")
            or run_info.metadata.get("runtime_error")
        )
    if compile_error:
        prompt += COMPILE_FAILURE.format(compile_error=compile_error)
```

Gate on `not run_info.correctness` (not `not run_info.compiled`) so the section also fires for compiled-but-incorrect runs whose error surfaced at the correctness/launch stage. Only append the block when `compile_error` is truthy, so a correctness mismatch with no captured traceback doesn't emit an empty failure section.

Fallback path (no parsed dict — e.g. CUDA backend, or Triton path that produced no parseable traceback) keeps the current behavior of dumping the raw traceback. No per-category markdown rendering, no template rename — the parser owns the structure, the prompt just substitutes its JSON form.

Optionally strip the debug fields `truncated` / `raw_length` from `parsed` before `json.dumps` if they prove distracting; default is to leave them in (they're useful signal that a field was cut).

## Critical files

- `src/triton_error_parser.py` — new, ~150 lines of regex + truncation.
- `src/eval.py` — parse hook in three except branches: STAGE 1 compile (~line 458), STAGE 2 model-load (~line 488), and STAGE 3 correctness-check (~line 517, where the gcc launcher build error surfaces). Reuses `extract_last_error` at line 592.
- `agent/mcts.py` — one extra `json.dump` near the existing `step_*_metrics.json` write (~line 767).
- `agentprompt/evaluator_prompt.py` — replace the `compile_error` value computation at lines 148–154 with the json.dumps branch. `COMPILE_FAILURE` template at lines 58–67 stays unchanged.

## Verification

1. **Unit-level**: write a small script that loads each of the 5 known-failing `step_*_metrics.json` files under `outputs/KB-l2_AdaExplore_50/` (paths listed below), feeds `metadata.compilation_error` through `parse_triton_compile_error`, and prints the resulting dict. Confirm:
   - `2_1/step_0`, `2_2/step_0`, `2_3/step_0`: `categories.autotune` populated, `summary` mentions `is not in list`.
   - Any sample with `SyntaxError`: `categories.syntax` populated and includes the `^^^` line.
   - Truncation kicks in only for fields longer than 1500 chars and the `truncated` list reflects it.
2. **End-to-end**: run a single MCTS step on a problem known to fail compile (e.g. `2_1`) with the current entry point in `agent/`, and confirm:
   - `outputs/<run>/2_1/step_0_compile_error.json` exists and is well-formed.
   - The evaluator prompt for that step (written to `step_0_prompt.txt`) shows the parsed dict as indented JSON inside the existing `### Compile failure` block, not the raw 1KB+ traceback.
3. **Backend split**: run one CUDA-backend step (set `backend="cuda"` ad-hoc) and confirm `compilation_error_parsed` is NOT set and the fallback path produces a bounded but unchanged-style prompt.
