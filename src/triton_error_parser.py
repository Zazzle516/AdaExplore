"""Regex-based extractor for Triton compile/runtime tracebacks.

When a Triton kernel fails to compile (or fails at instantiation), `src/eval.py`
captures the full ``traceback.format_exc()`` string. Those tracebacks can be
large and noisy: gcc/ptxas subprocess dumps, generated-source listings with
``^^^`` carets, autotuner key errors, and chained ``tl.constexpr`` failures.

This module runs a pure-regex pass over that string, sorts the relevant bits
into named categories, truncates per-field, and returns a JSON-serializable
dict. It has no Triton import, so it's cheap and testable in isolation.

The single public entry point is :func:`parse_triton_compile_error`.
"""

from __future__ import annotations

import re

# Reuse the last-frame extractor rather than duplicating it.
from src.eval import extract_last_error

__all__ = ["parse_triton_compile_error"]


# One compiled pattern per category. Anchors validated against the real samples
# under ``outputs/KB-l2_AdaExplore_50/``. Each pattern is matched against the
# whole (multiline) traceback string; the matched line plus a window of
# surrounding lines is captured for context.
_CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    # Python SyntaxError in the generated kernel source — the ``^^^`` caret line
    # followed by ``SyntaxError:``.
    "syntax": re.compile(r"^\s*\^+\s*$\n^SyntaxError: .+$", re.MULTILINE),
    # Triton autotuner failures. The recurring signature in our corpus is the
    # ``'<key>' is not in list`` ValueError raised from autotuner.py while
    # resolving config keys.
    "autotune": re.compile(
        r"autotuner\.py|triton\.runtime\.autotuner|'.+' is not in list"
    ),
    # Host C compiler failures, surfaced as a subprocess.CalledProcessError on
    # the gcc/g++ command line, or a raw gcc diagnostic.
    "gcc": re.compile(
        r"subprocess\.CalledProcessError: Command '\[.*?gcc.*?\]'"
        r"|gcc(?:-\d+)?: .*?error:"
        r"|\bg\+\+ .*? error:"
    ),
    # PTX / ptxas / nvcc backend failures.
    "ptx": re.compile(
        r"ptxas .*?error|PTX assembly aborted|nvcc .*? error:|cuModule"
    ),
    # ``tl.constexpr`` misuse — e.g. reassigning a constexpr.
    "tl_constexpr": re.compile(r"tl\.constexpr|constexpr cannot be reassigned"),
    # Generic Triton compiler errors, including the ``at <line>:<col>:`` source
    # arrow block that follows.
    "triton_compile": re.compile(r"triton\.compiler\.errors\.\w+"),
}

# Surrounding-line window captured around each category match.
_WINDOW_BEFORE = 6
_WINDOW_AFTER = 4

# Pulls a trailing ``<dotted.ClassName>: <msg>`` headline line out of the
# traceback. Anchored at column 0 (so indented source lines never match) and
# requires a dotted-identifier path before the colon — this catches Python
# builtins (``SyntaxError:``, ``ValueError:``) as well as Triton classes whose
# names don't end in Error/Exception (``OutOfResources:``,
# ``UnsupportedLanguageConstruct:``). The last match is the actual exception.
_SUMMARY_RE = re.compile(r"^[A-Za-z_][\w.]*: \S.*$", re.MULTILINE)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``limit`` chars, appending a cut marker if shortened."""
    if len(text) <= limit:
        return text, False
    cut = len(text) - limit
    return text[:limit] + f"...(+{cut} chars truncated)", True


def _extract_window(lines: list[str], match_line_idx: int) -> str:
    """Return the matched line plus a window of surrounding lines, joined."""
    start = max(0, match_line_idx - _WINDOW_BEFORE)
    end = min(len(lines), match_line_idx + _WINDOW_AFTER + 1)
    return "\n".join(lines[start:end]).strip()


def _line_index_of(text: str, char_offset: int) -> int:
    """Map a character offset within ``text`` to its 0-based line index."""
    return text.count("\n", 0, char_offset)


def parse_triton_compile_error(
    traceback_str: str,
    *,
    error_class_name: str | None = None,
    per_field_max: int = 1500,
    summary_max: int = 240,
) -> dict:
    """Parse a Triton compile/runtime traceback into a structured dict.

    Args:
        traceback_str: The full ``traceback.format_exc()`` string.
        error_class_name: The exception class name (e.g. from
            ``metadata["compilation_error_name"]``), used as a fallback summary.
        per_field_max: Max chars per category excerpt before truncation.
        summary_max: Max chars for the one-line summary.

    Returns:
        A JSON-serializable dict with ``error_class``, ``summary``,
        ``last_frame``, ``categories`` (only non-empty matches), ``truncated``
        (which fields were cut), and ``raw_length``.
    """
    traceback_str = traceback_str or ""
    lines = traceback_str.split("\n")

    categories: dict[str, str] = {}
    truncated: list[str] = []

    for name, pattern in _CATEGORY_PATTERNS.items():
        match = pattern.search(traceback_str)
        if not match:
            continue
        line_idx = _line_index_of(traceback_str, match.start())
        excerpt = _extract_window(lines, line_idx)
        if not excerpt:
            continue
        excerpt, was_cut = _truncate(excerpt, per_field_max)
        categories[name] = excerpt
        if was_cut:
            truncated.append(name)

    # last_frame: trailing `File ...` frame plus the line that raised.
    last_frame = extract_last_error(traceback_str).strip() if traceback_str else ""

    # summary: the last `<ErrName>: <msg>` line, else fall back to the error
    # class name plus the head of the last frame.
    summary_matches = _SUMMARY_RE.findall(traceback_str)
    if summary_matches:
        summary = summary_matches[-1].strip()
    elif error_class_name:
        summary = f"{error_class_name}: {last_frame[:120]}".strip()
    else:
        summary = last_frame[:summary_max]
    summary, _ = _truncate(summary, summary_max)

    return {
        "error_class": error_class_name or "",
        "summary": summary,
        "last_frame": last_frame,
        "categories": categories,
        "truncated": truncated,
        "raw_length": len(traceback_str),
    }
