"""Per-altitude skill registry for the kernel-optimization agent.

The package walks ``agentprompt/skills/*.md`` once at import and exposes
``generate_skill_prompt(arch_src, step_type)``, which detects the operator
families used by a reference arch (via ``detect_families``) and assembles the
matching skill sections at the requested altitude.
"""

import re
from pathlib import Path

from agentprompt.Utils.detect import detect_families
from agentprompt.Utils.hardware import (
    get_hardware_params,
    substitute_placeholders,
)

__all__ = ["detect_families", "generate_skill_prompt", "get_hardware_params"]

# Skill content lives in the sibling ``agentprompt/skills`` directory (.md
# files); the loader code lives here in ``Utils``.
_SKILLS_DIR = Path(__file__).parent.parent / "skills"
_SECTIONS: dict[tuple[str, str], str] = {}

# Filenames (stems) treated as headerless: no ## Design / ## Tuning split.
_HEADERLESS = {"_base"}


def _parse_headered(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code:
            m = re.match(r"^##\s+(Design|Tuning)\b", line)
            if m:
                current = m.group(1)
                sections.setdefault(current, [])
                continue
        if current is not None:
            sections[current].append(line)
    return {
        "Design": "\n".join(sections.get("Design", [])).strip(),
        "Tuning": "\n".join(sections.get("Tuning", [])).strip(),
    }


def _populate() -> None:
    for md in _SKILLS_DIR.glob("*.md"):
        family = md.stem
        text = md.read_text()
        if family in _HEADERLESS:
            _SECTIONS[(family, "full")] = text.strip()
            continue
        parts = _parse_headered(text)
        _SECTIONS[(family, "large")] = parts["Design"]
        _SECTIONS[(family, "small")] = parts["Tuning"]
        _SECTIONS[(family, "both")] = (
            (parts["Design"] + "\n\n" + parts["Tuning"]).strip()
        )


_populate()  # eager: runs once at import


def _load_section(family: str, step_type: str) -> str:
    return _SECTIONS.get((family, step_type), "")


def generate_skill_prompt(arch_src: str | None,
                          step_type: str = "both",
                          task_params: dict | None = None) -> str:
    """Assemble the skill prompt for a reference arch at one altitude.

    step_type in {"large", "small", "both"}.
    "large" -> _base + family ## Design (+ _default Design if unknown ops)
    "small" -> _base + family ## Tuning (+ _default Tuning if unknown ops)
    "both"  -> _base + both sections per family (used by the evaluator)

    ``task_params`` (when provided) is read for ``gpu_name`` /
    ``gpu_architecture`` / ``dtype_str`` and used to fill ``{key}``
    hardware placeholders in the assembled skill text.
    """
    base = _load_section("_base", "full")
    families, has_unknown = (
        detect_families(arch_src) if arch_src else ([], False)
    )
    blocks = [base]
    for fam in families:
        blocks.append(_load_section(fam, step_type))
    if has_unknown:
        blocks.append(_load_section("_default", step_type))
    text = "\n\n".join(b for b in blocks if b)
    if task_params:
        hw = get_hardware_params(
            gpu_name=task_params.get("gpu_name"),
            gpu_architecture=task_params.get("gpu_architecture"),
            dtype_str=task_params.get("dtype_str"),
        )
        text = substitute_placeholders(text, hw)
    return text

