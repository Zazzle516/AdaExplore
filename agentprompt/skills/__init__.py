"""Per-altitude skill registry for the kernel-optimization agent.

The package walks ``agentprompt/skills/*.md`` once at import and exposes
``generate_skill_prompt(arch_src, step_type)``, which detects the operator
families used by a reference arch (via ``detect_families``) and assembles the
matching skill sections at the requested altitude.
"""

import re
from pathlib import Path

from agentprompt.skills.detect import detect_families

__all__ = ["detect_families", "generate_skill_prompt"]

_SKILLS_DIR = Path(__file__).parent
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
                          step_type: str = "both") -> str:
    """Assemble the skill prompt for a reference arch at one altitude.

    step_type in {"large", "small", "both"}.
    "large" -> _base + family ## Design (+ _default Design if unknown ops)
    "small" -> _base + family ## Tuning (+ _default Tuning if unknown ops)
    "both"  -> _base + both sections per family (used by the evaluator)
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
    return "\n\n".join(b for b in blocks if b)


if __name__ == "__main__":
    # Section extractor smoke test (Verification §2): a fenced code block
    # containing `## Design` must NOT be mistaken for a section marker.
    fixture = (
        "# conv skills\n\n"
        "## Design (large step)\n\n"
        "design body line\n"
        "```\n"
        "## Design inside a fence should be ignored\n"
        "```\n\n"
        "## Tuning (small step)\n\n"
        "tuning body line\n"
    )
    parts = _parse_headered(fixture)
    assert parts["Design"].startswith("design body line"), parts["Design"]
    assert "inside a fence" in parts["Design"], parts["Design"]
    assert parts["Tuning"] == "tuning body line", parts["Tuning"]
    print("[OK] fenced ## Design not treated as a section marker")

    # Real conv.md round-trips through the three step types.
    large = _load_section("conv", "large")
    small = _load_section("conv", "small")
    both = _load_section("conv", "both")
    assert large and "im2col" in large
    assert small and "BLOCK_M" in small
    assert both == (large + "\n\n" + small)
    assert "BLOCK_M" not in large and "im2col" not in small
    print("[OK] conv large/small/both sections load correctly")

    print("\n----- generate_skill_prompt(step_type='large') -----")
    print(generate_skill_prompt("import torch.nn as nn\nnn.Conv2d(3, 4, 3)",
                                step_type="large"))
