"""Hardware-target adaptation for skill prompts.

Skill markdown (e.g. ``agentprompt/skills/conv.md``) uses ``{name}``
placeholders for hardware-specific guidance — the SMEM budget heuristic,
the GPU label printed in the prose, the bytes-per-element multiplier, etc.
``get_hardware_params`` resolves those placeholders for a target machine
so the same skill text adapts to A6000 / A100 / RTX-4090 / H100 without
rewrites.

When ``gpu_name`` / ``gpu_architecture`` are not supplied, we fall back
to ``detect_current_device``, which queries the active CUDA device via
``torch.cuda.get_device_properties``. That gives the ground-truth SMEM
size for the SM the kernel will actually run on (e.g. sm_86 A6000 has
100 KB while sm_80 A100 has 164 KB — the same "Ampere" architecture
table entry would be wrong for one of them).

The static tables remain as a fallback for callers that supply only the
architecture name, or run on a host without CUDA visible (offline
prompt generation, CI). When a value cannot be determined we fall back
to a conservative default rather than raising — the prompt is still
useful with a slightly off SMEM number, but a crash inside the prompt
loader would block every kernel-generation step.
"""

import re

# Max dynamic shared memory per SM, in KB, by NVIDIA architecture name.
# Sources: CUDA C Programming Guide, compute-capability tables.
_ARCH_SMEM_KB_PER_SM = {
    "Volta":     96,    # sm_70  (V100)
    "Turing":    64,    # sm_75  (T4, RTX 20xx)
    "Ampere":   164,    # sm_80  (A100); sm_86 (A6000, RTX 30xx) is 100
    "Ada":      100,    # sm_89  (RTX 4090, L4)
    "Hopper":   228,    # sm_90  (H100)
    "Blackwell": 228,   # sm_100 (B100/B200) — conservative
}

# Per-GPU overrides for SMs that diverge from their architecture default
# (e.g. A6000 / RTX-30xx are sm_86 with 100 KB even though A100 is 164).
_GPU_SMEM_KB_OVERRIDE = {
    "A6000":   100,
    "RTX3090": 100,
    "RTX 3090": 100,
    "RTX-3090": 100,
}

# Short label used in skill prose when both name and architecture are
# known. Falls back to ``"<arch> / <name>"`` otherwise.
_GPU_LABEL_OVERRIDES = {
    ("Ada",     "RTX-4090"): "Ada / RTX-4090",
    ("Ada",     "RTX 4090"): "Ada / RTX-4090",
    ("Ada",     "RTX4090"):  "Ada / RTX-4090",
    ("Ada",     "L4"):       "Ada / L4",
    ("Ampere",  "A100"):     "Ampere / A100",
    ("Ampere",  "A6000"):    "Ampere / A6000",
    ("Ampere",  "RTX-3090"): "Ampere / RTX-3090",
    ("Hopper",  "H100"):     "Hopper / H100",
    ("Turing",  "T4"):       "Turing / T4",
    ("Volta",   "V100"):     "Volta / V100",
}

_DTYPE_BYTES = {
    "fp64": 8, "float64": 8, "double": 8,
    "fp32": 4, "float32": 4, "tf32": 4, "float": 4,
    "fp16": 2, "float16": 2, "half": 2,
    "bf16": 2, "bfloat16": 2,
    "fp8":  1, "float8": 1, "e4m3": 1, "e5m2": 1,
    "int8": 1, "uint8": 1,
}

# When dtype is unknown, fall back to fp32 — it is the most common path
# and gives the most conservative (largest) tile-byte estimate.
_DEFAULT_DTYPE = "fp32"
_DEFAULT_DTYPE_BYTES = 4

# When neither name nor architecture is known. Picked to land near the
# A6000 / RTX-4090 ballpark so the SMEM budget heuristic stays useful.
_DEFAULT_SMEM_KB_PER_SM = 100

# Fraction of SMEM left available for the operand tile after Triton's
# bookkeeping + register-spill scratch. 0.8 mirrors the original conv.md
# "80 KB out of ~100 KB" heuristic.
_SMEM_BUDGET_FRACTION = 0.8


# Compute-capability (major, minor) -> NVIDIA architecture family.
# Used to translate ``torch.cuda.get_device_properties().major/.minor`` into
# the same arch keys used in ``_ARCH_SMEM_KB_PER_SM`` and the override table.
_ARCH_BY_COMPUTE_CAPABILITY = {
    (7, 0): "Volta",   (7, 2): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",  (8, 6): "Ampere",  (8, 7): "Ampere",
    (8, 9): "Ada",
    (9, 0): "Hopper",
    (10, 0): "Blackwell", (10, 1): "Blackwell",
    (12, 0): "Blackwell",
}

# Patterns to extract a short model code from a verbose vendor name like
# ``"NVIDIA GeForce RTX 4090"`` or ``"NVIDIA A100-SXM4-40GB"``. First match
# wins; the captured group is uppercased and inner whitespace becomes ``-``.
_GPU_MODEL_PATTERNS = (
    r"\b(H\d{2,3})\b",            # H100, H200
    r"\b(A\d{3,4})\b",            # A100, A6000
    r"\b(L\d{1,3})\b",            # L4, L40
    r"\b(V100)\b",
    r"\b(T4)\b",
    r"\b(RTX\s*\d{4})\b",         # RTX 4090, RTX4090, RTX 3090
    r"\b(B\d{2,3})\b",            # B100, B200 (Blackwell)
)

def _norm(s):
    """Strip whitespace + a leading ``NVIDIA `` from a vendor string."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if s.upper().startswith("NVIDIA "):
        s = s[len("NVIDIA "):].strip()
    return s


def _short_gpu_name(full_name):
    """Reduce a verbose CUDA device name to the model code.

    Examples:
      ``NVIDIA GeForce RTX 4090``  -> ``RTX-4090``
      ``NVIDIA A100-SXM4-40GB``    -> ``A100``
      ``NVIDIA H100 80GB HBM3``    -> ``H100``
      ``NVIDIA RTX A6000``         -> ``A6000``
    Falls back to the vendor-stripped string when no pattern matches.
    """
    if not isinstance(full_name, str) or not full_name:
        return ""
    for pat in _GPU_MODEL_PATTERNS:
        m = re.search(pat, full_name, flags=re.IGNORECASE)
        if m:
            return re.sub(r"\s+", "-", m.group(1).strip().upper())
    return _norm(full_name)


def _arch_from_compute_capability(major, minor):
    if not isinstance(major, int) or not isinstance(minor, int):
        return None
    return _ARCH_BY_COMPUTE_CAPABILITY.get((major, minor))


def detect_current_device():
    """Introspect the active CUDA device via ``torch.cuda``.

    Returns a dict with ``gpu_name`` / ``gpu_architecture`` /
    ``smem_kb_per_sm`` (any field may be ``None`` if not derivable), or
    ``None`` if torch is unavailable, no GPU is visible, or the query
    raises. Never raises — callers can safely unpack ``or {}``.
    """
    try:
        import torch  # local import: keeps ``hardware.py`` importable on hosts without torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
    except Exception:
        return None

    name = _short_gpu_name(getattr(props, "name", "") or "")
    arch = _arch_from_compute_capability(
        getattr(props, "major", None),
        getattr(props, "minor", None),
    )
    # ``shared_memory_per_multiprocessor`` is the per-SM SMEM cap and is
    # exposed on PyTorch >= 2.0. Returns bytes; convert to KB. Older
    # PyTorch only has ``shared_memory_per_block`` (per-block, smaller),
    # which is not what we want for the operand-tile budget. When torch
    # doesn't expose it, fall back to Triton's driver API which carries
    # ``max_shared_mem`` for every supported arch — and Triton is what
    # actually generates the kernels we're prompting for, so its number
    # is the ceiling that matters.
    smem_bytes = getattr(props, "shared_memory_per_multiprocessor", None)
    if not isinstance(smem_bytes, int) or smem_bytes <= 0:
        smem_bytes = _triton_max_shared_mem(idx)
    smem_kb = smem_bytes // 1024 if isinstance(smem_bytes, int) and smem_bytes > 0 else None
    return {
        "gpu_name": name or None,
        "gpu_architecture": arch,
        "smem_kb_per_sm": smem_kb,
    }


def _triton_max_shared_mem(device_index):
    """Triton's reported max dynamic SMEM per SM, in bytes, or ``None``."""
    try:
        import triton
        props = triton.runtime.driver.active.utils.get_device_properties(device_index)
        v = props.get("max_shared_mem") if hasattr(props, "get") else None
        return v if isinstance(v, int) and v > 0 else None
    except Exception:
        return None


def _smem_kb_per_sm(gpu_name, gpu_architecture):
    name = _norm(gpu_name)
    arch = _norm(gpu_architecture)
    if name in _GPU_SMEM_KB_OVERRIDE:
        return _GPU_SMEM_KB_OVERRIDE[name]
    if arch in _ARCH_SMEM_KB_PER_SM:
        return _ARCH_SMEM_KB_PER_SM[arch]
    return _DEFAULT_SMEM_KB_PER_SM


def _gpu_label(gpu_name, gpu_architecture):
    name = _norm(gpu_name)
    arch = _norm(gpu_architecture)
    if arch and (arch, name) in _GPU_LABEL_OVERRIDES:
        return _GPU_LABEL_OVERRIDES[(arch, name)]
    if arch and name:
        return f"{arch} / {name}"
    return arch or name or "GPU"


def _dtype_bytes(dtype_str):
    if not isinstance(dtype_str, str):
        return _DEFAULT_DTYPE_BYTES
    return _DTYPE_BYTES.get(dtype_str.lower().strip(), _DEFAULT_DTYPE_BYTES)


def _smem_budget_kb(smem_total_kb):
    # Round to a value tile-sweep prose can quote naturally (multiple of 4).
    raw = smem_total_kb * _SMEM_BUDGET_FRACTION
    return max(8, int(round(raw / 4.0)) * 4)


def get_hardware_params(gpu_name=None, gpu_architecture=None, dtype_str=None):
    """Resolve the placeholder values used by skill markdown.

    When ``gpu_name`` or ``gpu_architecture`` is missing, the active CUDA
    device (if any) is queried for the missing field. When *both* fields
    are detected from the live device, the device's reported SMEM-per-SM
    is preferred over the static architecture table — this matters for
    SMs that diverge from their architecture default (sm_86 A6000 has
    100 KB while sm_80 A100 has 164 KB; both are "Ampere").

    Returned keys (all values are strings or ints, safe for substitution):

    * ``gpu_name``         — vendor-trimmed GPU model (e.g. ``"A6000"``)
    * ``gpu_architecture`` — architecture family (e.g. ``"Ampere"``)
    * ``gpu_label``        — short ``"Arch / Name"`` label for prose
    * ``smem_kb_per_sm``   — max dynamic SMEM per SM, KB
    * ``smem_budget_kb``   — usable operand-tile SMEM budget, KB
    * ``dtype``            — display dtype (e.g. ``"fp32"``)
    * ``dtype_bytes``      — bytes per element for the SMEM math
    """
    detected = {}
    if not gpu_name or not gpu_architecture:
        detected = detect_current_device() or {}
        gpu_name = gpu_name or detected.get("gpu_name")
        gpu_architecture = gpu_architecture or detected.get("gpu_architecture")

    # Prefer the live SMEM number, but only when both name and arch came
    # from the live device — otherwise we'd mix a user-supplied name with
    # the wrong device's SMEM.
    smem_total = None
    if (detected.get("smem_kb_per_sm")
            and gpu_name == detected.get("gpu_name")
            and gpu_architecture == detected.get("gpu_architecture")):
        smem_total = detected["smem_kb_per_sm"]
    if smem_total is None:
        smem_total = _smem_kb_per_sm(gpu_name, gpu_architecture)

    return {
        "gpu_name": _norm(gpu_name) or "GPU",
        "gpu_architecture": _norm(gpu_architecture) or "GPU architecture",
        "gpu_label": _gpu_label(gpu_name, gpu_architecture),
        "smem_kb_per_sm": smem_total,
        "smem_budget_kb": _smem_budget_kb(smem_total),
        "dtype": (dtype_str or _DEFAULT_DTYPE).lower().strip(),
        "dtype_bytes": _dtype_bytes(dtype_str),
    }


def substitute_placeholders(text, params):
    """Replace ``{key}`` tokens in ``text`` with values from ``params``.

    Uses literal ``str.replace`` rather than ``str.format`` so unrelated
    braces in the markdown (e.g. set notation ``{(32,32,16), ...}`` or
    ``Σ_{h,w}`` in math) pass through untouched.
    """
    if not isinstance(text, str) or not text:
        return text
    for key, value in params.items():
        text = text.replace("{" + key + "}", str(value))
    return text
