# Plan — Evaluator-driven MCTS with per-altitude skill registry

## Context

Two interlocking changes to the agent's iteration mechanics:

1. **Skill registry replaces `OPTIMIZATION_RULES`.** The current single
   `OPTIMIZATION_RULES` block in `agentprompt/prompt_modules.py:19-79` ships on
   every iteration to proposer / reviser / tuner. It is conv/linear-centric —
   a KernelBench L1+L2 survey (200 files) found 74 unique operators and 14 of
   the top 20 (GroupNorm, GELU, Sigmoid, Mish, Tanh, mean/sum/min, logsumexp,
   MaxPool, clamp, …) are not mentioned in the rules at all. It is also
   design-altitude in nature — algebraic-shortcut bans, im2col sketch,
   scatter-add for ConvTranspose, fusion patterns — and tells the tuner
   nothing about *which knobs to turn* on an already-written kernel (block
   sizes, `num_warps`, `num_stages`, layout, autotune).

   Fix: family-grouped skill files with per-altitude `## Design` / `## Tuning`
   sections, loaded by an `ast.parse` walk over `ref_arch_src`. The
   orchestrator detects the operator families used by the reference arch and
   loads only those skill files — and within each file, only the section
   matching the call site's altitude.

2. **Evaluator-driven MCTS replaces the reviser piggyback.** Today the
   "reviser" is misnamed — it doesn't revise any kernel, it emits guidance
   that feeds the tuner inside `single_small_step` (`agent/actions.py:17`).
   MCTS decides large-vs-small via a fixed `p_large=0.25` Bernoulli plus a
   `small_step_limit` cap (`agent/mcts.py:616-624`); the reviser's analysis
   never reaches that decision.

   Fix: rename to **evaluator**, lift it out of `single_small_step` into its
   own pass that runs after every new kernel (both step types), have it emit
   guidance for *both* altitudes plus a binary direction tag, and use the
   direction tag to soft-bias MCTS. The proposer gains an optional
   `large_guidance` slot (treated as optional context, not a directive); the
   tuner's existing `tuning_guidance` slot is now sourced from the
   evaluator's `small_guidance`.

The two changes are coupled: the evaluator depends on the skill registry
(`step_type="both"` is what makes the dual-guidance evaluator possible), and
the skill registry's per-altitude split is what makes routing guidance to
the right action's prompt clean.

The LLM harness (`agent/inference_server.py:147–214`) is pure text completion
— no tool/function calling. So skill *lookup* runs in the orchestrator, not
the LLM. `ast.parse(ref_arch_src)` is sub-millisecond, so it runs inline at
every prompt assembly — no `args`-level caching needed.

Scope of the direction-tag piece: **MCTS only.** IR-mode orchestrators
(IRS / IRL / IRB / IRLE / PS, dispatched at `agent/agent_entry.py:105-129`)
use fixed step ratios by design. They still call the evaluator each
iteration for its guidance text (which improves whatever step they were
already going to take), but parse-and-ignore the direction tag.

## Workflow

```
Iteration 0 (root):
  proposer (skill_prompt step_type="large") → kernel_0
  execute kernel_0 → metrics_0
  evaluator (skill_prompt step_type="both", inputs: kernel_0 + metrics_0)
    → (small_guidance, large_guidance, direction ∈ {"large","small",None})
  store all three on the MCTSNode wrapping kernel_0

Iteration N ≥ 1 — MCTS expansion:
  selected_node = MCTS.select(...)
  p_large = 0.25                 # base
  direction_bias = 2.5           # default multiplier
  if selected_node.evaluator_direction == "large":
    p_large = min(0.95, p_large * direction_bias)
  elif selected_node.evaluator_direction == "small":
    p_large = max(0.05, p_large / direction_bias)
  use_large = (num_small_children >= small_step_limit) or (random() < p_large)

  if use_large:
    proposer (step_type="large", optional large_guidance=selected_node.large_guidance)
      → kernel_N
  else:
    tuner (step_type="small", tuning_guidance=selected_node.small_guidance)
      → kernel_N  (str_replace mutation of selected_node.kernel)

  execute kernel_N → metrics_N
  evaluator on kernel_N → store on new node
  MCTS backprop
```

IR-mode (IRS / IRL / IRB / IRLE / PS): same evaluator call, same guidance
threading; direction tag parsed and logged but unused.

## Design

### Skill registry

#### `ast.parse` detection

`ast.parse(arch_src)` followed by `ast.walk` produces a stream of `ast.Call`
nodes. For each call we resolve the short name from `node.func.attr` (for
`nn.Conv2d(...)` or `x.sum(...)`) or `node.func.id` (for `Conv2d(...)` after
`from torch.nn import Conv2d`). Every resolved name lands in exactly one of
two buckets:

1. **Known primitive ops** — the name is a key in `FAMILY_MAP`. We look up
   the family slug (`conv`, `linear`, `norm`, `activation`, `reduction`,
   `pooling`) and add it to the detected-families set. The prompt assembler
   loads `agentprompt/skills/<family>.md` for each detected family.
2. **Unknown calls** — the name is *not* in `FAMILY_MAP` (long-tail ops,
   user-defined helpers, builtins like `range`/`len`, tensor factories like
   `torch.zeros`). These flip a single `has_unknown_ops` boolean. When true,
   the prompt assembler additionally loads `agentprompt/skills/_default.md`.

Calls in `__init__` (e.g. `nn.Sequential(...)`, `nn.Conv2d(...)` as a
constructor) and calls in `forward` (e.g. `F.gelu(x)`, `x.sum(dim=1)`) are
treated identically — both are evidence the arch *uses* that op family.

#### Layout

```
agentprompt/skills/
├── __init__.py            # exports detect_families, generate_skill_prompt; private _load_section
├── _base.md               # always-on, single-section: algebraic-shortcut ban + custom-kernel authorization
├── _default.md            # generic fallback, has ## Design + ## Tuning; loaded when ast.parse sees unknown calls
├── conv.md                # Conv1/2/3d, ConvTranspose1/2/3d; ## Design + ## Tuning
├── linear.md              # Linear, matmul, bmm, einsum, addmm; ## Design + ## Tuning
├── norm.md                # BatchNorm*, GroupNorm, LayerNorm, InstanceNorm*, RMSNorm; ## Design + ## Tuning
├── activation.md          # GELU, Sigmoid, Tanh, Mish, ReLU/LeakyReLU, SiLU, Softplus, ELU; ## Design + ## Tuning
├── reduction.md           # sum, mean, min, max, prod, softmax, log_softmax, logsumexp; ## Design + ## Tuning
├── pooling.md             # MaxPool{1,2,3}d, AvgPool{1,2,3}d, AdaptiveAvgPool{1,2,3}d; ## Design + ## Tuning
├── registry.py            # FAMILY_MAP: operator-name → family-slug
└── detect.py              # ast-based detector: arch_src → (sorted family slugs, has_unknown_ops bit)
```

#### `_base.md` (always on, single section, ~15 lines)

The *safety contract* — two operator-agnostic rules that apply regardless of
which ops appear AND regardless of altitude:

1. **The algebraic-shortcut ban** — every operator in the reference forward
   must execute at runtime on the actual input tensor; do not collapse a
   heavy op into a downstream reduction at init time. General phrasing — no
   `Conv2d`/`Linear` examples baked in (those move into the conv/linear
   skills).
2. **Custom-kernel authorization** — any `nn.*` op in the reference is a
   valid replacement target; you may rewrite it with a Triton or CUDA
   extension kernel.

Single-section because both rules apply at both altitudes. To keep the rule
load-bearing, family `## Design` / `## Tuning` sections do NOT restate it.

#### Family skill file structure

Each family file is split into two H2 sections:

```markdown
# {family} skills

Covers {ops}.

## Design (large step)

{strategy for designing the kernel from scratch — kernel layout, fusion
patterns, starter sketches, what to absorb into what}

## Tuning (small step)

{knobs for refining an already-written kernel — block/tile sizes,
`num_warps`, `num_stages`, layout (`channels_last`/`tl.dot` hints),
autotune configs, occupancy traps}
```

Concrete content per family:

- **`conv.md`** —
  - *Design:* im2col-+-tiled-GEMM sketch; ConvTranspose as scatter-add of
    input × kernel outer product; fold BN-into-conv at eval; absorb
    bias/ReLU into the same kernel.
  - *Tuning:* BLOCK_M / BLOCK_N / BLOCK_K choices for the GEMM; `num_warps`
    and `num_stages`; channels_last layout; autotune config grid; coalesced
    load patterns.
- **`linear.md`** —
  - *Design:* tiled GEMM; fuse bias and downstream activation.
  - *Tuning:* tile shapes for `tl.dot`, `num_warps`/`num_stages`, `GROUP_M`
    for L2 reuse, autotune over a small config list.
- **`norm.md`** —
  - *Design:* Welford for BN/IN, two-pass for LN/GN; fuse with adjacent
    activation; `rsqrt` for RMSNorm; fold affine into preceding linear *only*
    when it's a pure affine.
  - *Tuning:* one program per (N, C) or (N, H, W) tile depending on stat
    axis; block size for the reduction dim; persistent kernels for small
    feature dims.
- **`activation.md`** —
  - *Design:* fuse into the producing kernel (conv-bias-ReLU,
    matmul-bias-GELU); never run as a standalone Triton kernel except as a
    starter sketch.
  - *Tuning:* if forced to keep a standalone activation kernel, vectorize
    loads (`BLOCK_SIZE * tl.constexpr` per program), pick `num_warps` to
    saturate memory bandwidth.
- **`reduction.md`** —
  - *Design:* block-level reduction with shared-memory tree; online softmax
    / log-sum-exp shift for numerical stability.
  - *Tuning:* BLOCK_SIZE for the reduction axis; one program per output
    element; `num_warps` sized to the reduction width; persistent kernels if
    the reduction dim is small.
- **`pooling.md`** —
  - *Design:* coalesced strided reads; combine pool + activation; for
    adaptive pooling, precompute index map at init.
  - *Tuning:* one program per (N, C, output-tile); vectorized strided loads;
    BLOCK choices that match the kernel size.

Tuning content for each family is sourced from standard Triton vocabulary
**plus** a read of 3–5 successful refinement steps per family in
`outputs/KB-l*_AdaExplore_50/` — pull concrete knob names from real
trajectories, not invented (see Verification §7).

#### `_default.md` (loaded only when `has_unknown_ops` is true)

Carries the generic permitted-transformations toolkit, also split into
`## Design` and `## Tuning`. Phrased op-agnostically. When every call is
recognized, the family skills already cover the permitted transformations
relevant to those ops, so `_default.md` is omitted; when at least one call
is unknown, the family skills (if any) plus `_default.md` together give the
agent both targeted and generic toolkits.

#### `registry.py` — operator-to-family map

```python
FAMILY_MAP = {
    # conv
    "Conv1d": "conv", "Conv2d": "conv", "Conv3d": "conv",
    "ConvTranspose1d": "conv", "ConvTranspose2d": "conv", "ConvTranspose3d": "conv",
    # linear
    "Linear": "linear", "matmul": "linear", "bmm": "linear", "einsum": "linear",
    "addmm": "linear",
    # norm
    "BatchNorm1d": "norm", "BatchNorm2d": "norm", "BatchNorm3d": "norm",
    "GroupNorm": "norm", "LayerNorm": "norm",
    "InstanceNorm1d": "norm", "InstanceNorm2d": "norm", "InstanceNorm3d": "norm",
    "RMSNorm": "norm",
    # activation
    "GELU": "activation", "ReLU": "activation", "LeakyReLU": "activation",
    "Sigmoid": "activation", "Tanh": "activation", "Mish": "activation",
    "SiLU": "activation", "Softplus": "activation", "ELU": "activation",
    "gelu": "activation", "relu": "activation", "sigmoid": "activation",
    "tanh": "activation", "mish": "activation", "silu": "activation",
    # reduction
    "sum": "reduction", "mean": "reduction", "min": "reduction", "max": "reduction",
    "prod": "reduction", "softmax": "reduction", "log_softmax": "reduction",
    "logsumexp": "reduction", "argmax": "reduction", "argmin": "reduction",
    # pooling
    "MaxPool1d": "pooling", "MaxPool2d": "pooling", "MaxPool3d": "pooling",
    "AvgPool1d": "pooling", "AvgPool2d": "pooling", "AvgPool3d": "pooling",
    "AdaptiveAvgPool1d": "pooling", "AdaptiveAvgPool2d": "pooling",
    "AdaptiveAvgPool3d": "pooling",
}
```

Covers the top 39 ops from the survey (≥90% of L1+L2 files). Long-tail ops
not in the map flip the `has_unknown_ops` bit — base + `_default.md` still
apply, and adding a new entry is one line.

#### `detect.py` — AST walker (~30 LOC)

```python
import ast
from agentprompt.skills.registry import FAMILY_MAP

# Plumbing / shape ops / tensor factories / Python builtins. These are NOT
# evidence of a missing skill file — ignoring them keeps `has_unknown_ops`
# from firing on virtually every arch.
BUILTIN_IGNORE = {
    # Python builtins commonly seen in forward()
    "range", "len", "int", "float", "str", "list", "tuple", "dict", "print",
    "isinstance", "getattr", "setattr", "hasattr", "min", "max", "abs",
    # NOTE: "min"/"max" appear here as builtins AND in FAMILY_MAP as reductions.
    # The FAMILY_MAP lookup runs first, so torch tensor `.min(dim=)` still
    # tags `reduction`; bare `min(a, b)` is treated as a builtin.
    # tensor shape / view ops
    "view", "reshape", "contiguous", "permute", "transpose", "flatten",
    "squeeze", "unsqueeze", "expand", "expand_as", "repeat", "unbind",
    "split", "chunk", "stack", "cat", "concat",
    # tensor factories / dtype / device
    "zeros", "ones", "empty", "full", "arange", "linspace", "tensor",
    "zeros_like", "ones_like", "empty_like", "full_like", "to", "cpu",
    "cuda", "float", "half", "double", "long", "int", "bool",
    # nn.Module plumbing (constructor calls in __init__)
    "Sequential", "ModuleList", "ModuleDict", "ParameterList", "Parameter",
    # misc
    "size", "numel", "item", "clone", "detach", "requires_grad_",
}


def detect_families(arch_src: str) -> tuple[list[str], bool]:
    """Returns (sorted family slugs, has_unknown_ops flag).

    has_unknown_ops fires only when an `nn.*` attribute call (or top-level
    capitalized constructor matching the `nn.*` convention) is unrecognized.
    Bare names that look like Python builtins / tensor plumbing are ignored.
    """
    if not arch_src or not isinstance(arch_src, str):
        return [], False
    try:
        tree = ast.parse(arch_src)
    except SyntaxError:
        return [], False
    found = set()
    has_unknown = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # Distinguish `nn.Foo(...)` / `F.foo(...)` (Attribute) from bare
        # `Foo(...)` / `foo(...)` (Name). Only Attribute calls whose parent
        # module is `nn` / `F` / `functional` count as nn.* evidence.
        if isinstance(f, ast.Attribute):
            name = f.attr
            parent = f.value.id if isinstance(f.value, ast.Name) else None
            is_nn_call = parent in {"nn", "F", "functional"}
        elif isinstance(f, ast.Name):
            name = f.id
            is_nn_call = False
        else:
            continue
        if name in FAMILY_MAP:
            found.add(FAMILY_MAP[name])
            continue
        if name in BUILTIN_IGNORE:
            continue
        # Unknown call. Flip the flag only if it looks like a real nn op —
        # i.e. an `nn.*` / `F.*` attribute call we don't recognize. Bare
        # unknown names are conservatively ignored (likely user helpers).
        if is_nn_call:
            has_unknown = True
    return sorted(found), has_unknown
```

#### `__init__.py` — section-aware loader

At module import time, the package walks `agentprompt/skills/*.md` once
and populates a module-level dict `_SECTIONS: dict[(family, step_type), str]`.
Two file shapes are handled:

- **Headerless files** (filename stem listed in `_HEADERLESS`, currently
  just `_base`): no `## Design` / `## Tuning` split. Only `(family, "full")`
  is populated; the `"large"` / `"small"` / `"both"` keys are not set for
  these files.
- **Family files** (everything else, including `_default`): the line-scanner
  splits on `^## (Design|Tuning)\b` and populates three step-type entries
  (`large`, `small`, `both`). No `"full"` key is written for these.

The per-file parser is a line-scanner that:

- treats `^## (Design|Tuning)\b` as section markers,
- tracks a fenced-code-block toggle (lines starting with ` ``` `) so headers
  inside code blocks are NOT mistaken for section markers,
- stores the body of each section trimmed.

For each headered file it inserts three entries into `_SECTIONS`:
`(family, "large")` → Design body, `(family, "small")` → Tuning body,
`(family, "both")` → Design + Tuning concatenated. Headerless files
get only `(family, "full")` (the whole file). No `"full"` entry is
written for headered files — nothing reads it.

`_load_section(family, step_type)` collapses to a single dict lookup
returning empty string for missing keys. No lazy memoization, no per-call
file I/O — everything is parsed exactly once at `import agentprompt.skills`
time. Skill files are static for the lifetime of a run, so there is nothing
to invalidate.

```python
# agentprompt/skills/__init__.py
import re
from pathlib import Path

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
        _SECTIONS[(family, "both")]  = (
            (parts["Design"] + "\n\n" + parts["Tuning"]).strip()
        )


_populate()  # eager: runs once at import


def _load_section(family: str, step_type: str) -> str:
    return _SECTIONS.get((family, step_type), "")


def generate_skill_prompt(arch_src: str | None,
                          step_type: str = "both") -> str:
    """
    step_type ∈ {"large", "small", "both"}.
    "large" → _base + family ## Design (+ _default Design if unknown ops)
    "small" → _base + family ## Tuning (+ _default Tuning if unknown ops)
    "both"  → _base + both sections per family (used by the evaluator)
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
```

`_base.md` is handled via the `_HEADERLESS` set: only `("_base", "full")`
is populated, and `generate_skill_prompt` always pulls the `"full"`
variant for it regardless of step_type. To add another headerless skill
file later, add its stem to `_HEADERLESS`.

Order: `_base` → family skills (sorted) → `_default` (if any unknown calls).
`detect_families` runs inline here on every call — sub-millisecond cost, no
`args` plumbing needed.

### Evaluator

#### Prompt file

New file `agentprompt/evaluator_prompt.py` (replaces `reviser_prompt.py`).
Structure mirrors the existing reviser prompt but with three required
output tags. The `### Goal` becomes:

```
### Goal

You evaluate the most recent custom Triton kernel and its measured
performance, then emit guidance for the next iteration. Produce exactly
three tagged blocks, in this order:

<small_guidance>
1-3 concrete tuning bullets — block sizes, num_warps, num_stages, layout,
autotune configs, fusion opportunities within the existing kernel
structure. These will be consumed by the tuner if MCTS chooses a
refinement step.
</small_guidance>

<large_guidance>
1-3 concrete design bullets — what a from-scratch rewrite should change
about the kernel's strategy (im2col vs. direct, scatter-add vs. gather,
fusion boundary, persistence). These will be consumed by the proposer if
MCTS chooses a redesign step.
</large_guidance>

<direction>large</direction>  if you believe the current kernel is
structurally unable to reach the target and a from-scratch rewrite is more
promising than continued tuning.

<direction>small</direction>  if you believe continued tuning of the
current kernel will close the gap.

Emit exactly one <direction> tag. Both guidance blocks are always required
even if one is short.
```

Skill prompt loaded with `step_type="both"` so the evaluator sees both
Design and Tuning content per family.

#### `run_evaluator` helper (`agent/actions.py`)

```python
def run_evaluator(
    ref_arch_src: str,
    kernel: str,
    metrics: KernelExecResult,
    inference_server: str,
    args: argparse.Namespace,
) -> tuple[str, str, Optional[str]]:
    """Run evaluator on a freshly produced kernel.
    Returns (small_guidance, large_guidance, direction).
    Either guidance may be empty string on parse failure; direction is None
    if the tag is missing or malformed.
    """
    prompt = generate_evaluator_prompt(
        task_params=args.task_params,
        custom_triton_kernels=kernel,
        run_info=metrics,
        experience_guidance_path=args.general_memory_path,
        knowledge_1_threshold=args.knowledge_1_threshold,
    )
    output = query_inference_server(server=inference_server,
                                    model_name=args.model_name,
                                    prompt=prompt,
                                    max_completion_tokens=args.max_completion_tokens)
    small_g = _extract_tag(output, "small_guidance")
    large_g = _extract_tag(output, "large_guidance")
    direction = _extract_direction(output)
    return small_g, large_g, direction
```

Both extractors are last-match — if the LLM drafts a tag and then re-emits a
corrected version, the final occurrence is the binding one. Same rule for
all three tags so behavior is consistent:

```python
def _extract_tag(output: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", output, re.DOTALL)
    return matches[-1].strip() if matches else ""

def _extract_direction(output: str) -> Optional[str]:
    matches = re.findall(r"<direction>\s*(large|small)\s*</direction>", output)
    return matches[-1] if matches else None
```

#### Compile-failure handling

`generate_evaluator_prompt` checks `run_info.compiled` (the
`KernelExecResult.compiled` flag — confirm exact attribute when implementing).
When `False`, inject a dedicated section into the prompt before the `### Goal`
block:

```
### Compile failure

The kernel failed to compile. Traceback / error:

{run_info.metadata["compilation_error"] or run_info.metadata["runtime_error"]}

Your <large_guidance> should diagnose the structural cause; your
<small_guidance> may suggest a minimal patch if one is obvious, or
restate that a redesign is needed.
```

The `### Goal` block also gains a sentence: *"On compile failure, default
to `<direction>large</direction>` unless the traceback points at a
trivially-fixable issue (e.g. a typo, a wrong constexpr, a missing import)."*

### Action functions become pure executors

`single_small_step` no longer calls the reviser. It receives
`tuning_guidance` as a parameter (read by MCTS from
`selected_node.small_guidance`):

```python
def single_small_step(ref_arch_src, inference_server, previous_kernels,
                     previous_metrics, args, tuning_guidance: str = ""):
    tuner_prompt = generate_tuner_prompt(..., tuning_guidance=tuning_guidance)
    tuner_output = query_inference_server(...)
    tuned_kernel = apply_str_replace_edits(previous_kernels[-1], tuner_output)
    tuned_metrics = wrapped_eval_kernel_against_ref(...)
    return tuned_kernel, tuned_metrics, {"tuner_prompt": tuner_prompt, ...}
```

`single_large_step` gains an optional `large_guidance` parameter:

```python
def single_large_step(..., large_guidance: Optional[str] = None):
    proposer_prompt = generate_proposer_prompt(..., large_guidance=large_guidance)
    ...
```

`generate_proposer_prompt` gets a new optional `large_guidance` slot,
rendered as an "optional context" block when set:

```
### Optional context — diagnosis of the previous attempt

The evaluator analyzed the previous kernel and suggested the following
design directions. Use them if they fit; ignore them if a different
approach is better.

{large_guidance}
```

The "optional context" framing is deliberate — the proposer is still
greenfield; the guidance informs but does not direct.

### MCTS wiring

`MCTSNode` (`agent/mcts.py:43-65`) gets three new fields:

```python
small_guidance: str = ""
large_guidance: str = ""
evaluator_direction: Optional[str] = None
```

`_create_node` (`agent/mcts.py:246-283`) accepts and stores them.

In `expand_large` and `expand_small`: pull
`selected_node.{large,small}_guidance` into the step call; after the step
returns, call `run_evaluator` on the new kernel and store the three outputs
on the new child node. If the kernel failed to compile, still run the
evaluator — the prompt assembler injects the traceback into a dedicated
`### Compile failure` section (see §Evaluator → Compile-failure handling).

When `expand_large` runs, it pulls `large_guidance=selected_node.large_guidance`
from the **parent** that was selected (not the new child) — the parent's
evaluator output is what triggered the large-step decision. Same for
`expand_small` and `small_guidance`.

`step()` decision block (`agent/mcts.py:616-624`) becomes:

```python
if selected_node.created_by == "dummy_root":
    use_large_step = True
else:
    p_large = getattr(self.args, 'p_large', 0.25)
    direction_bias = getattr(self.args, 'direction_bias', 2.5)
    direction = getattr(selected_node, 'evaluator_direction', None)
    if direction == "large":
        p_large = min(0.95, p_large * direction_bias)
    elif direction == "small":
        p_large = max(0.05, p_large / direction_bias)
    num_small_step_children = sum(
        1 for c in selected_node.children if c.created_by == "small_step"
    )
    use_large_step = (
        num_small_step_children >= self.small_step_limit
        or random.random() < p_large
    )
```

Symmetric multiplicative bias: one `direction_bias` knob (default 2.5×)
applied as `p_large *= bias` for `"large"` and `p_large /= bias` for
`"small"`, clamped to `[0.05, 0.95]`. Composes cleanly with whatever base
`p_large` the user picks — at the default `(p_large=0.25, bias=2.5)` this
yields `0.625` on `"large"` and `0.10` on `"small"`, matching the original
intent. Raising base `p_large` to `0.5` correctly shifts both directions
upward (`0.95` and `0.20`) instead of silently capping the large branch.

Soft bias: the `small_step_limit` floor still force-flips to large
(preserves exploration). Nodes from `expand_large` start with default
`p_large` until their own evaluator runs.

### IR-mode handling

`agent/large_loop.py` and `agent/small_loop.py` run fixed-count `for` loops.
After each `single_*_step`, run `run_evaluator` on the new kernel and pass
the resulting guidance into the next iteration:

- `run_large_loop`: thread `large_guidance` from iteration N's evaluator
  into iteration N+1's `single_large_step` call.
- `run_small_loop`: thread `small_guidance` similarly.

Direction tag is parsed and logged but unused (these orchestrators have
fixed step ratios).

## Files to modify / create

### Skill registry

| Path | Change |
|---|---|
| `agentprompt/skills/_base.md` | **new** — single-section safety contract (~15 lines) |
| `agentprompt/skills/_default.md` | **new** — generic fallback with `## Design` + `## Tuning` (~40 lines) |
| `agentprompt/skills/conv.md` | **new** — `## Design` (im2col / scatter-add / BN-fold) + `## Tuning` (tile sizes, num_warps, layouts) |
| `agentprompt/skills/linear.md` | **new** — `## Design` (tiled GEMM, fuse bias/activation) + `## Tuning` (BLOCK_M/N/K, tl.dot layouts, autotune) |
| `agentprompt/skills/norm.md` | **new** — `## Design` (Welford / two-pass / rsqrt, fuse w/ activation) + `## Tuning` (block per channel, reduction layout) |
| `agentprompt/skills/activation.md` | **new** — `## Design` (fuse into producer) + `## Tuning` (standalone-kernel knobs) |
| `agentprompt/skills/reduction.md` | **new** — `## Design` (block tree, online softmax, LSE shift) + `## Tuning` (BLOCK size, persistent kernels) |
| `agentprompt/skills/pooling.md` | **new** — `## Design` (coalesced strided reads, precomputed index map) + `## Tuning` (BLOCK, vectorized loads) |
| `agentprompt/skills/registry.py` | **new** — `FAMILY_MAP` dict |
| `agentprompt/skills/detect.py` | **new** — `detect_families(arch_src)` ast walker |
| `agentprompt/skills/__init__.py` | **new** — re-exports `detect_families`, `generate_skill_prompt`; private `_load_section(family, step_type)` is a dict lookup over `_SECTIONS`, populated once at import via a fenced-block-aware H2 parser walking `agentprompt/skills/*.md`. Filenames in `_HEADERLESS = {"_base"}` populate only the `"full"` key; family files populate `large` / `small` / `both`. |

### Evaluator and prompt files

| Path | Change |
|---|---|
| `agentprompt/evaluator_prompt.py` | **new** — renamed from `reviser_prompt.py`; emits three-tag output (`<small_guidance>`, `<large_guidance>`, `<direction>`). Loads `step_type="both"` skill content. |
| `agentprompt/reviser_prompt.py` | **delete** after imports migrate |
| `agentprompt/proposer_prompt.py` | Add optional `large_guidance` parameter to `generate_proposer_prompt`; render "optional context" block when set. Swap `generate_optimization_rules_prompt()` for `generate_skill_prompt(arch_src, step_type="large")`. Tweak `PROBLEM_STATEMENT` wording: "Optimization Rules" → "Optimization Skills". |
| `agentprompt/tuner_prompt.py` | Swap to `generate_skill_prompt(arch_src, step_type="small")`. Tweak `PROBLEM_STATEMENT`. The `tuning_guidance` slot already exists; source is now the evaluator's `small_guidance`. |
| `agentprompt/prompt_modules.py` | Delete `OPTIMIZATION_RULES` and `generate_optimization_rules_prompt`. Keep `generate_experience_guidance_prompt` and `generate_hardware_information_prompt`. |

### Action / MCTS / loop edits

| Path | Change |
|---|---|
| `agent/actions.py` | Add `run_evaluator(...)` helper plus `_extract_tag` / `_extract_direction`. Strip reviser invocation out of `single_small_step`; accept `tuning_guidance` parameter. Add optional `large_guidance` parameter to `single_large_step`. Import from `agentprompt.evaluator_prompt` instead of `agentprompt.reviser_prompt`. |
| `agent/mcts.py:43-65` | Add `small_guidance: str = ""`, `large_guidance: str = ""`, `evaluator_direction: Optional[str] = None` to `MCTSNode`. |
| `agent/mcts.py:246-283` | `_create_node` accepts and stores the three fields. |
| `agent/mcts.py` (`expand_large` ~lines 460-490, `expand_small` ~lines 490-530) | After step returns, run `run_evaluator` on the new kernel and write three fields onto the new node. Pull `selected_node.{large,small}_guidance` into the step call. |
| `agent/mcts.py:616-624` | Apply soft bias as shown in Design §MCTS wiring. |
| `agent/large_loop.py`, `agent/small_loop.py` | After each iteration, call `run_evaluator` and thread the corresponding guidance into the next iteration. Parse but ignore direction tag. |
| `agent/agent_entry.py` | No change (dispatch logic unchanged). |

## What is intentionally NOT being done

- **No backward-compat shim for `reviser_prompt.py`.** Old name is dropped
  entirely; all imports update. The file is git-history-recoverable.
- **No persistent evaluator state across problems.** Each problem starts
  fresh — evaluator output lives on MCTSNode instances within one problem's
  run.
- **No evaluator before iteration 0.** The iteration-0 proposer call runs
  with `large_guidance=None` — there's no kernel yet to evaluate. The
  evaluator first runs *after* kernel_0 is produced and its output lands on
  kernel_0's MCTSNode, so iteration 1's MCTS bias has a direction tag to
  read.
- **No scalar bias.** Binary `<direction>large|small</direction>` only,
  with `None` if missing/malformed. LLMs don't calibrate scalars reliably.
  The `direction_bias` multiplier is a hyperparameter on the orchestrator
  side, not something the evaluator emits.
- **Direction tag does NOT override `small_step_limit`.** The `or` in MCTS
  `step()` is preserved — once a node hits its small_step ceiling it
  force-flips to large regardless of bias.
- **Proposer guidance is "optional context," not directive.** The
  proposer's job is still greenfield design; `large_guidance` informs but
  does not constrain.
- **No per-family direction tags.** One direction signal per evaluator
  call.
- **No schema-validating evaluator output.** Missing tags → empty string
  for guidance, `None` for direction. No retry, no penalty.
- **IR modes (IRS/IRL/IRB/IRLE/PS) ignore the direction tag.** Fixed step
  ratios by design; tag is logged only.
- **No caching of `detect_families` on `args`.** Inline call in
  `generate_skill_prompt`. AST parse is sub-millisecond; plumbing through
  three loop entry points is not worth the cost. Skill markdown is parsed
  exactly once at import (eager `_populate()`), so no per-iteration
  memoization is needed either.
- **No skill files for the long tail.** Ops below the top-39 trigger
  `_default.md` instead of any family skill. Adding a dedicated skill
  later is one line in `FAMILY_MAP` + one new `.md`.
- **`agentprompt/benchmarks/KB_prompt.py` and `agentprompt/examples/*`
  unchanged.** The skill system is upstream.
- **No `KB_TRITON_PROMPT` change.** Already minimal; the new skill prompt
  is layered upstream of it.

## Verification

1. **Detector correctness.** Run `detect_families` against representative
   archs and confirm both buckets:
   - `datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py` →
     `(["activation", "conv"], False)`
   - `datasets/KernelBench/level2/14_Gemm_Divide_Sum_Scaling.py` →
     `(["linear", "reduction"], False)`
   - `datasets/KernelBench/level2/15_ConvTranspose3d_BatchNorm_Subtract.py` →
     `(["conv", "norm", "reduction"], False)`
   - Arch using a long-tail op like `nn.PixelShuffle` →
     `has_unknown == True`
   - FIT definition (no `arc_src` passed) → `([], False)`
   Add a `pytest` or simple `__main__` block in `detect.py`.

2. **Section extractor.** In `agentprompt/skills/__init__.py`'s `__main__`,
   parse a tiny fixture with `## Design`, `## Tuning`, and a fenced code
   block containing `## Design` (should NOT match). Confirm
   `_load_section("conv", "large")` returns Design body, `"small"` returns
   Tuning, `"both"` returns both.

3. **Evaluator output parsing.** Unit test `run_evaluator` (or its
   underlying extractors) with four fixture LLM outputs:
   - Well-formed: all three tags → returns three non-empty values.
   - Missing `<direction>`: → `(small_g, large_g, None)`.
   - Missing one guidance block: → empty string for that one, others
     intact.
   - `<direction>` mid-output then again at end: → end-anchored / "last
     match" returns the final value.

4. **Smoke-render the three prompts.** Run
   `python -m agentprompt.proposer_prompt`,
   `python -m agentprompt.evaluator_prompt`, and
   `python -m agentprompt.tuner_prompt` on
   `datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py`. Confirm:
   - Base block appears exactly once in each.
   - Proposer prompt contains `conv.md ## Design` + `activation.md ## Design`.
     No Tuning content. With `large_guidance=None`, no "optional context"
     section. With a non-empty `large_guidance`, the block appears.
   - Evaluator prompt contains both Design AND Tuning sections per family,
     and a final `### Goal` describing the three-tag output contract.
   - Tuner prompt contains `conv.md ## Tuning` + `activation.md ## Tuning`.
     No Design content.
   - For an arch mixing Conv2d with `nn.PixelShuffle`, both `conv.md` and
     `_default.md` appear at the call site's altitude.
   - No `KeyError` on format substitution.

5. **MCTS bias is wired.** Run a 10-iteration MCTS trial on `tid=14`
   (`Gemm_Divide_Sum_Scaling`). Inspect `step_*_log.json`:
   - Every node except `dummy_root` has `small_guidance`, `large_guidance`,
     `evaluator_direction` populated (or `None` for direction if the LLM
     malformed its output) — including kernel_0's node, whose evaluator
     runs at the end of iteration 0.
   - When `selected_node.evaluator_direction == "large"`, the next
     expansion is large more often than 25% (sample size permitting).
   - Add a debug `logger.debug(f"p_large={p_large} direction={direction}
     selected_node_id={selected_node.node_id}")` during the trial.

6. **End-to-end on diagnostic offenders.**
   - **tid=14** (177× ratio from algebraic-shortcut rewrite): detected
     families `{linear, reduction}`. `_base.md` bans the `weight.sum`
     rewrite. After a few stuck small steps, evaluator emits
     `<direction>large</direction>` and MCTS pivots to a fresh proposal
     informed by `large_guidance`.
   - **tid=15** (0.32× ratio, ConvTranspose dominates): detected families
     `{conv, norm, reduction}`. Evaluator should emit
     `<direction>large</direction>` early, with `<large_guidance>`
     recommending a Triton conv-transpose; proposer consumes it.

7. **Proposer with vs. without `large_guidance`.** Compare two MCTS runs on
   tid=15:
   - Run A: `large_guidance` slot always empty (ablation).
   - Run B: full pipeline.
   Confirm Run B reaches a custom conv-transpose attempt fewer iterations
   into the trajectory than Run A.

8. **Tuning content is grounded, not freestyled.** For each family Tuning
   section, sanity-check against successful refinement trajectories in
   `outputs/KB-l1_AdaExplore_50/` and `outputs/KB-l2_AdaExplore_50/`. The
   knobs the section names should be knobs the agent has actually turned
   productively in past runs.

9. **Token-count check.** For five archs (one per family-count bucket:
   1, 2, 3, 4, 5+ families), compare total proposer-prompt and tuner-prompt
   lengths before and after. Goal: ≥25% reduction on the median arch.
   Evaluator prompt is *longer* than the old reviser — log its length
   separately and confirm it stays under the model's context window.

10. **IR-mode runs still complete.** Run one short IRS and one IRL trial.
    Confirm `run_evaluator` is invoked, direction tag is parsed and logged,
    and the orchestrator does NOT change its step ratio in response.

## Candidate-kernel selection — unchanged from AdaExplore

The existing AdaExplore selection scheme is kept as-is:

- **Large step** (`agent/mcts.py:341-448`, `_get_diverse_pool_for_large_step`):
  global cross-branch view — one best-correct kernel per
  `large_step` branch on the path to root, with optional softmax
  fill from off-path branches.
- **Small step** (`agent/mcts.py:490-530`, `expand_small`):
  branch-local lineage — `get_path_to_cut()` back to the nearest
  `large_step` ancestor, trimmed to the last `max_memory_round`
  (default 5) ancestors.

This explore/exploit asymmetry is the right spine and is preserved
under the evaluator-driven MCTS changes above. The guidance
threading described in §MCTS wiring composes with this selection
logic without modifying it — the evaluator's `small_guidance` /
`large_guidance` flow alongside whatever kernel pool the existing
selector produces.

---

# Plan — Add `<valid>` tag to evaluator to gate algebraic-shortcut wins

## Context

In `old_version/fixed_input/step_35_log.json` the evaluator correctly diagnosed an
algebraic shortcut — the kernel replaced a `mean` of `ConvTranspose3d` with three
`ConvTranspose2d` calls plus boundary corrections. Its `large_guidance` and
`small_guidance` both spell out the violation, and `evaluator_direction` is
`"large"`. But the kernel still:

- compiled (`compiled: true`)
- passed correctness (5/5 trials)
- ran 9.67× faster than the baseline (`fast_p`)
- got recorded as `global_best_node_id: 35` with `score: [1, 1, 9.67]`

That is, **the evaluator said "no" but the harness still counted the result**.
The score tuple `(compiled, correctness, speedup)` has no slot for "the LLM
referee thinks this cheated," so the cheat survived all the way to
`global_best_kernel_50.py`.

Fix: have the evaluator emit a 4th tag `<valid>true|false</valid>`. When
`valid=false`, collapse the node's score to `(1, 0, 0)` — same shape as a
correctness failure. That single collapse propagates cleanly through every
existing comparison: MCTS reward (via `reward` → `score` → speedup), global
best tracking (`node.score > global_best.score`), elite pool sorts in
`large_loop.py` (`calculate_score(metrics)`), small-loop best-kernel tracking,
and the candidate pool selectors. No new code paths, just one choke point that
hides cheating kernels from every downstream chooser.

Decisions (per user):

- **Invalid → score `(1, 0, 0)`.** Same handling as correctness-failed kernels.
- **Default valid when tag missing.** An LLM that forgets the tag does not
  accidentally invalidate a real win.
- **Apply everywhere — MCTS and IR-mode loops.** Cheating is cheating
  regardless of orchestrator.

## Design

### 1. Evaluator prompt — document the 4th tag

`agentprompt/evaluator_prompt.py:70-102` (the `GOAL` string). Extend the
contract to four tagged blocks. Add a short rubric for when to emit
`<valid>false</valid>`:

```
<valid>false</valid>  if the kernel reaches its measured performance via an
algebraic shortcut — i.e. one or more operators in the reference forward have
been collapsed at init time, folded into a downstream reduction, or replaced
with a substitute that does materially less arithmetic than the reference.
The "_base" skill section describes this contract; emit `false` whenever it
is violated, even if compile + correctness checks pass and the speedup looks
real.

<valid>true</valid>  in all other cases — including kernels that are slow,
compile-failed, or numerically wrong. Validity is about *whether the kernel
is doing the reference work*, not whether it is doing it well.
```

State that `<valid>true</valid>` is the default — emit one tag, missing or
malformed defaults to `true` (downstream).

### 2. Tag extractor — `agent/actions.py`

Add a sibling to `_extract_direction` (currently lines 28-31):

```python
def _extract_validity(output: str) -> Optional[bool]:
    """Last <valid>true|false</valid>. None on missing/malformed."""
    matches = re.findall(r"<valid>\s*(true|false)\s*</valid>",
                         output, re.IGNORECASE)
    return matches[-1].lower() == "true" if matches else None
```

Update `run_evaluator` (lines 33-57) to return a 4-tuple:

```python
return small_guidance, large_guidance, direction, valid
```

### 3. Default-valid policy — single helper

To honor "missing tag → treat as valid," resolve the optional bool to a
concrete bool exactly once, at the score-collapse boundary. Add to
`agent/utils.py` next to `calculate_score`:

```python
def calculate_score(metric, evaluator_valid: Optional[bool] = None):
    """(compiled, correctness, speedup). evaluator_valid=False forces
    (1, 0, 0) — same shape as a correctness failure — so cheating kernels
    are sorted/selected like incorrect ones. None and True pass through."""
    if metric is None or not metric.compiled:
        return (0, 0, 0)
    if not metric.correctness:
        return (1, 0, 0)
    if evaluator_valid is False:
        return (1, 0, 0)
    fast_p = metric.runtime_stats.get("fast_p", 0) if metric.runtime_stats else 0
    return (1, 1, fast_p)
```

The new arg is optional, so every existing call site (the IR loops, plus the
plain `calculate_score(metric)` form in `MCTSNode.score`) keeps working
unchanged. Sites that need the gate pass `evaluator_valid` explicitly.

### 4. MCTS — node field, score collapse, log serialization

`agent/mcts.py`:

- **Field** (`MCTSNode` dataclass, lines 43-72): add
  `evaluator_valid: Optional[bool] = None` after `evaluator_direction`.
- **Score property** (lines 76-83): pass the new field through:
  ```python
  self._score_cache = calculate_score(self.metrics, self.evaluator_valid)
  ```
  This is the choke point. Because `reward` (lines 87-103) is derived from
  `self.score`, MCTS backprop sees `(1, 0, 0)` → `REWARD_COMPILED_BUT_INCORRECT`
  for invalid nodes, and global-best tracking in `_create_node` (lines 289-292)
  uses `node.score` so it inherits the collapse for free. UCB1 / candidate
  pool selection / `_get_diverse_pool_for_large_step` (which filters on
  `n.score[1]`) all see the collapsed tuple — invalid nodes are
  indistinguishable from incorrect ones for every selector.
- **`_create_node`** (lines 252-295): add `evaluator_valid: Optional[bool] = None`
  parameter and forward it to the `MCTSNode(...)` constructor.
- **`expand_large` / `expand_small`** (lines 462-526, 528-592): unpack the
  4-tuple from `run_evaluator` and pass `evaluator_valid=valid` into the
  `_create_node` call.
- **`_save_step_log`** (lines 755-786): add
  `"evaluator_valid": node.evaluator_valid,` to `log_dict` next to the
  existing `"evaluator_direction"` line, so trajectory inspection shows when
  a kernel was invalidated and what its raw score would have been.

Note: `mcts_utils.py:178` (`max(... key=lambda n: n.score)`) keeps working —
the score is collapsed at the property, not at the call site.

### 5. IR-mode loops — large_loop and small_loop

Both loops already destructure 3-tuples from `run_evaluator`. Update the
unpacks and pass the validity flag into every `calculate_score` and
best-kernel comparison in scope.

`agent/large_loop.py`:

- 3-tuple → 4-tuple at line ~144. Bind `valid` (alongside `direction`).
- Step log (lines ~159-180): include `"evaluator_valid": valid` and
  `"score": calculate_score(proposal_metrics, valid)`.
- Elite pool sort (lines ~182-211, `sorted_data = sorted(...)`): the elite
  pool is keyed on `calculate_score(x[1])`; thread the per-kernel validity
  alongside the metrics so the sort key passes `evaluator_valid`. Concretely,
  store a parallel `validity_pool: list[Optional[bool]]` next to
  `kernel_pool` / `metrics_pool` / `proposal_ids`, and zip it into the
  sort tuple.

`agent/small_loop.py`:

- All three `run_evaluator` calls (lines ~170, ~195, ~216): 3-tuple → 4-tuple.
- Best-kernel tracking (lines ~226-230): the comparison
  `score > local_best_score` already uses `calculate_score(tuned_metrics)`
  — pass `valid` alongside (`calculate_score(tuned_metrics, valid)`). An
  invalid tuned kernel collapses to `(1, 0, 0)` and cannot become local best.

### 6. agent_entry.py / global best save

No change. `global_best_kernel_*.py` is whatever lives at
`self.global_best_node` after the run; that selection now consults the
collapsed score automatically.

## Files to modify

| Path | Change |
|---|---|
| `agentprompt/evaluator_prompt.py` | Extend `GOAL` string to document `<valid>true\|false</valid>` and the default-true rule. |
| `agent/actions.py` | Add `_extract_validity`. Change `run_evaluator` return to 4-tuple. |
| `agent/utils.py` | Add optional `evaluator_valid` arg to `calculate_score`; collapse to `(1,0,0)` when `False`. |
| `agent/mcts.py` | New `MCTSNode.evaluator_valid` field; thread through `_create_node`, `expand_large`, `expand_small`; pass to `calculate_score` in the `score` property; emit in `_save_step_log`. |
| `agent/large_loop.py` | Unpack 4-tuple; track `validity_pool`; pass validity into `calculate_score` + step log. |
| `agent/small_loop.py` | Unpack 4-tuple at all three `run_evaluator` sites; pass validity into best-kernel `calculate_score`. |

No file deletions, no rename, no schema migration — old logs without the
field remain readable, new logs gain one field.

## What is intentionally NOT being done

- **No new reward/score code path.** The collapse goes through the existing
  `(1, 0, 0)` shape, so every selector that already handled correctness
  failures handles invalid kernels identically.
- **No retroactive re-evaluation of existing trajectories.** Old runs in
  `outputs/` retain their old scores; only new runs gate on `<valid>`.
- **No "soft" invalidation / penalty scalar.** Binary flag, mirroring the
  binary direction tag. Easier for the LLM to emit reliably.
- **No skill-level changes.** `_base.md`'s algebraic-shortcut ban is still
  the contract the evaluator enforces; we're just giving the evaluator a
  way to flag a violation that survived compile+correctness.
- **No retry on missing tag.** Default to `valid=true` and move on.

## Verification

1. **Extractor unit test.** Feed `_extract_validity` four fixtures: well-formed
   `true`, well-formed `false`, missing tag, two tags (last wins). Expect
   `True`, `False`, `None`, last-match.
2. **Score collapse.** Build a `KernelExecResult(compiled=True,
   correctness=True, runtime_stats={"fast_p": 9.67})`. Confirm
   `calculate_score(m)` → `(1, 1, 9.67)`, `calculate_score(m, True)` →
   `(1, 1, 9.67)`, `calculate_score(m, None)` → `(1, 1, 9.67)`,
   `calculate_score(m, False)` → `(1, 0, 0)`.
3. **MCTSNode collapse end-to-end.** Construct an `MCTSNode` mirroring
   step_35's metrics with `evaluator_valid=False`. Confirm `node.score ==
   (1, 0, 0)` and `node.reward == REWARD_COMPILED_BUT_INCORRECT`.
4. **Replay step_35.** Hand-feed step_35's `large_guidance` text plus
   `<valid>false</valid>` into the new extractors and assert the parsed
   tuple. (Sanity that the prompt change is observable by the parser.)
5. **MCTS smoke run on tid=15** (`ConvTranspose3d_BatchNorm_Subtract`, the
   trajectory family that produced the step_35 incident). Run a 40-step
   trial. Inspect `step_*_log.json`:
   - Every node has `evaluator_valid` populated (`true`, `false`, or `null`).
   - At least one invalid kernel appears in the log with high raw `fast_p`
     but score `[1, 0, 0]`, and `global_best_node_id` does NOT point to it.
6. **IR-mode smoke (one IRS run).** Confirm the step log shows
   `evaluator_valid` and the elite pool sort excludes any
   `evaluator_valid=false` entry — i.e. an invalid kernel does not appear
   ahead of a slower-but-valid kernel in `recent`/`elite` context for the
   next iteration.
7. **Backward read.** Open an old `step_*_log.json` from
   `outputs/KB-l*_AdaExplore_50/` — confirm nothing in the new code path
   chokes on a missing `evaluator_valid` field (the IR/MCTS code only
   writes it; nothing reads it back from old logs).
