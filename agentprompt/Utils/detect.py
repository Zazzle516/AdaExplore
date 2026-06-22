import ast
import io
import tokenize
from dataclasses import dataclass
from agentprompt.Utils.registry import FAMILY_MAP

# Plumbing / shape ops / tensor factories / Python builtins. These are NOT
# evidence of a missing skill file -- ignoring them keeps `has_unknown_ops`
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

    has_unknown_ops fires only when an `nn.*` / `F.*` attribute call is
    unrecognized. Bare names that look like Python builtins / tensor plumbing
    are ignored.
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
        # Unknown call. Flip the flag only if it looks like a real nn op --
        # i.e. an `nn.*` / `F.*` attribute call we don't recognize. Bare
        # unknown names are conservatively ignored (likely user helpers).
        if is_nn_call:
            has_unknown = True
    return sorted(found), has_unknown


# --- Shortcut-risk detection ------------------------------------------------
#
# A "shortcut risk" is a chain in the reference forward where a heavy op
# (Conv*, Linear, matmul) feeds -- possibly through pointwise / shape-only ops
# -- into a *linear* reduction (sum / mean / avg-pool). By linearity such a
# chain can be collapsed into a smaller GEMM, which is the algebraic shortcut
# forbidden by `_base.md`. We surface the chain so the evaluator can demand the
# kernel actually materialize the heavy op's full output.

# Heavy ops -- attribute/builtin/functional names that perform a full MAC pass.
_HEAVY_CONV = {
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
}
_HEAVY_LINEAR = {"Linear", "linear", "matmul", "bmm", "addmm", "einsum"}
_HEAVY = _HEAVY_CONV | _HEAVY_LINEAR

# Linear reductions only -- max / argmax / softmax are NOT collapsible this way.
_REDUCTION = {
    "sum", "mean",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d",
    "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
}

# Ops transparent to the chain: activations, shape/plumbing ops, clamp, dropout.
_ACTIVATION_NAMES = {k for k, v in FAMILY_MAP.items() if v == "activation"}
_TRANSPARENT = (
    _ACTIVATION_NAMES
    | BUILTIN_IGNORE
    | {"clamp", "clamp_", "dropout", "Dropout", "Identity", "hardtanh",
       "relu_", "Hardtanh", "softplus", "leaky_relu", "elu", "selu"}
)


@dataclass
class ShortcutChain:
    """A heavy-op -> linear-reduction chain found in the reference forward."""
    heavy_op: str          # e.g. "Conv2d", "Linear", "matmul"
    reduction: str         # e.g. "mean", "AvgPool2d", "sum"
    shape_descriptor: str  # expected intermediate (heavy-op output) shape


def _resolved_call_name(node: ast.Call, init_map: dict):
    """Return (name, ctor_node) for a call.

    `name` is the underlying op name (resolving `self.attr` through __init__);
    `ctor_node` is the constructor Call from __init__ when the call is a module
    attribute, else None.
    """
    f = node.func
    if isinstance(f, ast.Attribute):
        parent = f.value.id if isinstance(f.value, ast.Name) else None
        if parent == "self" and f.attr in init_map:
            ctor_name, ctor_node = init_map[f.attr]
            return ctor_name, ctor_node
        # nn.X / F.x / torch.x / tensor-method `x.mean()` -> the attribute name.
        return f.attr, None
    if isinstance(f, ast.Name):
        return f.id, None
    return None, None


def _classify(name: str):
    """Map an op name to a chain category."""
    if name is None:
        return "opaque"
    if name in _HEAVY:
        return "heavy"
    if name in _REDUCTION:
        return "reduction"
    if name in _TRANSPARENT:
        return "transparent"
    return "opaque"


def _const_arg(ctor_node: ast.Call, pos: int, *kw_names):
    """Best-effort read of a constructor argument as a display string."""
    if ctor_node is not None:
        if len(ctor_node.args) > pos:
            arg = ctor_node.args[pos]
            if isinstance(arg, ast.Constant):
                return str(arg.value)
        for kw in ctor_node.keywords:
            if kw.arg in kw_names and isinstance(kw.value, ast.Constant):
                return str(kw.value.value)
    return "?"


def _shape_descriptor(heavy_name: str, ctor_node) -> str:
    if heavy_name in _HEAVY_CONV:
        out_ch = _const_arg(ctor_node, 1, "out_channels")
        return (f"(N, {out_ch}, H_out, W_out)  "
                "[channels from constructor; H/W not statically derivable]")
    if heavy_name == "Linear":
        out_f = _const_arg(ctor_node, 1, "out_features")
        return f"(N, ..., {out_f})  [out_features from constructor]"
    return "shape not statically derivable (dynamic matmul / functional op)"


def _ordered_ops(expr, init_map):
    """Yield (category, name, ctor_node) in evaluation order for an expression.

    Children (call arguments, operands) are visited before the enclosing node
    so nested function application -- e.g. mean(relu(conv(x))) -- is emitted
    innermost-first: conv, relu, mean.
    """
    for child in ast.iter_child_nodes(expr):
        yield from _ordered_ops(child, init_map)
    if isinstance(expr, ast.Call):
        name, ctor_node = _resolved_call_name(expr, init_map)
        yield (_classify(name), name, ctor_node)
    elif isinstance(expr, (ast.BinOp, ast.UnaryOp)):
        # Pointwise arithmetic (bias add, scaling, division) is transparent.
        yield ("transparent", "<arith>", None)


def _find_model_class(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            return node
    # Fallback: first class subclassing something named *Module.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node
    return None


def _build_init_map(cls: ast.ClassDef) -> dict:
    """Map `self.X` attribute names to (constructor_name, constructor_node)."""
    init_map = {}
    init_fn = next(
        (n for n in cls.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "__init__"),
        None,
    )
    if init_fn is None:
        return init_map
    for stmt in ast.walk(init_fn):
        if not isinstance(stmt, ast.Assign):
            continue
        if not (isinstance(stmt.value, ast.Call)):
            continue
        ctor = stmt.value.func
        ctor_name = ctor.attr if isinstance(ctor, ast.Attribute) else (
            ctor.id if isinstance(ctor, ast.Name) else None)
        if ctor_name is None:
            continue
        for tgt in stmt.targets:
            if (isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"):
                init_map[tgt.attr] = (ctor_name, stmt.value)
    return init_map


def detect_shortcut_risk(arch_src: str) -> list[ShortcutChain]:
    """Find heavy-op -> linear-reduction chains in the reference forward.

    Returns one ShortcutChain per detected chain, or [] when the source is
    missing/malformed, has no Model.forward, or contains no such chain.
    """
    if not arch_src or not isinstance(arch_src, str):
        return []
    try:
        tree = ast.parse(arch_src)
    except SyntaxError:
        return []

    cls = _find_model_class(tree)
    if cls is None:
        return []
    init_map = _build_init_map(cls)
    forward_fn = next(
        (n for n in cls.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "forward"),
        None,
    )
    if forward_fn is None:
        return []

    # Flatten the forward into an evaluation-ordered op stream, statement by
    # statement, then scan for heavy -> (transparent*) -> reduction.
    chains = []
    pending = None  # (heavy_name, ctor_node)
    for stmt in forward_fn.body:
        for cat, name, ctor_node in _ordered_ops(stmt, init_map):
            if cat == "heavy":
                pending = (name, ctor_node)
            elif cat == "transparent":
                continue
            elif cat == "reduction":
                if pending is not None:
                    heavy_name, heavy_ctor = pending
                    chains.append(ShortcutChain(
                        heavy_op=heavy_name,
                        reduction=name,
                        shape_descriptor=_shape_descriptor(heavy_name, heavy_ctor),
                    ))
                    pending = None
            else:  # opaque -> breaks the chain
                pending = None
    return chains


# --- Dead-branch heavy-op rewrite detection ---------------------------------
#
# Paired with runtime verification (src/heavyop_checker.py): the runtime signal
# says the reference heavy op still dispatched through aten inside ModelNew (it
# ran in PyTorch, not Triton); this static signal says ModelNew nonetheless
# *built* a custom Triton replacement of that heavy op. Together they catch the
# cheat where a `@triton.jit` conv is parked in a dead branch (e.g. the `else:`
# of `if self.training:`) while the real PyTorch conv runs on the live path.
#
# Discriminator (calibrated on outputs/KB-l2_AdaExplore_50/2_15): legit kept-op
# kernels (steps 1/3/7) only build a BN/sub Triton kernel and never read the
# heavy module's weight; cheats (steps 2/4/5/6/9) extract `self.<conv>.weight`
# to feed a custom conv. Reading the heavy module's `.weight` is therefore the
# tell of a built conv replacement -- the Triton-kernel count is not (step 7 is
# legit yet defines two `@triton.jit` kernels).


def _find_class_named(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    # Fallback: first class definition in the module.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node
    return None


def _ref_heavy_op_names(arch_src: str) -> set:
    """Heavy-op names used in the reference Model.forward (resolved via __init__)."""
    if not arch_src or not isinstance(arch_src, str):
        return set()
    try:
        tree = ast.parse(arch_src)
    except SyntaxError:
        return set()
    cls = _find_model_class(tree)
    if cls is None:
        return set()
    init_map = _build_init_map(cls)
    forward_fn = next(
        (n for n in cls.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "forward"),
        None,
    )
    if forward_fn is None:
        return set()
    names = set()
    for stmt in forward_fn.body:
        for cat, name, _ in _ordered_ops(stmt, init_map):
            if cat == "heavy":
                names.add(name)
    return names


def _defines_triton_kernel(tree: ast.Module) -> bool:
    """True if the module defines/launches a custom Triton kernel.

    Detects a `@triton.jit`/`@jit`-decorated function, any `tl.*` attribute use,
    or a `fn[grid](...)` launch (a Call whose func is a Subscript).
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                d = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(d, ast.Attribute) and d.attr == "jit":
                    return True
                if isinstance(d, ast.Name) and d.id == "jit":
                    return True
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "tl":
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            return True
    return False


def _reads_heavy_module_weight(cls: ast.ClassDef, heavy_attrs: set) -> bool:
    """True if the class reads `self.<heavy_attr>.weight` for a heavy module.

    `self.conv_transpose.weight` parses as Attribute(attr='weight',
    value=Attribute(attr='conv_transpose', value=Name('self'))). Extracting the
    heavy module's weight is the signature of building a custom heavy-op kernel
    (vs. legit kernels that just call `self.conv_transpose(x)`).
    """
    for node in ast.walk(cls):
        if (isinstance(node, ast.Attribute)
                and node.attr in ("weight", "bias")
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr in heavy_attrs):
            return True
    return False


def replaced_heavy_op_built(kernel_src: str, arch_src: str) -> bool:
    """Did ModelNew *build* a custom Triton implementation of the reference heavy op?

    Branch-agnostic intent signal (the caller ANDs it with runtime verification).
    Returns True iff: the reference uses a heavy op (Conv*/ConvTranspose*/Linear),
    the kernel source defines/launches a custom Triton kernel, and ModelNew reads
    the heavy module's `.weight`/`.bias` (the tell of a built heavy-op replacement,
    not merely an `nn.*`/`F.*` call). Fail-safe to False on a missing heavy op, no
    Triton kernel, no weight extraction, or any parse error.
    """
    try:
        ref_heavy = _ref_heavy_op_names(arch_src)
        if not ref_heavy:
            return False
        if not kernel_src or not isinstance(kernel_src, str):
            return False
        try:
            tree = ast.parse(kernel_src)
        except SyntaxError:
            return False
        cls = _find_class_named(tree, "ModelNew")
        if cls is None:
            return False
        init_map = _build_init_map(cls)
        heavy_attrs = {attr for attr, (ctor, _) in init_map.items() if ctor in _HEAVY}
        if not heavy_attrs:
            return False
        return _defines_triton_kernel(tree) and _reads_heavy_module_weight(cls, heavy_attrs)
    except Exception:
        return False


# --- Kernel comment / docstring stripping -----------------------------------
#
# LLM-generated kernels frequently carry comments like
# ``# Fused conv+mean shortcut -- mathematically equivalent`` or module
# docstrings that assert correctness. Those assertions are exactly what an
# adversarial reviewer must not take on faith, so we remove them and force the
# reviewer to reason from the code itself.


def _strip_comments(src: str) -> str:
    """Drop COMMENT tokens via the tokenizer, preserving everything else.

    On a tokenizer error (e.g. unbalanced source) return the input unchanged.
    """
    try:
        tokens = tokenize.tokenize(io.BytesIO(src.encode("utf-8")).readline)
        kept = [tok for tok in tokens if tok.type != tokenize.COMMENT]
        return tokenize.untokenize(kept).decode("utf-8")
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return src


def _strip_docstrings(src: str) -> str:
    """Remove module / class / function docstrings.

    A docstring is an ``ast.Expr`` whose value is a string ``Constant`` sitting
    at the head of a body. Assignments such as ``cpp_source = \"\"\"...\"\"\"``
    are ``ast.Assign`` and are left alone. On a parse error return the input
    unchanged.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src

    # Collect (lineno, end_lineno) ranges of docstring statements to drop.
    drop_ranges = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        head = body[0]
        if (isinstance(head, ast.Expr)
                and isinstance(head.value, ast.Constant)
                and isinstance(head.value.value, str)):
            drop_ranges.append((head.lineno, head.end_lineno))

    if not drop_ranges:
        return src

    drop_lines = set()
    for start, end in drop_ranges:
        drop_lines.update(range(start, (end or start) + 1))

    lines = src.splitlines(keepends=True)
    kept = [ln for i, ln in enumerate(lines, start=1) if i not in drop_lines]
    return "".join(kept)


def strip_comments_and_docstrings(src: str) -> str:
    """Return ``src`` with comments and docstrings removed.

    Falls back to returning the original source unchanged if either pass hits
    a parse/tokenize error, so a syntactically-broken kernel is never lost.
    """
    if not src or not isinstance(src, str):
        return src
    return _strip_docstrings(_strip_comments(src))
