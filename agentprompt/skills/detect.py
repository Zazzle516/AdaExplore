import ast
from agentprompt.skills.registry import FAMILY_MAP

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


if __name__ == "__main__":
    import os

    REPO_TOP_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    def _check(rel_path, expected):
        with open(os.path.join(REPO_TOP_PATH, rel_path)) as fh:
            got = detect_families(fh.read())
        status = "OK" if got == expected else "FAIL"
        print(f"[{status}] {rel_path}: got={got} expected={expected}")

    _check(
        "datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py",
        (["activation", "conv"], False),
    )
    _check(
        "datasets/KernelBench/level2/14_Gemm_Divide_Sum_Scaling.py",
        (["linear", "reduction"], False),
    )
    _check(
        "datasets/KernelBench/level2/15_ConvTranspose3d_BatchNorm_Subtract.py",
        (["conv", "norm", "reduction"], False),
    )

    # Long-tail op -> has_unknown_ops fires.
    pixelshuffle_src = (
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.ps = nn.PixelShuffle(2)\n"
        "    def forward(self, x):\n"
        "        return self.ps(x)\n"
    )
    got = detect_families(pixelshuffle_src)
    print(f"[{'OK' if got[1] else 'FAIL'}] PixelShuffle unknown bit: got={got}")

    # No arch (e.g. FIT definition) -> empty.
    got = detect_families(None)
    print(f"[{'OK' if got == ([], False) else 'FAIL'}] None arch: got={got}")
