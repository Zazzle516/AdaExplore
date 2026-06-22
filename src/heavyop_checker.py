"""Runtime verification of the executed forward path.

The eval harness runs models at PyTorch's default ``training=True`` and never
calls ``.eval()``. An agent can exploit this by parking a ``@triton.jit``
heavy-op "rewrite" in a branch that never executes (e.g. the ``else:`` arm of
``if self.training:``) while the real PyTorch heavy op runs on the live path --
correctness passes on the PyTorch path and timing measures PyTorch, so the kernel
*looks* like a heavy-op rewrite while being dead code.

``_HeavyOpRecorder`` is a ``TorchDispatchMode`` that records which aten ops
actually dispatched during a forward. Triton kernel launches do **not** go
through ``__torch_dispatch__``, so a heavy aten op appearing here means the
reference op truly ran in PyTorch/cuDNN rather than in a custom kernel.
"""

from torch.utils._python_dispatch import TorchDispatchMode

# aten heavy-op allowlist. Conv / ConvTranspose dispatch through aten::convolution
# / aten::_convolution (and the functional conv*/conv_transpose* entry points);
# Linear / matmul through addmm / mm / bmm / linear / matmul / einsum.
_HEAVY_ATEN = {
    "aten::convolution", "aten::_convolution",
    "aten::conv1d", "aten::conv2d", "aten::conv3d",
    "aten::conv_transpose1d", "aten::conv_transpose2d", "aten::conv_transpose3d",
    "aten::addmm", "aten::mm", "aten::bmm",
    "aten::linear", "aten::matmul", "aten::einsum",
}


def _normalize_op_name(func) -> str:
    """Normalize a dispatched op to the ``namespace::op`` form.

    ``str(func)`` looks like ``aten.convolution.default`` / ``aten.addmm.default``;
    drop the trailing overload and swap the ``.`` namespace separator for ``::``.
    """
    name = str(func)
    parts = name.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}::{parts[1]}"
    return f"aten::{parts[0]}"


class _HeavyOpRecorder(TorchDispatchMode):
    """Record dispatched aten op names during a forward pass.

    Use as a context manager around an *already-needed* forward (no extra pass):

        with _HeavyOpRecorder() as rec:
            out = model(*inputs)
        heavy = rec.heavy_ops()   # subset of _HEAVY_ATEN that actually ran
    """

    def __init__(self):
        super().__init__()
        self.ops = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            self.ops.add(_normalize_op_name(func))
        except Exception:
            pass
        return func(*args, **(kwargs or {}))

    def heavy_ops(self) -> set:
        """Recorded ops that are on the heavy-op allowlist."""
        return {op for op in self.ops if op in _HEAVY_ATEN}
