import os

EXPERIENCE_GUIDANCE = """## Common Pitfalls (from previous optimization attempts)

The following are frequently observed failure patterns from prior kernel optimization runs. **Avoid** these patterns in your implementation:

{experience_guidance}

"""

HARDWARE_INFORMATION = """## Hardware Information

Here is some information about the underlying hardware that you should keep in mind:

- The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.

"""

OPTIMIZATION_RULES = """## Optimization Rules

The reference architecture is a *contract*: every operator in the reference forward must
execute at runtime on the actual input tensor. Your job is to make those operators run
faster — not to make some of them disappear.

### Forbidden: graph-level algebraic shortcuts

Do **not** replace a heavy operator with a cheaper one by exploiting identities derived
from downstream reductions. Specifically forbidden:

- Folding a post-`Conv*`/`ConvTranspose*` global reduction (`mean`/`sum` over spatial
  dims, `mean`/`sum` over a depth/channel slice) into a per-element rewrite of the
  weight tensor at init time, so that forward executes a small GEMM instead of the
  convolution.
- Replacing `linear(x).sum(dim=...)` with `x @ weight.sum(dim=...)` (or any analogous
  identity that turns a `(B, in) x (in, out)` matmul into a `(B, in)` matvec by
  precomputing a column/row sum of the weight).
- Any rewrite that, on inputs of the documented shape, turns the work of a `Conv2d` /
  `ConvTranspose2d` / `ConvTranspose3d` / `matmul` / `Linear` into an operator with
  strictly fewer FLOPs by collapsing it with a downstream reduction.

These rewrites pass the loose `atol=rtol=5e-2` correctness check but are not the
optimization target — the comparison baseline is a strong inference engine that runs the
graph as written.

### Permitted: kernel-level fusion and standard constant-folding

You **may**:

- Fuse adjacent operators into a single Triton kernel (e.g., `bias + activation`,
  `LayerNorm + GELU`, `conv-bias + ReLU`).
- Reorder elementwise ops when the reordering does not change which heavy op runs
  (e.g., merge two scalar subtractions into one).
- Apply standard inference-time folds: BatchNorm-into-Conv (eval mode), scalar
  multipliers absorbed into adjacent affine ops, identity activations elided.
- Use numerical-stability rewrites: online softmax, log-sum-exp shift, RMS-norm via
  `rsqrt`.
- Cache parameter-derived tensors that are *the same shape* as the original parameter
  (not collapsed by a reduction).

### Authorized: custom Conv / ConvTranspose kernels

`nn.Conv2d`, `nn.ConvTranspose2d`, and `nn.ConvTranspose3d` are **not** off-limits.
When the torch convolution dominates runtime — symptom: `fast_p < 0.8` while the
epilogue is already fused — the right next step is to **replace the conv with a custom
Triton (or CUDA-cpp via `torch.utils.cpp_extension`) kernel**, not to keep tuning the
tail. Reasonable starting points:

- Conv2d: im2col-style indexing into a tiled GEMM, one program per `(N, OC tile, H tile,
  W tile)`.
- ConvTranspose2d/3d: scatter-add of the input x kernel outer product into the output
  grid, with stride / padding / output_padding handled in the index math.

Custom conv kernels are large but supported by the framework. Do not silently assume the
conv is the immovable part of the graph — it is exactly the part with the most headroom.

"""

def generate_optimization_rules_prompt() -> str:
    return OPTIMIZATION_RULES

def generate_experience_guidance_prompt(experience_guidance_path: str, threshold: int=2) -> str:
    """
    threshold: the threshold of the experience guidance, if the experience guidance is less than the threshold, it will not be included in the experience guidance
    """
    if experience_guidance_path is None or not os.path.exists(experience_guidance_path):
        return ""
    with open(experience_guidance_path, "r") as f:
        lines = f.readlines()
        experience_guidance_content = []
        for line in lines:
            if float(line.strip().split("||")[1]) < threshold:
                continue
            experience_guidance_content.append(line.strip().split("||")[0])
        experience_guidance_content = "\n".join(experience_guidance_content)
    return EXPERIENCE_GUIDANCE.format(experience_guidance=experience_guidance_content)

def generate_hardware_information_prompt(gpu_name: str, gpu_architecture: str) -> str:
    if gpu_name is None or gpu_architecture is None:
        return ""
    return HARDWARE_INFORMATION.format(gpu_name=gpu_name, gpu_architecture=gpu_architecture)