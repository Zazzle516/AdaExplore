为什么 case 9 完全失败——根因是具体明确的

`2_9`（`Matmul_Subtract_Multiply_ReLU`）的 50 次尝试全部以完全相同的方式失败：

```text
compiled: true
correctness: false
runtime_error: subprocess.CalledProcessError: '/usr/bin/gcc' ...
              '__triton_launcher.cpython-312-x86_64-linux-gnu.so' ... returned non-zero exit status 1
```

kernel 的 PTX 编译通过了，但 Triton 的 C launcher（`main.c`）构建失败。关键证据就在 kernel signature 本身——对比一个成功工作的 kernel 和 case 9 的 kernel：

`2_2` 的 global best（与 case 9 类似，也有标量操作形态：`* scaling_factor`）——`agentprompt/skills/linear.md/.../global_best_kernel_50.py:33`：

```python
SCALE: tl.constexpr, INV_SCALE: tl.constexpr,
...
float(self.scaling_factor), float(1.0 / self.scaling_factor),  # baked at JIT time
```

而 `2_9` 的每一步，包括 global best，例如 `step_1.py:24`、`step_32.py:11`、`step_50.py:19`：

```python
sub_val, mul_val,                # no type, no constexpr → runtime args
...
float(sub_val), float(mul_val),  # plain Python floats passed at launch
```

当 Triton 看到一个未标注类型的 runtime float 标量时，它会生成 launcher 的 `main.c` 代码；而这两个参数的 ABI specialization 触发了本地 gcc 构建失败（返回非零退出码 1）。agent 尝试过的所有变体——autotune、no-autotune（`step_32.py`）、预转置 weight、persistent CTA——都保留了相同的 `sub_val, mul_val` signature 模式，所以每个变体都撞上了同一堵墙。

为什么 agent 一直没有跳出来——evaluator/skills 漏掉了这个 failure mode

1. evaluator 把失败误诊为环境问题。来自 `step_1_log.json` 的 `small_guidance`：

   > "the failure is a subprocess gcc error during launcher compilation, which is environmental rather than kernel-structural."

   这个判断随后传播了 50 步。统计为：`evaluator_direction = {None: 1, 'small': 48, 'large': 2}`——50 次 evaluator 调用里只有 2 次转向 large；即便那两次 large guidance（第 16 步和第 41 步）也只是说“保持 skeleton，尝试 persistent-CTA / split-K”。没有任何一次指出“把 `sub_val` 声明为 `tl.constexpr`（或 `tl.float32`）”。

2. skill files 没有覆盖这个问题。`linear.md` 提到“把尾部 scalar multiply 吸收到 epilogue 中”，但 `agentprompt/skills/*` 里没有类似下面的规则：

   > 传给 Triton kernel 的 Python-float 标量必须标为 `tl.constexpr`（或封装为 0-d tensor）——否则 launcher specialization 可能失败。

   `_base.md` 覆盖的是代数 shortcut；`_default.md` 只在 `BLOCK_SIZE` 这类参数上提到 `tl.constexpr`，没有覆盖 runtime scalars。

3. MCTS 卡在了 small-step 洼地里。50 个 children 中，39 个是 `small_step`，只有 11 个是 `large_step`（尽管 `p_large=0.2`、`small_step_limit=2`、`direction_bias=2.5`）。由于几乎每个 node 上 evaluator 都给出 `valid=True` 且 `evaluator_direction="small"`，每条分支都继续围绕同一个坏掉的 signature 做局部调参。score 全部坍缩为 `(1, 0, 0)` → 没有 reward gradient → MCTS UCB1 基本变成随机 → 没有压力促使它跳到结构上不同的 signature。

因此，`2_9` 的完全失败是一个缺失 prompt rule 加 evaluator 误诊的问题，不是环境 bug，也不是 `<valid>` 已经修复的代数 shortcut 类问题。

为什么即使规避了代数 shortcut，和 TRT 的差距仍然很大

观察那些能够端到端跑通的 cases：

1. 你是在和 cuDNN/cuBLAS 调优过的 tactics 竞争。TRT 显式启用了 `CUBLAS`、`CUBLAS_LT`、`CUDNN`、`EDGE_MASK_CONVOLUTIONS`（`TRT_Baseline/config.yaml`）。agent 写出的 Triton kernels 是在一些形状上重新实现 GEMM/conv，例如 `1024×8192×8192`、batched 3D convs；这些形状上 cuBLAS/cuDNN 的 tensor-core kernels 已经非常接近峰值性能。如果没有针对 `tl.dot`、tensor cores 和正确的 swizzle，一个手写 fp32 Triton GEMM 往往只能达到 cuBLAS 的约 30%–70%。这解释了 `fast_p ≈ 0.4–0.8` 的一组结果（`2_1`、`2_3`、`2_7`、`2_8`）。

2. 只有当 reference 里有大量可融合的“冗余空间”时，agent 才能赢。agent 超过 TRT 的 cases（`2_2 = 1.60×`、`2_4 = 1.29×`、`2_5 = 1.28×`、`2_6 = 2.69×`）都是多算子链，优化杠杆是 epilogue fusion 加避免中间 tensor 物化。那些被单个 compute-bound primitive 主导的 cases（`2_1` 的 Conv2D、`2_3` 的 ConvTranspose3d、`2_7` 的 Conv3D 加大量 activations、`2_8` 的 Conv3d 加 global avg pool）基本上碰不到 TRT 的 library kernels。

3. exploration budget 太浅。`total_steps: 50`、`small_step_limit: 2`、`pool_size: 5`、`num_processes: 1`。由于每个 node 的 small-step 被限制为最多 2 次，tuner 几乎只能扫一个 knob 方向，MCTS 就会强制 redesign。而 `p_large=0.2` 加 `direction_bias=2.5` 意味着，除非 evaluator 准确打出 large tag，否则这个 bias 很少真正起作用；但 evaluator 目前做得并不稳定（见第 1 点）。

4. skills 更偏向正确性，而不是追平 library kernels。`linear.md` 的 Tuning 里提到了 `BLOCK_M/N/K`、`tl.dot`、`num_warps`、`num_stages`、`GROUP_M`，但没有说明：

   > 在 4090 的 fp32 GEMM 上，Triton 变体很难稳定超过 cuBLAS；应该考虑用 `torch.matmul` 做主计算，只用 Triton 写 epilogue。

   缺少这种更高层的策略引导，proposer 会不断重新 roll 自己的 GEMM，然后持续落后。

我会优先修改什么

1. 向 `_base.md`（或 `_default.md` 的 Design）添加一条 “scalar-args” 规则：

   > 任何来自 reference parameter 的 Python-float，如果要作为 kernel 输入传入，必须作为 `tl.constexpr` 传递（优先选择——这会让 Triton 把值 bake 到 PTX 中），或者封装成 0-d CUDA tensor。不要把它作为未标注类型的 positional arg 传入——Triton 对 raw Python floats 的 launcher specialization 很脆弱。

   这一条单独就能解锁 `2_9`。

2. 强化 evaluator 的 failure-mode rubric。当 `metrics.compiled == True`，但每次 retry 都在构建 `__triton_launcher.*.so` 时遇到相同的 `subprocess.CalledProcessError`，evaluator 应该把它视为结构性的 signature 问题，而不是 “environmental/transient”，并输出：

   ```xml
   <direction>large</direction>
   ```

   现在它输出了相反判断，所以 MCTS 一直没有跳出来。

3. 在 `linear.md` 的 Design 中增加一条 “use cuBLAS for the heavy compute” 的高层提示：当 GEMM 主导 wall-clock，且从零手写 Triton GEMM 不太可能追上时，让 agent 使用 `torch.matmul` 做重计算，再写 Triton epilogue。这样不需要增加更多 compute budget，也能缩小 `2_1`、`2_3`、`2_7` 与 TRT 的大部分差距。
