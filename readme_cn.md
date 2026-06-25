# AdaExplore

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.16625-b31b1b.svg)](https://arxiv.org/abs/2604.16625)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://stiglidu.github.io/AdaExplore/)

> **Weihua Du, Jingming Zhuo, Yixin Dong, Andre He, Weiwei Sun, Zeyu Zheng, Manupa Karunaratne, Ivan Fox, Tim Dettmers, Tianqi Chen, Yiming Yang, Sean Welleck**
> [《AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation》（2026）](https://arxiv.org/abs/2604.16625)

AdaExplore 是一个面向 **LLM 驱动的 GPU kernel 工程** 的研究代码库。该框架围绕两个互补阶段构建：

* **Adapt**：合成训练任务，收集执行失败，并将反复出现的错误模式蒸馏为可复用的跨任务技能记忆。
* **Explore**：使用保持多样性的树搜索优化每个目标 kernel，在局部细化和更大范围的结构重生成之间交替进行。

本仓库还包含评测框架、合成任务生成流水线、用于多 GPU 或集群环境的可选远程评测服务，以及 TensorRT FP32 baseline，使报告的加速比以强推理引擎而非 torch eager 作为对照。

<p align="center">
  <img src="assets/overview.png" alt="AdaExplore overview" />
</p>

## 概览

AdaExplore 使用两阶段工作流解决 kernel 生成问题。**Adapt** 阶段通过将合成任务中反复出现的编译 / 运行时失败转化为可复用的约束规则来提升正确性，并将这些规则作为技能记忆注入后续 prompt。**Explore** 阶段通过将候选 kernel 组织为搜索树，在短视野的局部编辑与更大范围的结构重建之间取得平衡，从而提升优化质量。

在实践中，本仓库是 AdaExplore 的简化开源版本，其中：

* `synthesis/` 生成用于 adaptation 的 playground；
* `skill_memory/` 构建并更新可复用记忆；
* `agent/` 运行用于 benchmark 或任务时优化的搜索流程；
* `agentprompt/` 组合 proposer / evaluator prompt，包括算子感知技能、反捷径契约、三种 review 模式；
* `src/` 包含评测框架、Triton 错误解析器、重型算子运行时检查器，以及 Nsight profiling 流水线；
* `TRT_Baseline/` 构建并计时 TensorRT FP32 参考 baseline。

## 安装

### 1. 创建 Python 环境并安装依赖

```bash
conda env create -f environment.yml
conda activate adaexplore
```

这会创建一个名为 `adaexplore` 的 Python 3.12 环境，并安装 `requirements.txt` 中固定版本的依赖。该环境已在 Linux x86_64、Ampere 级 GPU 和 CUDA 12.4 用户态驱动上验证；随附的 `torch==2.5.0` wheel 自带 CUDA 12.4 runtime，因此只需要 NVIDIA 主机驱动支持 CUDA 12.4。

### 2. 配置模型访问

为你希望使用的后端设置 API 凭证：

```bash
# OpenAI
export OPENAI_API_KEY=...

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT=...
export AZURE_API_KEY=...

# Anthropic
export ANTHROPIC_API_KEY=...
```

### 3. 启动评测服务

大多数提供的配置使用运行在端口 `12017` 的远程评测服务。

```bash
bash online_judge/start_server.sh
```

也可以直接启动：

```bash
python -m uvicorn online_judge.app_with_queue:app --host 0.0.0.0 --port 12017
```

## Adapt：失败驱动的技能获取

Adaptation 阶段对应论文的前半部分：AdaExplore 合成多样化的 kernel 风格任务，在这些任务上运行 agent，并将反复出现的失败总结为跨任务规则记忆，例如无效的 Triton 用法模式。在本仓库中，该工作流由 `synthesis/` 实现任务生成，由 `skill_memory/` 提取或更新记忆文件。

我们还在 `results/memory/general_memory_v1_200.txt` 提供了预生成的技能记忆，因此你可以直接将其作为 exploration run 的起点。要运行该阶段，可以直接使用 `datasets/KernelBench_syn/syn_v1` 中的预生成数据集。

可选：如果要从头重建合成数据集，请生成任务并将其物化为相同的 KernelBench 风格目录：

```bash
python synthesis/generate_data.py \
  --server_type azure \
  --model_name gpt-5-mini \
  --prompt_style composite \
  --input_levels 1 \
  --num_generations 200 \
  --num_examples_per_request 3 \
  --temperature 1.0

python synthesis/rename.py \
  --source_path outputs/data_generation/<generated_data_dir_from_previous_step> \
  --data_path datasets/KernelBench_syn/<your_dataset_name> \
  --force
```

在开启在线记忆更新的情况下，在合成数据集上运行 agent：

```bash
python agent/agent_entry.py --config config/SYN-v1/config_SYN-v1_none_MCTS.yaml
```

该配置使用 `memory_update: true`，因此失败的生成会随着运行推进被蒸馏到 `outputs/SYN-v1_example_run/general_memory.txt`。如果要基于 `outputs/...` 下的历史日志刷新随仓库提供的记忆，也可以通过离线方式构建：

```bash
python skill_memory/skill_memory.py \
  --log-dir outputs/example_run \
  --knowledge-store-path outputs/example_run/general_memory.txt \
  --server openai \
  --model-name gpt-5-mini \
  --max-logs 3000 \
  --seed 42
```

## Explore：保持多样性的 Kernel 搜索

Exploration 阶段对应论文的后半部分：AdaExplore 使用 adaptation 阶段收集到的技能记忆来引导搜索，同时基于树的优化器保留多个候选分支，并在局部编辑与结构重建之间交替进行。在本仓库中，该行为通过 `agent/agent_entry.py` 中的 `MCTS` agent 以及 `config/` 下的 benchmark 配置实现。

每个搜索步骤都运行一个 **propose → evaluate → tune** 循环。*proposer* 编写一个新的 kernel；候选 kernel 经过评测框架运行；随后 LLM *evaluator* 诊断结果并返回结构化指导，包括 `small_guidance`、`large_guidance`、`direction`（`large` / `small`）和 `valid` 标志，用于引导下一次扩展是走局部调优（`small`）还是从头重新设计（`large`）。

单个 *evaluator*（`agentprompt/evaluator_prompt.py`）取代了以前单独的 *reviser* agent：诊断、有效性认证和下一步指导现在由 `run_evaluator`（`agent/actions.py`）一次性生成，并由 `agent/mcts.py` 以及 large / small 循环调用。每个 `MCTS` 树节点现在都会内联携带 evaluator 的输出（`small_guidance`、`large_guidance`、`evaluator_direction`、`evaluator_valid`）以及发送给它的精确 prompt，并在每一步持久化为 `step_*_evaluator_prompt.txt` 以便检查。tuner 直接消费这些指导（“Guidance from Evaluator Agent”）。

### 算子感知技能 Prompt

Prompt 内容不再是一个单体块。可复用的优化指导现在按算子拆分为 **skills**，位于 `agentprompt/skills/*.md` 下，包括 `conv`、`linear`、`norm`、`activation`、`pooling`、`reduction`，以及 `_base` 和 `_default`。在 prompt 组装时，`agentprompt/Utils` 解析参考架构源码（`detect_families`，对 `nn.*` / `F.*` 调用执行 AST 遍历），只选择匹配的技能文件，并将实时硬件参数（GPU 名称、架构、dtype）替换到文本中。每个算子技能被拆分为 `## Design` 部分（用于 redesign / large step）和 `## Tuning` 部分（用于局部 tuning）；`_base` 承载始终启用的安全契约。

作为这一迁移的一部分，benchmark 模板（`agentprompt/benchmarks/KB_prompt.py`）和 tuner prompt 中独立的 “Hardware Information” 块被移除；硬件事实现在通过技能系统的 placeholder 替换（`agentprompt/Utils/hardware.py`）从每个配置的 `gpu_name` / `gpu_architecture` / `dtype_str` 注入，从而保持单一事实来源。

### 三种 Prompt 组合模式

Evaluator（`agentprompt/evaluator_prompt.py`）会为每个候选选择三种 review 模式之一，这会共同控制技能抽象层级、结构性警报以及下一步方向：

* **Mode A — redesign（`slow`）**：kernel 正确但较慢（`0 < fast_p < 0.8`），或者其重型算子在实际执行路径上从未真正被替换。只展示 `## Design` 技能内容，方向被强制为 `large`，并将搜索推向结构性重写。
* **Mode B — adversarial（`fast_p ≥ 5`）**：巨大的测量加速比正是代数捷径容易出现的区间，因此 prompt 会将默认判断翻转为 `<valid>false</valid>`，并要求 evaluator 在认证 kernel 之前同时指出完整 shape 的重型算子轴以及执行完整 MAC 次数的累加循环。
* **Mode C — default**：普通 tuning，以及所有编译失败和正确性失败。展示 `## Design` 和 `## Tuning` 两类技能内容，使小步 tuning 仍然可达。

### 代数捷径与重型算子检测

一个反复出现的失败模式是：agent 通过*消除*重型算子而不是加速它来“获胜”——例如在初始化时将 conv 后面的 `mean` / `sum` fold 到权重中，从而用一个很小的 GEMM 替代卷积。这类实现可以通过宽松的 `atol=rtol=5e-2` 正确性检查，但并不是目标优化对象。AdaExplore 从两个方面防御这一问题：

* **静态检测**（`agentprompt/Utils/detect.py`，`detect_shortcut_risk`）：对参考实现执行 AST 分析，检测 `heavy-op → linear-reduction` 链，并注入 *Structural Alert*，说明预期的中间 shape 以及需要检查的捷径模式。
* **运行时检测**（`src/heavyop_checker.py`）：使用 `TorchDispatchMode` 记录在计时 forward 中实际通过 PyTorch dispatch 的 `aten` 重型算子。由于 Triton kernel launch 不会经过 `__torch_dispatch__`，如果出现了重型 `aten` op，就意味着参考算子确实在 PyTorch 中运行了——这可以捕捉那些把“重写”放进死代码中的 kernel，例如永远不会执行的 `if self.training:` 分支。这类 kernel 会被强制进入 Mode A，并被标记为未替换重型算子。

共享的反捷径契约位于 `agentprompt/skills/_base.md`，并注入到每个 prompt 中：每个参考算子都必须物化其完整输出 shape，并执行相同渐近复杂度的乘加次数；普通 kernel 级融合和标准推理时 fold 仍然允许；自定义 `Conv` / `ConvTranspose` kernel 明确被授权。

### 结构化失败反馈

当候选失败时，评测框架不再把数 KB 的原始 traceback 直接塞进 evaluator prompt。`src/triton_error_parser.py` 会对捕获的 Triton traceback 执行 regex 解析，将错误归入命名类别（`syntax`、`autotune`、`gcc`、`ptx`、`tl_constexpr`、`triton_compile`），提取一行 `summary` 和 `last_frame`，按字段截断，并将结构化结果同时持久化到结果 metadata 和相邻的 `step_*_compile_error.json` 中。Evaluator prompt 会渲染这个紧凑 JSON，而不是原始 dump。

### Nsight Profiling

对每个*正确*候选，`src/nsight_profiler.py` 会分阶段 profile kernel：先用低成本的 `nsys` trace 按 GPU 时间对所有 launch 的 kernel 排序，过滤掉框架 / 库噪声（`at::`、`cudnn`、`cutlass` 等），再使用定向 `ncu` pass（默认阶段）只 replay 占主导的候选 kernel，并采集丰富的显式 `--metrics` bundle 以及 NVIDIA 的 section **rule engine**（SpeedOfLight、Occupancy、MemoryWorkloadAnalysis、SchedulerStats）。具备 roofline 感知的专家判断及其估计加速空间会直接暴露在 evaluator prompt 中，使 agent 优化硬件报告的瓶颈，而不是追逐错误杠杆。整个流水线是 best-effort：任何失败都会降级为 `*_error` 字段，不会向评测路径抛出异常。`ncu` 需要 root 权限——在非 root 主机上请提供 `nsight_ncu_sudo`。

## TensorRT FP32 Baseline

`TRT_Baseline/` 构建了一个强 stock-inference 参考，使“kernel 比 torch eager 更快”不会被误认为“kernel 足够好”。它将每个 KernelBench `Model` 导出为 ONNX（`convertONNX/convert_all.py`），通过 `trtexec` 构建并计时 TensorRT FP32 engine（`build_and_time.py`），验证其与 PyTorch 的数值一致性（`atol=rtol=5e-2`），并将每个问题的计时写入 `results/timing/RTX-4090-D/baseline_time_trt.json`，与已有 torch baseline 并列。所有参数（precision、workspace cap、tactic sources、opset、trials）都由 `TRT_Baseline/config.yaml` 驱动；不使用 engine 或 timing cache，因此每次运行都会从头 rebuild。该流程覆盖 KernelBench Level 1 + Level 2，共 200 个文件。

## 复现实验结果

下方两个配置直接对应论文表 1 中 **AdaExplore，50-step，gpt-5-mini** 的两行：

| Config                                         | Benchmark           | Problems               | Steps / problem | Model                |
| ---------------------------------------------- | ------------------- | ---------------------- | --------------- | -------------------- |
| `config/KB-l2/config_KB-l2_AdaExplore_50.yaml` | KernelBench Level 2 | 100（`test_list_2.txt`） | 50              | `gpt-5-mini`（OpenAI） |
| `config/KB-l3/config_KB-l3_AdaExplore_50.yaml` | KernelBench Level 3 | 50（`test_list_3.txt`）  | 50              | `gpt-5-mini`（OpenAI） |

还提供了 Level 1 配置（`config/KB-l1/config_KB-l1_AdaExplore_50.yaml`，由 `test_list_1.txt` 驱动）。

这两个配置已经指向随仓库提供的跨任务技能记忆 `results/memory/general_memory_v1_200.txt`，因此无需重新运行 Adapt 阶段即可直接启动复现。

> **注意 — 随仓库提供的配置默认值 vs. 论文设置。** 上表描述的是论文中的标准配置（`gpt-5-mini`、OpenAI、Ampere、remote judge）。当前分支中的 KB-l2 配置目前以本地开发默认值提交：`server_type: claude` / `model_name: claude-opus-4-7`、`use_remote_eval: false`、`gpu_name: RTX-4090-D` / `gpu_architecture: Ada`、为始终启用的 ncu 阶段设置了 `nsight_ncu_sudo`，并包含调优后的搜索参数（`knowledge_1_threshold: 5`、`small_step_limit: 5`）。若要复现表 1，请将 `server_type` / `model_name` / `use_remote_eval` / `gpu_*` 改回论文设置。

**硬件。** 论文实验使用固定频率（1500MHz）的 A6000，精度为 fp32。

**并发。** 每个配置会并行运行 `num_processes` 个 agent worker，并全部向同一个 online judge 提交评测。

**汇总完成的 run。** 配置运行完成后，使用 `tool_scripts/stats.py` 恢复表 1 中的数字：

```bash
python tool_scripts/stats.py --log_folder outputs/KB-l2_AdaExplore_50 --step 50
python tool_scripts/stats.py --log_folder outputs/KB-l3_AdaExplore_50 --step 50
```

该脚本会报告正确率和聚合加速统计，包括 mean / median / fast_p 阈值准确率。如果要在报告前使用更多 trials 重新测量性能，请先运行 `tool_scripts/re_evaluate.py`，见下方 Evaluation 部分。

**成本 / 运行时间预期。** 一个完整的 50-step Level 2 run 对 OpenAI API 的调用量大约为 $10^4$ 级别；在单张 Ampere GPU 上需要预留数小时 wall-clock 时间，并产生不可忽略的 token 成本。可以通过编辑 test list（例如 `2 1` 表示只评测 level-2 problem 1）或降低 `total_steps` 来减少成本，用于 smoke testing。

## Evaluation

AdaExplore 使用围绕 `online_judge/app_with_queue.py` 和 `src/eval_subprocess_runner.py` 构建的轻量评测框架。Online judge 暴露一个 API 服务，带有按 GPU 限制的并发和队列机制，因此多个 agent worker 可以同时提交评测而不会过度占用设备；请求会被路由到负载最低的 GPU，并在隔离 subprocess 中执行，防止崩溃或非法内存访问拖垮主服务。

在结果质量方面，评测框架将正确性与性能测量分离。正确性会在一个或多个随机 trials 上检查，并使用较小的数值容差（`atol=rtol=5e-2`）以避免因无关紧要的浮点噪声拒绝 kernel；性能会在 warmup 后跨多次运行测量，并在移除 outlier 后汇总为 runtime statistics。在同一条正确路径上，评测框架还会记录重型算子 dispatch trace 以及上文描述的 best-effort Nsight profile。

**每次运行使用新的随机输入。** 数据集（`datasets/KernelBench/level{1-4}`、`datasets/KernelBench_syn/syn_v1`）中的每个 `get_inputs()` 现在都以 `torch.seed()` 开头，从 OS entropy 重新播种。此前，评测框架会在 `get_inputs()` 之前立即调用 `torch.manual_seed(trial_seed)`，导致不同 run 的测试输入完全相同，kernel 可能过拟合到某一次固定输入采样；重新播种使每次 run 都抽取新输入。该 reseed 行由 `tool_scripts/randomize_get_inputs.py` 批量插入。

对于后处理，`tool_scripts/re_evaluate.py` 可以重新运行日志目录中保存的 best kernels，例如：

```bash
python tool_scripts/re_evaluate.py --log_folder outputs/<run> --num_correct_trials 5 --num_perf_trials 100 --use_remote_eval
```

要汇总完成的 run，请使用 `tool_scripts/stats.py`，例如：

```bash
python tool_scripts/stats.py --log_folder outputs/<run> --step 50
```

该脚本会报告 accuracy 和聚合 speedup statistics。

`tool_scripts/` 下的其他辅助脚本：

* `fill_summary.py` / `fill_summary_l2.py`：构建 `outputs/KB-l{level}_AdaExplore_50/Summary.xlsx`，每个 test 一行，包括 name、TRT baseline ms、agent result、ratio、fail flag、implementation，并将 agent 结果与 `baseline_time_trt.json` join。
* `nsight_runner.py`：供 nsys / ncu attach 的独立 whole-process 入口；它会像 `src/eval.py` 一样加载参考 arch 和候选 `ModelNew`，运行 warmup 和 forward pass，但自身不做计时，由 profiler 记录 GPU 工作。
* `eval_one_kernel.py`：将单个保存的 kernel 通过评测框架运行，用于临时检查。

