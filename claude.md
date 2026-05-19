I am still trying to learn this proj, these are my notes

Run local

### The process

agent_entry.py.create_inference_server() -> agent/inference_server.py

Search


### Stages

Adapt: failure-driven skill memory

- skill_memory/skill_memory.py
- synthesis/generate_data.py
- agent/actions.py

Explore: diversity-preserving tree search

Uses the skill memory above as a prompt input and runs MCTS over candidate kernels.

- agent/mcts.py
- agent/large_loop.py
- agent/small_loop.py

# How to run

1. Start a remote server

change the `config/KB-l2/config_KB-l2_AdaExplore_50.yaml` API server

Might also need to change GPU number

```sh
bash online_judge/start_server.sh
```

2. Smock Test

run a smock test

```sh
python -c "
from agent.inference_server import create_inference_server, query_inference_server
s = create_inference_server('claude')
print(query_inference_server(s, 'claude-opus-4-7', 'reply with: ok', max_completion_tokens=32))
"
```

if returns "ok", process

3. Run the agents

```sh
python agent/agent_entry.py --config config/KB-l2/config_KB-l2_AdaExplore_50.yaml
```

output: outputs/KB-l2_AdaExplore_50/2_1/

```sh
python tool_scripts/stats.py --log_folder outputs/KB-l2_AdaExplore_50 --step 5
```


The output files


Baseline Kernel: **Torch Eager Mode**

- outputs/KB-l2_AdaExplore_50/2_1/reference_src.py

Iterator Kernel

- outputs/KB-l2_AdaExplore_50/2_1/step_x_log.json       result of MCTS search
- outputs/KB-l2_AdaExplore_50/2_1/step_x_metrics.json   result of the current kernel(compiled, correctness, runtime, run_stats)
- outputs/KB-l2_AdaExplore_50/2_1/step_x.py             LLM output(candidate kernel)
- outputs/KB-l2_AdaExplore_50/2_1/step_x_prompt.txt     LLM input


Step0:

MCTS root placeholder, no real work


### Pipeline

Each MCTS step does roughly:

1. Pick a node to expand (UCB over _log.json stats across the tree).
2. LLM generates a new kernel using the picked node's code + general_memory_v1_200.txt rules → write step_N.py.
3. Submit step_N.py to the online judge → write step_N_metrics.json.
4. Backprop the score up the tree → write step_N_log.json for the new node, update parents.
5. If step_N_prompt.txt exists, that's the exact prompt sent to the LLM (useful for debugging proposals).


## Can a pure-PyTorch PI05 model be dropped in directly?

```py
class Model(nn.Module):       # 必须精确命名为 `Model`
    def __init__(self, ...):  # 构造函数参数由 get_init_inputs() 提供
        ...
    def forward(self, x, ...):
        ...

def get_init_inputs():        # 返回一个列表，传递给 Model(*list)
    return [...]

def get_inputs():             # 返回一个列表，传递给 forward(*list)
    return [torch.rand(...)]  # CPU 张量；评测时会将它们移动到 GPU
```

1. Drop the PyTorch source into a new file, e.g. datasets/KernelBench/level3/51_PI05.py.
2. Rename your top-level module class to Model (or alias it).
3. Add get_init_inputs() returning the constructor args, and get_inputs() returning a representative input list (use small but realistic shapes — every step runs forward several times for correctness + perf).
1. Add a line 3 51 to a custom test list (or your own copy of config/test_list/test_list_3.txt).
2. Run with the L3 config (server_type: claude, etc.).


## Change Baseline

因为目前比较的基准是 torch.eager 模式的 没有什么说服力

所以想修改 Baseline 的话  可以对应修改初始文件 `datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py` 

output/ 中每次重新生成都会覆盖

如果把 `max-autotune` 选项打开的话  同时执行多个任务  任务之间会有竞争的情况

如果一个子进程在另一个工作进程写入缓存后才访问，它会直接获取之前选定的 kernel

比如 `outputs/KB-l2_AdaExplore_50/2_1/step_5_metrics.json`, `outputs/KB-l2_AdaExplore_50/2_1/step_6_metrics.json` 和 `outputs/KB-l2_AdaExplore_50/2_1/step_7_metrics.json`

但是这样比较就失去意义了  现在选择直接写死一个比较基准  尝试激发 Agent 的能力

```sh
python - <<'EOF'
import json, time, importlib.util, torch, sys, os
sys.path.append(os.getcwd())
spec = importlib.util.spec_from_file_location("ref", "datasets/KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
M = m.Model(*m.get_init_inputs()).cuda().eval()
x = m.get_inputs()[0].cuda()

# 充分预热：触发编译 + 自动调优 + 缓存写入
for _ in range(20): M(x)
torch.cuda.synchronize()

# 计时
N = 200
t = time.perf_counter()
for _ in range(N): M(x)
torch.cuda.synchronize()
mean = (time.perf_counter() - t) * 1000 / N
print(f"baseline mean {mean:.3f} ms")

out = {"level2": {"1_Conv2D_ReLU_BiasAdd.py": {
      "mean": mean, "std": 0.0, "min": mean, "max": mean,
      "num_trials": N, "hardware": "NVIDIA GeForce RTX 4090 D", "device": "cuda:0"}}}
os.makedirs("results/timing/RTX-4090-D", exist_ok=True)
json.dump(out, open("results/timing/RTX-4090-D/baseline_time_torch.json", "w"), indent=2)
EOF
```

# Hardware

4090, Thor (TRT)

SASS -> PTX

和 TensorRT 做 SOTA 性能对比的时候  记录 TRT.exe 的参数
搭建一个 pipeline 把目前的几个 level 的测试模型 torch->onnx->TRT 中执行  获取 Baseline 数据