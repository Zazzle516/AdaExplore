# AdaExplore

## Structure

Q: 目前的项目是怎么和 MCTS 交互的

### Prompts

single_large_step

  1. Builds a "pool prompt" from kernel_pool + metrics_pool (recent and elite) via generate_pool_prompt_dual.
  2. Wraps it with generate_proposer_prompt (problem statement + memory + hardware + task).
  3. LLM returns a complete kernel; it gets compiled+evaluated by wrapped_eval_kernel_against_ref → KernelExecResult.

single_small_step

  1. Reviser (generate_reviser_prompt, tuner_prompt.py is the partner): receives the last kernel + last run_info and returns 1–3 sentence guidance.
  2. Tuner (generate_tuner_prompt): receives the whole memory window of (kernel, metric) pairs plus the reviser's guidance, and returns <old_str>/<new_str> patch edits applied via str_replace.


## MCTS Layer

  1. select() — UCB1 walk from root; may stop early if should_expand() says the "expand new child" UCB beats the best child.
  2. Decide use_large_step:
    - dummy_root → always large
    - else → True if num_small_step_children >= small_step_limit or random() < p_large (default 0.25).
  3. expand_large (mcts.py:444) calls _get_diverse_pool_for_large_step to assemble context: walks path-to-root, takes one best correct kernel per large_step "branch", then optionally tops up via softmax sampling of best nodes from off-path branches → calls single_large_step.
  4. expand_small (mcts.py:484) takes node.get_path_to_cut() (this node back to its enclosing large_step ancestor), trims to max_memory_round → calls single_small_step.
  5. simulate simply returns node.reward (no rollouts).
  6. backpropagate updates visits, total_reward, max_reward up to root.


## How a "fail" or "longer" result reaches the next prompt

There is no learned signal back-propagated into prompts (the MCTS reward only steers selection, not text). The result reaches the next prompt purely as inlined evidence


- Reviser prompt (reviser_prompt.py:35-47) — single most recent attempt:
  The runtime metrics of the custom Triton kernels are:
  {run_info}    ← str(KernelExecResult)
  A fail surfaces as compiled=False metadata={'compilation_error': '...'} or correctness=False metadata={'correctness_issue': 'Output 
  mismatch', 'max_difference': [...], ...}. A "longer" result surfaces as correctness=True runtime=... runtime_stats={'fast_p': 0.4, ...}
  — fast_p < 1 is the explicit "slower than baseline" signal. 
  
- Tuner prompt (tuner_prompt.py:188-190) — the whole window:

```py
  "\n".join(
      f"### {i}-th attempt:\n```python\n{kernel}\n```\n### {i}-th Runtime Metrics:\n{metric}"
      for i, (kernel, metric) in enumerate(zip(previous_kernels, previous_metrics))
  )
```
  The model sees the trajectory's history of kernels next to their stringified metrics. filter_wrong_attempts (off by default in MCTS
  path) can drop incorrect attempts here.


  Proposer pool prompt (single_large_step → generate_pool_prompt_dual) — same idea: each pool entry shows kernel + its metrics string,
  separated into "recent" and "elite" buckets.

  So the loop closes like this: metrics → str(metrics) → spliced into the next reviser/tuner/proposer prompt → LLM reads 
  "compilation_error: ..." or "fast_p: 0.4" and produces patches/new kernel. The metrics string is the only feedback channel; everything
  else (UCB, softmax pool, p_large, small_step_limit) decides which past attempts and which mode (large vs small) get assembled into that
  next prompt.

Two side channels worth knowing:
  - --general_memory_path (generate_experience_guidance_prompt) injects long-lived skill notes accumulated across runs by update_memory
  after the run finishes (agent_entry.py:139-159).
  - The MCTS reward 0 / 0.05 / 0.4–1.6 (mcts.py:33-92) only affects select() via UCB1; it never appears in any prompt.




# MCTSNode

- **visits** — how many times this node has been on a backprop path

Every call to backpropagate(new_node, reward) walks current = node; while current is not None: current.visits += 1; current = current.parent. So visits increments for new_node and every ancestor up to root on each MCTS iteration that succeeds.

Consequences:
- The root's visits equals the total number of nodes ever back-propagated through it ≈ the number of completed step() calls (plus 1 for the seed if initialize() was called with a seed kernel; see mcts.py:570-571).
- A node's visits equals 1 + (number of descendants ever created beneath it), because each descendant's backprop bumps it.
- visits == 0 means this node was created (e.g. via _create_node) but has not yet been the target of a simulate → backpropagate cycle. UCB1 treats this as ∞ (mcts.py:114-115) so unvisited children are tried first.

How it's used:
- Denominator of UCB1's exploration term: c · sqrt(ln(parent.visits) / self.visits) (mcts.py:122). A node with many visits has a small exploration bonus → pressure to look elsewhere.
- Denominator of avg_reward: total_reward / visits (mcts.py:97-99).
- In expand_ucb1 it's the parent's visits that go into ln(...), with n_expand² = len(children)² as the denominator (mcts.py:144-146) — that's how the "open a new child" action gets its own exploration term, scaled to compete against descending into existing children.

- **total_reward** — sum of every reward backed up through this node

Same backprop loop: current.total_reward += reward. Each ancestor accumulates the same per-iteration reward added at the new node. Since rewards are in [0, REWARD_MAX] with REWARD_MAX = 1.6 (mcts.py:33-36), total_reward grows roughly proportional to visits × the typical descendant quality.

Consequences:
- The dataclass's own reward property (mcts.py:77-92) is a per-node, deterministic value computed from self.metrics — it's what this kernel scored. Don't confuse it with total_reward (sum over the subtree's history) or avg_reward (mean over that history).
- total_reward / visits = avg_reward. This is the "exploitation by average" term and is what _blended_reward falls back on when reward_alpha < 1.
- It carries no information that avg_reward doesn't, but storing the running sum is cheaper than recomputing the mean and avoids floating-point drift.

How it's used:
- Only via avg_reward (mcts.py:94-99), which feeds _blended_reward(α) = α·max_reward + (1-α)·avg_reward (mcts.py:101-103), which is the exploitation half of UCB1.
  
- **max_reward** — best reward ever observed anywhere in this node's subtree

Backprop conditional: if reward > current.max_reward: current.max_reward = reward. So a node's max_reward is the best per-iteration reward of any descendant it has ever propagated, plus its own reward when it was itself created and backed up.

Why this exists: 

with reward_alpha = 1.0 (the default in agent_entry.py:305), UCB1's exploitation term becomes purely max_reward (mcts.py:103, 117).

That's a deliberate design choice for kernel search — average reward is misleading because most refinement attempts fail, but you don't care: you want to bias selection toward the branch that produced one great kernel, not the branch with the highest mean.

So a single lucky descendant raises every ancestor's UCB1 score and pulls the next select() walk back into that subtree.

Consequences:
- A node's max_reward is monotonically non-decreasing over a run.
- The root's max_reward ≈ the global best reward seen so far (modulo edge cases like the seeded root in initialize).
- For a leaf that has just been created, max_reward equals the value simulate(node) returned — i.e. its own reward.
- Two siblings with similar avg_reward but very different max_reward will be ranked by max_reward under α=1 — the one whose subtree once landed a fast kernel wins.

Putting them together

  UCB1 (mcts.py:105-123):

  ucb1 = α · max_reward + (1-α) · (total_reward / visits)   ← exploitation
       + exploration_weight · sqrt(ln(parent.visits) / visits)   ← exploration

  So:
  - visits is the only one of the three on the denominator — fewer visits ⇒ bigger exploration bonus.
  - total_reward / visits is the average-quality signal.
  - max_reward is the peak-quality signal of the subtree.
  - reward_alpha (default 1.0) decides which of the latter two dominates exploitation.

  A useful sanity-check identity: just before any backprop, all three are zero on a freshly created node, and ucb1 returns inf so the node is selected next.

  What is not in these three

  - They do not record what the kernel scored. That's node.reward (a property over node.metrics).
  - They do not track ranking against other nodes globally. The "is this the best kernel found?" decision is made in _create_node against self.global_best_node using the score
  tuple (compiled, correctness, speedup) (mcts.py:273-275), which is unrelated to UCB1 stats.
  - They are not persisted directly per kernel; they're per-node, written every backprop, and dumped to step_<i>_log.json by _save_step_log (mcts.py:687-703) so resume can
  reconstruct them.

  Concrete example

  Say steps run as: dummy_root → propose A (fail, reward 0) → propose B (correct, fast_p=4, reward ≈ 0.94) → refine B into B' (fail, reward 0.05).

  After step 3, on the root:
  - visits = 4 (one per backprop: seeded root, A, B, B')
  - total_reward = 0.0 + 0.0 + 0.94 + 0.05 = 0.99 
  - max_reward = 0.94

  On B:
  - visits = 2 (B itself, then B')
  - total_reward = 0.94 + 0.05 = 0.99
  - max_reward = 0.94

  On A (a sibling):
  - visits = 1, total_reward = 0.0, max_reward = 0.0.
  
  Under α=1, B's exploitation is 0.94 vs A's 0.0 — the next select() walk goes into B's branch unless A's exploration bonus dominates (and A has only 1 visit, so its exploration
  bonus is meaningful; this is why exploration_weight matters). That's exactly the trade-off these three numbers exist to express.