# 为什么 episodes_return 会是很大的负数？

## 一、原因分析

### 1. 环境每步的 reward 怎么算

在 `apt_cyber_sim/_env/cyberbattle_env.py` 的 `step()` 里：

- **达成攻击目标**（占领比例 ≥ ownership_goal）或 **防守约束被破坏**：`done=True`，本步 `reward = winning_reward`（例如 300）。
- **防守达成目标**（例如驱逐攻击者）：`done=True`，本步 `reward = losing_reward`（例如 0）。
- **未结束**：本步 reward 来自动作结果；若 `non_negative_reward=True`（默认），则 `reward = max(-step_cost, reward)`。  
  当前评估默认 **step_cost=1.0**，所以**未结束的每一步至少是 -1**。

因此：

- 若一局在 **T 步内** 因达成目标而结束，该局 return ≈ `(T-1)×(-1) + 300`，可能为正（例如 T 较小）。
- 若一局**从未达成目标**，会一直跑到 **eval_max_steps**（例如 600）才停，则每步都是 -1，该局 return ≈ **-600**。
- 评估时 **episodes_return** = 所有评估局 return 的**总和**。例如 10 局、每局都跑满 600 步且未达成目标，则  
  **episodes_return ≈ 10 × (-600) = -6000**。

所以：**“绝对值很大的负数” = 多数局都没在步数上限内达成目标，每步都在扣 step_cost。**

### 2. 和训练/数据的关系

- 离线数据里，若采集时也是 `step_cost=1.0`，则“好轨迹”的 return 会包含较多负步数惩罚 + 偶尔的 300。  
- 策略若没学好“尽快达成目标”，评估时就会大量出现“跑满 600 步、每步 -1”的局，导致 episodes_return 很大负值。

---

## 二、解决方案（可选组合）

### 方案 1：看“平均每局回报”和“提前结束局数”（推荐）

- **mean_return_per_episode** = `episodes_return / eval_episodes`  
  例如 -6000 / 10 = -600，表示平均每局约 -600，更直观。
- 统计 **episodes_done_early**：在未跑满 `eval_max_steps` 时就 `done=True` 的局数。  
  若该数 > 0，说明有局达成了目标；若为 0，说明所有局都跑满步数、一局都没赢。

评估脚本已增加返回字段：`mean_return_per_episode`、`episodes_done_early`，训练时打印的评估信息里会一起看到，便于判断是“策略没赢”还是“赢了几局但被很多负步拉低”。

### 方案 2：评估时减小 step_cost（仅用于观察）

- 若只想“看策略是否能在有限步内达成目标”，可在**评估**时把 `step_cost` 调小（例如 0.1 或 0），这样未达成目标的局 return 不会到 -600 那么夸张。  
- 注意：这会改变 return 的数值尺度，和用 step_cost=1.0 采集的离线数据不完全一致；**选 best 模型时仍建议用与数据一致的 step_cost**，否则“最佳”可能和真实目标不一致。

### 方案 3：提高策略质量（治本）

- 增加/改进离线数据（更多“能赢”的轨迹）。
- 调整训练（更长 epoch、合适 lr、return 归一化等），让策略更多学到“尽快占领到 60%”的行为。  
- 这样评估时会有更多局在 600 步内 `done=True`，episodes_return 会上升（负得少或变正）。

### 方案 4：评估时增加 max_steps 的考量（可选）

- 若希望“跑得越久扣得越多”在指标里更明显，可以除一下步数，例如报告 **mean_return_per_step** = `episodes_return / episodes_len`（总回报/总步数），约在 -1 附近表示几乎每步都在扣 step_cost。

---

## 三、小结

| 现象 | 含义 |
|------|------|
| episodes_return 绝对值很大（如 -6000） | 多数局未在步数上限内达成目标，每步 -step_cost 累积 |
| mean_return_per_episode ≈ -600 | 平均每局约 600 步、且未达成目标 |
| episodes_done_early = 0 | 没有一局提前因达成目标而结束 |
| episodes_done_early > 0 | 有局达成目标；episodes_return 会因这些局而提高 |

改进方向：优先看 **mean_return_per_episode** 和 **episodes_done_early** 判断行为，再通过数据与训练（方案 3）提升“赢的局数”和回报；若仅需更易读的尺度，可同时用方案 1 的指标或方案 2 的评估 step_cost。
