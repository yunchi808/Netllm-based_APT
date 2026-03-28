# APT 仿真环境与数据集生成器一致性检查

本文档对比 **APT 项目** 中用于评估的 CyberBattleSim 环境与您提供的 **Learn-offline_rl_dataset_generator** 中的配置是否一致。

---

## 一、结论摘要

| 项目 | 是否一致 | 说明 |
|------|----------|------|
| 10 节点 Toy CTF 拓扑与漏洞配置 | ✅ 一致 | `apt_cyber_sim/samples/toyctf/toy_ctf.py` 与原始 `toy_ctf.py` 内容等价 |
| 底层环境逻辑 (step / reward / done / observation) | ✅ 一致 | `apt_cyber_sim/_env/cyberbattle_env.py` 与原始 `cyberbattle_env.py` 逻辑一致 |
| 评估时环境参数 (step_cost / winning_reward / ownership_goal) | ✅ 一致 | 与数据集生成脚本传入参数对齐 |
| observation_space 声明 | ⚠️ 仅 APT 多声明两项 | 见下文，不影响评估一致性 |

---

## 二、10 节点配置（toy_ctf.py）

- **原始**：`Learn-offline_rl_dataset_generator\cyberbattle\samples\toyctf\toy_ctf.py`
- **APT**：`automated_penetration_test\apt_cyber_sim\samples\toyctf\toy_ctf.py`

对比结果：

- 节点集合相同：Website, Website.Directory, Website[user=monitor], GitHubProject, AzureStorage, Sharepoint, AzureResourceManager, AzureResourceManager[user=monitor], AzureVM, client（共 10 个）。
- 各节点的 services、firewall、properties、vulnerabilities、outcome、reward_string、cost 等与原始一致。
- `global_vulnerability_library`、`ENV_IDENTIFIERS`、`new_environment()` 逻辑一致。
- 唯一差异：导入从 `cyberbattle.simulation` 改为 `apt_cyber_sim.simulation`，以及少量格式（逗号、换行），**语义与数据完全一致**。

---

## 三、底层环境（cyberbattle_env.py）

- **原始**：`Learn-offline_rl_dataset_generator\cyberbattle\_env\cyberbattle_env.py`
- **APT**：`automated_penetration_test\apt_cyber_sim\_env\cyberbattle_env.py`

对比结果：

1. **step()**：两处逻辑相同  
   - `__execute_action` → `__observation_reward_from_action_result`  
   - `__attacker_goal_reached()` / `__defender_constraints_broken()` → `done=True`, `reward=__WINNING_REWARD`  
   - `__defender_goal_reached()` → `done=True`, `reward=__LOSING_REWARD`  
   - `__non_negative_reward` 时 `reward = max(-step_cost, reward)`  
   - `OutOfBoundIndex` 时返回 blank observation、reward=0  

2. **reset()**：均调用 `__reset_environment()`，再 `__get_blank_observation()` 并填充 `action_mask`、`discovered_nodes_properties`、`nodes_privilegelevel`。

3. **observation 内容**：两处均在 `__get_blank_observation()` 和 `__observation_reward_from_action_result()` 中提供 `all_nodes_conquer_state`、`all_nodes_properties`，评估与数据集生成使用的 observation 字段一致。

4. **APT 相对原始的差异（不影响一致性）**：  
   - 导入改为 `apt_cyber_sim`；plotly 改为可选导入。  
   - **observation_space**：APT 在 `spaces.Dict` 中**显式声明**了 `all_nodes_conquer_state`、`all_nodes_properties`（MultiBinary）；原始环境在运行时已返回这两项，但未在 observation_space 里声明。APT 的写法更符合 Gym 规范，且不改变实际返回的 observation。  
   - `discriminatedunion` 中对 numpy random 的兼容处理（如 `integers` vs `randint`），仅影响采样接口，不影响 step/observation/reward 语义。

---

## 四、环境注册与运行参数

- **数据集生成**（如 `scripts/cyberattackerexp/run.py`、`main.py`）中：
  - `gym.make('CyberBattleToyCtf-v0', attacker_goal=AttackerGoal(own_atleast_percent=0.6), step_cost=1.0, winning_reward=300, maximum_node_count=10)`  
  - 即：**ownership_goal=0.6, step_cost=1.0, winning_reward=300, 10 节点**。

- **APT 评估**（`apt_cyber_sim/__init__.py`）注册 **AptCyberBattleToyCtf-v0** 时默认：
  - `step_cost=1.0`  
  - `winning_reward=300`  
  - `attacker_goal=AttackerGoal(own_atleast_percent=0.6)`  
  - `maximum_node_count=10`  

- **evaluate_plm_cyber_env.py** 中 `gym.make("AptCyberBattleToyCtf-v0", ...)` 再次传入的 `step_cost`、`winning_reward`、`ownership_goal`、`maximum_node_count` 与上述一致（或使用同一默认值）。

因此，**评估时使用的环境参数与您提供的数据集生成配置一致**。

---

## 五、建议

- 无需修改：当前 APT 的仿真环境与您提供的 **toy_ctf 十节点配置** 和 **cyberbattle_env 底层逻辑** 已对齐，评估与数据集生成使用同一套拓扑、漏洞、reward/done 与 observation 语义。  
- 若将来在 **Learn-offline_rl_dataset_generator** 中修改 `toy_ctf.py` 或 `cyberbattle_env.py`，需要将相同改动同步到 `apt_cyber_sim` 中对应文件，以保持一致性。
