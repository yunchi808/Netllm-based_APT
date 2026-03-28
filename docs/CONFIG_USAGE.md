# APT `config.py` 使用说明

APT 在 `automated_penetration_test/config.py` 中集中管理**绝对路径**、**仿真环境默认值**、**状态/动作维度**以及**命名经验池**，用法与 ABR 工程的 `adaptive_bitrate_streaming/config.py` 类似。

## 1. 文件里有什么？

| 类别 | 变量（`cfg.xxx`） | 含义 |
|------|-------------------|------|
| 路径 | `apt_root`, `artifacts_dir`, `exp_pools_dir`, `ft_plms_dir`, `results_dir` | 均以 `config.py` 所在目录为基准的绝对路径 |
| PLM 根目录 | `plm_dir` | 默认指向与 `automated_penetration_test` 同级的 `downloaded_plms`（可按本机布局修改） |
| 环境 | `env_id`, `step_cost`, `winning_reward`, `ownership_goal`, `maximum_node_count` | 与离线 CSV 采集配置对齐；`apt_cyber_sim` 注册环境时也会读这些值 |
| 抽象维度 | `action_dim`, `state_dim`, `state_feature_dim` | 与 `STATE_AND_ACTION_DIMS.md` / 数据集一致 |
| 评估 | `eval_max_steps_default` | 独立评估脚本与训练评估的默认每局最大步数 |
| 命名池 | `exp_pool_paths` | 字典：`NAME -> pkl` 绝对路径，供 `run_plm_cyber.py --exp-pool NAME` 使用 |
| LoRA | `plm_types` | `--plm-type` 非 `auto` 时的合法取值（与 LoRA `target_modules` 映射一致） |

在其它脚本中：

```python
from config import cfg

print(cfg.exp_pools_dir)
print(cfg.exp_pool_paths["final"])
```

## 2. 命令行如何配合？

### 训练 `run_plm_cyber.py`

- **必须**二选一：
  - `--exp-pool-path <绝对或相对路径>`：直接指定 `pkl`；
  - `--exp-pool <NAME>`：使用 `cfg.exp_pool_paths` 里的名字（如 `final`、`toyctf_sac2500`、`cyber_default`）。
- 其余参数（`action_dim`、`step_cost` 等）的**默认值**来自 `cfg`；仍可在命令行覆盖以保持实验可复现文档化。

示例：

```bat
cd /d E:\NetLLM\NetLLM\automated_penetration_test

REM 使用命名池（路径在 config 里维护）
python run_plm_cyber.py --exp-pool final --plm-path "E:\NetLLM\NetLLM\downloaded_plms\gpt2\small" ...

REM 与原来一样显式写 pkl
python run_plm_cyber.py --exp-pool-path "artifacts\exp_pools\final_dataset.pkl" ...
```

注意：`--exp-pool` 与 `--exp-pool-path` **不能同时**出现（互斥组）。

### CSV → PKL `artifacts/exp_pools/csv_to_exp_pool_cyber.py`

- `--out-path` 默认改为 `cfg.exp_pool_paths["cyber_default"]`（即 `artifacts/exp_pools/cyber_exp_pool.pkl` 的绝对路径），避免依赖当前工作目录。

### 评估 `evaluate_plm_cyber_env.py`

- 环境相关默认值与结果输出目录 `cfg.results_dir` 来自 config；`--target-return` 默认与 `cfg.winning_reward` 一致（可按需改命令行）。

### 画图 `plot_training_results.py`

- `--model-dir` 的帮助信息会提示 `cfg.ft_plms_dir`，报错时也会附带该路径，便于找到训练输出目录。

## 3. 如何扩展？

1. **新数据集 pkl**：在 `config.py` 的 `_default_exp_pool_paths()` 中增加键值，例如 `"my_run": os.path.join(EXP_POOLS_DIR, "my_run.pkl")`，然后使用 `--exp-pool my_run`。
2. **改环境参数**：只改 `config.py` 一处即可同步到 `apt_cyber_sim` 注册默认值与 `run_plm_cyber` / `evaluate_plm_cyber_env` 的 argparse 默认（除非命令行显式覆盖）。
3. **换磁盘/目录**：修改 `config.py` 中基于 `APT_ROOT` 的路径拼接，或增加你自己的根目录常量；**不要**在多个脚本里硬编码 `E:\...`。

## 4. 与 ABR `config` 的对应关系

| ABR | APT |
|-----|-----|
| `trace_dirs`, `video_size_dirs` | `exp_pool_paths`（命名数据） |
| `artifacts_dir`, `exp_pools_dir`, `results_dir` | 同名概念，在 APT 中均为绝对路径 |
| `plm_dir`, `plm_types`, `plm_embed_sizes` | APT 目前用 `plm_dir` + `plm_types`；嵌入维度由实际加载的 PLM `hidden_size` 推断 |

## 5. Return-conditioning（与 ABR 一致的 reward 口径）

训练数据中的 `returns` 来自 `ExperienceDataset`：对经验池做 min-max 归一化后再折扣并除以 `--scale`。

- **`run_plm_cyber.py`**：周期评估 `evaluate_on_env_cyber` 时，会用 `plm_special/utils/dt_reward.py` 中的 **`make_dt_process_reward(min,max,scale)`**（与 ABR `run_plm.py` 的 `process_reward` 相同）把环境**原始** `reward` 映射到训练尺度，再更新 `target_return`（`tgt`）。报告的 `episodes_return` 仍为**原始环境回报**之和，便于与 Cyber 奖励尺度对照。

- **独立运行 `evaluate_plm_cyber_env.py`**：若要与训练一致，请同时传入训练时经验池统计量与 `--scale`：
  - `--dt-reward-min`、`--dt-reward-max`：训练打印的 `Dataset info` 里 `min_reward` / `max_reward`
  - `--dt-scale`：`run_plm_cyber` 的 `--scale`
  - `--target-return`：应使用训练时的 `max_return * target_return_scale`（训练 return 空间），而不是单独的 `winning_reward`，除非你有意做启发式测试。
