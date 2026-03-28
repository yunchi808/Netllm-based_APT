# run_plm_cyber.py 可自定义参数说明

## 一、数据与经验池

| 参数 | 类型 | 默认值 | 对训练的影响 |
|------|------|--------|--------------|
| `--exp-pool-path` | str | 必填 | 经验池 pkl 路径（由 csv_to_exp_pool_cyber 生成）。决定离线轨迹数据来源。 |
| `--sample-step` | int | None | 从轨迹中采样窗口时的步长；None 时等于 `--w`。越小样本数越多、训练更慢；越大样本更稀疏。 |

---

## 二、模型结构（需与数据/环境一致）

| 参数 | 类型 | 默认值 | 对训练的影响 |
|------|------|--------|--------------|
| `--action-dim` | int | 21 | 动作空间维度，需与 CyberBattle 抽象动作数一致。 |
| `--state-dim` | int | 47 | 状态向量维度，需与 exp_pool 中 state 维度一致。 |
| `--state-feature-dim` | int | 256 | 状态编码器输出维度（MLP 隐层），影响 state→embedding 的表达能力。 |

---

## 三、序列与回报

| 参数 | 类型 | 默认值 | 对训练的影响 |
|------|------|--------|--------------|
| `--w` | int | 20 | 上下文窗口长度（Decision Transformer 的 max_length）。越大看到的轨迹越长，显存与计算增加。 |
| `--gamma` | float | 1.0 | 折扣因子，用于在数据集内计算 return。1.0 表示不折扣。 |
| `--scale` | int | 1000 | return 归一化除数，与 tokenizer/embed 数值尺度匹配，影响 return 条件化的尺度。 |

---

## 四、PLM 与 LoRA

| 参数 | 类型 | 默认值 | 对训练的影响 |
|------|------|--------|--------------|
| `--plm-path` | str | None | **本地 PLM 目录**（含 config.json、pytorch_model.bin 等）。优先于 `--plm-hf-id`。 |
| `--plm-hf-id` | str | None | HuggingFace 模型 id，当无 `--plm-path` 时下载使用。 |
| `--plm-type` | str | "auto" | LoRA 的 target_modules：gpt2/llama/mistral/opt/t5-lm；auto 时从模型 config 推断。**仅当 rank>0 时生效。** |
| `--rank` | int | -1 | LoRA 秩。-1 表示全量微调；>0 表示 LoRA 微调，只保存 adapter + modules_except_plm。 |
| `--which-layer` | int | -1 | 从 PLM 哪一层取 hidden states 做决策头输入；-1 通常表示最后一层。 |

---

## 五、优化与训练节奏

| 参数 | 类型 | 默认值 | 对训练的影响 |
|------|------|--------|--------------|
| `--lr` | float | 1e-4 | 学习率。过大易不稳定，过小收敛慢。 |
| `--weight-decay` | float | 1e-4 | AdamW 权重衰减，减轻过拟合。 |
| `--warmup-steps` | int | 2000 | 学习率线性 warmup 步数，之后保持 1x。 |
| `--num-epochs` | int | 5 | 训练轮数。 |
| `--grad-accum-steps` | int | 32 | 梯度累积步数，等效 batch = 32 个窗口再更新一次参数。 |
| `--seed` | int | 100003 | 随机种子，保证可复现。 |

---

## 六、检查点与评估（ABR 风格）

| 参数 | 类型 | 默认值 | 对训练的影响 |
|------|------|--------|--------------|
| `--save-checkpoint-per-epoch` | int | 1 | 每 N 个 epoch 保存一次 checkpoint（0, N, 2N...）。 |
| `--eval-per-epoch` | int | 1 | 每 N 个 epoch 在仿真环境里评估一次，用于选 best 模型。 |
| `--target-return-scale` | float | 1.0 | 评估时 return 条件 = 数据集中 max_return × 该系数。 |
| `--eval-episodes` | int | 10 | 每次评估跑的 episode 数，越多评估越稳、越慢。 |
| `--eval-max-steps` | int | 600 | 每个评估 episode 的最大步数。 |

---

## 七、评估用环境参数（与 toy_ctf 一致）

| 参数 | 类型 | 默认值 | 对训练的影响 |
|------|------|--------|--------------|
| `--step-cost` | float | 1.0 | 每步成本（负奖励）。 |
| `--winning-reward` | int | 300 | 达成目标时的奖励。 |
| `--ownership-goal` | float | 0.6 | 目标占领节点比例（0.6 = 60%）。 |
| `--maximum-node-count` | int | 10 | 环境最大节点数（v0 拓扑）。 |

---

## 八、设备

| 参数 | 类型 | 默认值 | 对训练的影响 |
|------|------|--------|--------------|
| `--device` | str | "cpu" | 主设备，如 "cuda" 或 "cuda:0"。 |
| `--device-out` | str | None | 与 device 一致；输出/部分计算可放另一设备。 |
| `--device-map` | str | None | 大模型多卡时用，如 "auto" 传给 from_pretrained，由 PEFT/transformers 分配层到多设备。 |

---

## 推荐：本地 PLM + 100 epochs

- 使用**本地已下载的 PLM**（例如 `E:\NetLLM\NetLLM\downloaded_plms\gpt2`）。
- 训练 **100 个 epoch**，每 epoch 保存 checkpoint，每 epoch 做一次仿真评估并按 eval return 更新 best 模型。
- 经验池使用你已有的 `toyctf_sac2500_exp_pool.pkl`（路径按你本机修改）。

**全量微调（rank=-1）示例：**

```cmd
cd /d E:\NetLLM\NetLLM\automated_penetration_test

python run_plm_cyber.py ^
  --exp-pool-path "E:\NetLLM\NetLLM\automated_penetration_test\artifacts\exp_pools\toyctf_sac2500_exp_pool.pkl" ^
  --plm-path "E:\NetLLM\NetLLM\downloaded_plms\gpt2" ^
  --num-epochs 100 ^
  --rank -1 ^
  --w 20 ^
  --lr 1e-4 ^
  --weight-decay 1e-4 ^
  --warmup-steps 2000 ^
  --grad-accum-steps 32 ^
  --eval-per-epoch 1 ^
  --save-checkpoint-per-epoch 10 ^
  --eval-episodes 10 ^
  --device cuda
```

**LoRA 微调（省显存、适合更大 PLM）示例：**

```cmd
python run_plm_cyber.py ^
  --exp-pool-path "E:\NetLLM\NetLLM\automated_penetration_test\artifacts\exp_pools\toyctf_sac2500_exp_pool.pkl" ^
  --plm-path "E:\NetLLM\NetLLM\downloaded_plms\gpt2" ^
  --num-epochs 100 ^
  --rank 8 ^
  --plm-type gpt2 ^
  --w 20 ^
  --lr 1e-4 ^
  --weight-decay 1e-4 ^
  --warmup-steps 2000 ^
  --grad-accum-steps 32 ^
  --eval-per-epoch 1 ^
  --save-checkpoint-per-epoch 10 ^
  --eval-episodes 10 ^
  --device cuda
```

说明：
- `--save-checkpoint-per-epoch 10`：每 10 个 epoch 存一次 checkpoint，避免 100 次全量保存占盘过多。
- `--eval-per-epoch 1`：每 epoch 评估一次，best 模型按评估 return 更新。
- 若 PLM 在其它盘或目录，只需把 `--plm-path` 和 `--exp-pool-path` 改成你本机路径即可。

---

## `evaluate_plm_cyber_env.py` 与训练评估对齐

独立评估**直接调用** `evaluate_on_env_cyber`（与 `run_plm_cyber` **同一代码路径**）。

### 推荐（与训练严格一致）

传入与训练**相同的经验池**与**相同的 `gamma` / `scale` / `w` / `sample-step` / `target-return-scale`**，脚本会按 `run_plm_cyber` 的规则自动设置：

- `target_return = exp_dataset_info["max_return"] * target_return_scale`
- `make_dt_process_reward(min_reward, max_reward, scale)`（池内原始 `min_reward` / `max_reward` + `--scale`）

示例：

```cmd
python evaluate_plm_cyber_env.py ^
  --model-dir "...\best_model" ^
  --plm-path "...\gpt2\small" ^
  --exp-pool-path "E:\NetLLM\NetLLM\automated_penetration_test\artifacts\exp_pools\your_pool.pkl" ^
  --gamma 1.0 --scale 1000 --w 20 --target-return-scale 1.0 ^
  --eval-episodes 50 --max-steps 600 ^
  --device cuda:0
```

（`--sample-step` 若训练时未改，可省略；与训练一致时请与 `run_plm_cyber` 相同。）

### 无经验池时（手动）

必须同时提供 **`--target-return`** 与 **`--dt-reward-min`、`--dt-reward-max`、`--dt-scale`**（与训练时数据与 `scale` 一致），否则会报错提示使用 `--exp-pool-path` / `--exp-pool`。

### 其它对应关系

| 训练 | 独立评估 |
|------|----------|
| `--eval-max-steps` | `--max-steps` 或 `--eval-max-steps`（同义） |
| `--eval-episodes` | `--eval-episodes` |
| 环境超参 | `--step-cost`、`--winning-reward` 等（默认与 `config` 一致） |

结果 JSON 的 `summary.config.alignment` 会记录对齐来源（`exp_pool` / `manual`）及数据集统计（使用经验池时）。
