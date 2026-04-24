# Dataset Generation Pipeline (APT)

This folder centralizes the dataset generation and processing scripts used for APT.

## What is included

- `run.py`: adapted from `Learn-offline_rl_dataset_generator/scripts/cyberattackerexp/run.py`.
  - Supports attacker training and expert transition sampling to CSV.
- `SAC_attacker.py`, `DQN_attacker.py`, `RAND_attacker.py`: adapted attacker implementations.
- `state_action_tools.py`: state/action abstraction used by attackers.
- `truncate_dataset_after_episodes.py`: post-process CSV by trimming early episodes.
- `mix_exp_pools.py`: merge multiple APT `.pkl` expert pools with ratio control.

## Full data flow

1. Train attacker / collect expert CSV:
   - `python dataset_pipeline/generator/run.py ...`
2. (Optional) Trim unstable early episodes:
   - `python dataset_pipeline/generator/truncate_dataset_after_episodes.py ...`
3. Convert CSV to APT pkl pool:
   - `python artifacts/exp_pools/csv_to_exp_pool_cyber.py --csv-path ... --out-path ...`
4. Mix multiple pkl pools:
   - `python dataset_pipeline/generator/mix_exp_pools.py ...`

## Example commands

Run commands from project root:

- `e:/NetLLM/NetLLM/automated_penetration_test`

### 1) Generate CSV with online attacker training

```bash
python dataset_pipeline/generator/run.py \
  --env-id AptCyberBattleNode10V1-v0 \
  --training_episode_count 200 \
  --iteration_count 600 \
  --agent_alg SAC \
  --step_cost 1.0 \
  --winning_reward 300
```

### 2) Generate expert CSV from existing checkpoint

```bash
python dataset_pipeline/generator/run.py \
  --env-id AptCyberBattleNode10V1-v0 \
  --training_episode_count 500 \
  --iteration_count 600 \
  --agent_alg SAC \
  --step_cost 1.0 \
  --winning_reward 300 \
  --expert_mode \
  --expert_checkpoint attacker-6500.pkl \
  --expert_output dataset_expert_v1.csv
```

### 3) (Optional) Trim early episodes from CSV

```bash
python dataset_pipeline/generator/truncate_dataset_after_episodes.py \
  --csv dataset.csv \
  --outcome Outcome/your_experiment_path/outcome.txt \
  --skip-episodes 1500 \
  --tail-episodes 6500 \
  -o dataset_after1500episodes.csv
```

### 4) Convert CSV to APT pkl exp_pool

```bash
python artifacts/exp_pools/csv_to_exp_pool_cyber.py \
  --csv-path dataset_expert_v1.csv \
  --out-path artifacts/exp_pools/dataset_expert_v1.pkl
```

### 5) Mix multiple pkl pools by ratio

50/50 (control total = min):

```bash
python dataset_pipeline/generator/mix_exp_pools.py \
  --input-paths "artifacts/exp_pools/final_dataset.pkl,artifacts/exp_pools/dataset_expert_v1.pkl" \
  --ratios "1,1" \
  --control-total min \
  --out-path "artifacts/exp_pools/mix_toy_v1_halfhalf.pkl"
```

1/3 each for three pools:

```bash
python dataset_pipeline/generator/mix_exp_pools.py \
  --input-paths "a.pkl,b.pkl,c.pkl" \
  --ratios "1,1,1" \
  --control-total min \
  --out-path "mix_v123_third.pkl"
```

70/30 (v1:v2):

```bash
python dataset_pipeline/generator/mix_exp_pools.py \
  --input-paths "v1.pkl,v2.pkl" \
  --ratios "7,3" \
  --control-total min \
  --out-path "mix_v1v2_70_30.pkl"
```

### 6) Arbitrary ratio support (including 10% steps)

`--ratios` accepts either decimals or integer weights:

- `--ratios "0.1,0.9"` (10/90)
- `--ratios "3,7"` (30/70)
- `--ratios "5,5"` (50/50)

The script auto-normalizes ratios, so they do not need to sum to `1` or `10`.

Optional fixed total-size control:

```bash
python dataset_pipeline/generator/mix_exp_pools.py \
  --input-paths "v1.pkl,v2.pkl" \
  --ratios "7,3" \
  --target-total 1000000 \
  --out-path "mix_v1v2_1m.pkl"
```

### 7) PowerShell loop for 10%~90% sweep

```powershell
$v1 = "artifacts/exp_pools/dataset_expert_ep6500_full.pkl"
$v2 = "artifacts/exp_pools/dataset_expert_v2_ep2030_full.pkl"
for ($i = 1; $i -le 9; $i++) {
  $j = 10 - $i
  python dataset_pipeline/generator/mix_exp_pools.py `
    --input-paths "$v1,$v2" `
    --ratios "$i,$j" `
    --control-total min `
    --out-path "artifacts/exp_pools/mix_v1v2_${i}0_${j}0.pkl"
}
```

## Notes

- The scripts are adapted to current APT module names (`apt_cyber_sim`, `apt_eval`).
- Environment IDs are selected with `--env-id` (for example `AptCyberBattleNode10V1-v0`).
- Output CSV format remains compatible with `csv_to_exp_pool_cyber.py`.
