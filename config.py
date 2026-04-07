"""
Centralized paths and defaults for the APT (CyberBattle offline RL) task.

All directory fields are absolute paths derived from this file's location so
behavior does not depend on the current working directory.

Edit this file to change default artifact locations, environment parameters
(keep aligned with offline dataset generation), and named experience pools.
"""

from __future__ import annotations

import os
from typing import Dict

# APT package root: .../automated_penetration_test
APT_ROOT = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS_DIR = os.path.join(APT_ROOT, "artifacts")
EXP_POOLS_DIR = os.path.join(ARTIFACTS_DIR, "exp_pools")
FT_PLMS_DIR = os.path.join(ARTIFACTS_DIR, "ft_plms")
RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "results")

# Repo layer that typically holds HuggingFace / local PLM weights (sibling of automated_penetration_test)
_NETLLM_ROOT = os.path.normpath(os.path.join(APT_ROOT, ".."))
PLM_DIR = os.path.join(_NETLLM_ROOT, "downloaded_plms")

# ---------------------------------------------------------------------------
# Environment defaults (aligned with Learn-offline_rl_dataset_generator + gym registration)
# ---------------------------------------------------------------------------
ENV_ID = "AptCyberBattleToyCtf-v0"
STEP_COST = 1.0
WINNING_REWARD = 300
OWNERSHIP_GOAL = 0.6
MAXIMUM_NODE_COUNT = 10

# ---------------------------------------------------------------------------
# State / action abstraction (ToyCTF-style dataset; see STATE_AND_ACTION_DIMS.md)
# ---------------------------------------------------------------------------
ACTION_DIM = 21
STATE_DIM = 47
STATE_FEATURE_DIM = 256

# Default max steps for standalone evaluation script (training uses --eval-max-steps)
EVAL_MAX_STEPS_DEFAULT = 600

# ---------------------------------------------------------------------------
# PLM / LoRA
# ---------------------------------------------------------------------------
# When --plm-type is not "auto", it must be one of these (LoRA target_modules mapping).
PLM_TYPES = (
    "gpt2",
    "llama",
    "llama3",
    "llava",
    "t5-lm",
    "opt",
    "mistral",
    "qwen2",
    "qwen3",
    "gemma",
    "gemma2",
    "deepseek",
    "deepseek_v2",
    "deepseek_v3",
)


def _default_exp_pool_paths() -> Dict[str, str]:
    """Named shortcuts for --exp-pool (paths under EXP_POOLS_DIR)."""
    return {
        "final": os.path.join(EXP_POOLS_DIR, "final_dataset.pkl"),
        "toyctf_sac2500": os.path.join(EXP_POOLS_DIR, "toyctf_sac2500_exp_pool.pkl"),
        "cyber_default": os.path.join(EXP_POOLS_DIR, "cyber_exp_pool.pkl"),
    }


class Config:
    """Same pattern as adaptive_bitrate_streaming.config.Config — use `from config import cfg`."""

    apt_root = APT_ROOT
    artifacts_dir = ARTIFACTS_DIR
    exp_pools_dir = EXP_POOLS_DIR
    ft_plms_dir = FT_PLMS_DIR
    results_dir = RESULTS_DIR
    plm_dir = PLM_DIR

    env_id = ENV_ID
    step_cost = STEP_COST
    winning_reward = WINNING_REWARD
    ownership_goal = OWNERSHIP_GOAL
    maximum_node_count = MAXIMUM_NODE_COUNT

    action_dim = ACTION_DIM
    state_dim = STATE_DIM
    state_feature_dim = STATE_FEATURE_DIM
    eval_max_steps_default = EVAL_MAX_STEPS_DEFAULT

    plm_types = list(PLM_TYPES)
    exp_pool_paths = _default_exp_pool_paths()


cfg = Config()
