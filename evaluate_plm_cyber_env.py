import os
import sys
import json
import time
import argparse
import pickle
from typing import Callable, Optional

import numpy as np
import torch
import gym

# Ensure APT-local cyber sim is importable and env registered
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import cfg  # noqa: E402

import apt_cyber_sim  # noqa: E402

from apt_eval.agent_wrapper import AgentWrapper, ActionTrackingStateAugmentation  # noqa: E402
from apt_eval.agent_wrapper import EnvironmentBounds  # noqa: E402
from apt_eval.state_action_tools import CyberBattleStateActionModel, AbstractAttackerModel  # noqa: E402

from plm_special.models.state_encoder_cyber import CyberStateEncoder  # noqa: E402
from plm_special.models.rl_policy_cyber import CyberOfflineRLPolicy  # noqa: E402
from plm_special.utils.dt_reward import make_dt_process_reward  # noqa: E402


def _resolve_exp_pool_path_optional(args) -> Optional[str]:
    """Same resolution as ``run_plm_cyber._resolve_exp_pool_path``; returns None if no pool specified."""
    if getattr(args, "exp_pool_path", None):
        return os.path.normpath(os.path.expanduser(str(args.exp_pool_path).strip()))
    name = getattr(args, "exp_pool", None)
    if name:
        paths = cfg.exp_pool_paths
        if name not in paths:
            raise ValueError(
                f"Unknown --exp-pool {name!r}. Valid keys: {sorted(paths.keys())}. "
                "Add names in config.py or pass --exp-pool-path."
            )
        return paths[name]
    return None


def _parse_env_ids(args) -> list[str]:
    """Resolve evaluation env ids from args."""
    env_ids: list[str] = []
    raw = getattr(args, "env_ids", None)
    if raw:
        for part in str(raw).split(","):
            item = part.strip()
            if item:
                env_ids.append(item)
    elif getattr(args, "env_id", None):
        env_ids.append(str(args.env_id).strip())
    else:
        env_ids.append(cfg.env_id)
    return env_ids


def _aggregate_multi_env_eval(per_env: dict) -> dict:
    """Aggregate multi-environment metrics for quick comparison."""
    if not per_env:
        return {}
    env_ids = list(per_env.keys())
    mean_returns = [float(per_env[e]["summary"]["mean_return"]) for e in env_ids]
    done_early_rates = []
    for e in env_ids:
        s = per_env[e]["summary"]
        episodes = max(int(s.get("episodes", 0)), 1)
        done_early = int(s.get("episodes_done_early", 0))
        done_early_rates.append(done_early / episodes)
    return {
        "env_count": len(env_ids),
        "env_ids": env_ids,
        "mean_return_avg": float(np.mean(mean_returns)),
        "mean_return_min": float(np.min(mean_returns)),
        "mean_return_max": float(np.max(mean_returns)),
        "mean_return_std": float(np.std(mean_returns)),
        "done_early_rate_avg": float(np.mean(done_early_rates)),
    }


def apply_run_plm_cyber_eval_alignment(args) -> None:
    """
    Set ``args.target_return`` and ``args._dt_process_reward_fn`` to match ``run_plm_cyber``:

    - If ``--exp-pool-path`` or ``--exp-pool`` is set: build ``CyberExperienceDataset`` with
      ``--gamma``, ``--scale``, ``--w``, ``--sample-step`` and use
      ``target_return = max_return * target_return_scale`` and
      ``make_dt_process_reward(min_reward, max_reward, scale)`` from dataset stats.
    - Otherwise: require ``--target-return`` and all three ``--dt-reward-min/max/scale``.

    Must run after ``parse_args()`` and before ``evaluate()``.
    """
    pool_path = _resolve_exp_pool_path_optional(args)
    if pool_path:
        from plm_special.data.dataset_cyber import CyberExperienceDataset

        exp_pool = pickle.load(open(pool_path, "rb"))
        ds = CyberExperienceDataset(
            exp_pool,
            gamma=args.gamma,
            scale=args.scale,
            max_length=args.w,
            sample_step=args.sample_step,
        )
        info = ds.exp_dataset_info
        args.target_return = float(info["max_return"]) * float(args.target_return_scale)
        args._dt_process_reward_fn = make_dt_process_reward(
            float(info["min_reward"]),
            float(info["max_reward"]),
            float(args.scale),
        )
        safe_info = {}
        for k, v in info.items():
            if isinstance(v, (np.floating, np.integer)):
                safe_info[k] = float(v)
            elif isinstance(v, (int, float)):
                safe_info[k] = v
            else:
                safe_info[k] = v
        args._eval_alignment = {
            "source": "exp_pool",
            "exp_pool_path": os.path.abspath(pool_path),
            "exp_dataset_info": safe_info,
            "gamma": args.gamma,
            "scale": args.scale,
            "target_return_scale": args.target_return_scale,
            "w": args.w,
            "sample_step": args.sample_step,
        }
        return

    # Manual: must match training numbers explicitly
    if getattr(args, "target_return", None) is None:
        raise SystemExit(
            "evaluate_plm_cyber: provide --exp-pool-path or --exp-pool (recommended; same as training) "
            "for automatic alignment with run_plm_cyber, OR set --target-return together with "
            "--dt-reward-min, --dt-reward-max, and --dt-scale."
        )
    triplet = (args.dt_reward_min, args.dt_reward_max, args.dt_scale)
    if any(v is not None for v in triplet) and not all(v is not None for v in triplet):
        raise SystemExit(
            "evaluate_plm_cyber: provide all three of --dt-reward-min, --dt-reward-max, --dt-scale, or none (use exp pool)."
        )
    if not all(v is not None for v in triplet):
        raise SystemExit(
            "evaluate_plm_cyber: without --exp-pool / --exp-pool-path, you must pass all three "
            "--dt-reward-min, --dt-reward-max, --dt-scale (same as run_plm_cyber --scale and pool stats)."
        )
    args._dt_process_reward_fn = make_dt_process_reward(
        float(args.dt_reward_min),
        float(args.dt_reward_max),
        float(args.dt_scale),
    )
    args._eval_alignment = {
        "source": "manual",
        "target_return": float(args.target_return),
        "dt_reward_min": float(args.dt_reward_min),
        "dt_reward_max": float(args.dt_reward_max),
        "dt_scale": float(args.dt_scale),
    }


def _load_policy(args) -> CyberOfflineRLPolicy:
    plm_path = args.plm_path
    if plm_path is None:
        plm_path = args.plm_hf_id
    if plm_path is None:
        raise ValueError("Please specify --plm-path (local) or --plm-hf-id.")

    # Strip accidental leading/trailing spaces from quoted CLI paths (e.g. `" E:\foo"`).
    plm_path = os.path.normpath(os.path.expanduser(str(plm_path).strip()))
    # If the path does not exist, transformers treats the string as a Hub repo id → confusing HFValidationError on Windows paths.
    if not os.path.isdir(plm_path):
        raise FileNotFoundError(
            f"PLM path is not a directory: {plm_path!r}\n"
            "Fix: use the same --plm-path as training (folder with config.json + tokenizer). "
            "Example: .../downloaded_plms/gpt2/small — not .../downloaded_plms/small."
        )

    from transformers import AutoModel, AutoTokenizer

    model_dir = args.model_dir
    if model_dir is None:
        raise ValueError("Please specify --model-dir pointing to best_model dir.")
    model_dir = os.path.normpath(os.path.expanduser(str(model_dir).strip()))

    adapter_config = os.path.join(model_dir, "adapter_config.json")
    modules_bin = os.path.join(model_dir, "modules_except_plm.bin")
    model_bin = os.path.join(model_dir, "model.bin")

    use_lora = os.path.isfile(adapter_config) and os.path.isfile(modules_bin)

    if use_lora:
        # LoRA checkpoint: base PLM from plm_path, adapter + modules_except_plm from model_dir
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(plm_path, use_fast=True)
        plm = AutoModel.from_pretrained(plm_path)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            plm.resize_token_embeddings(len(tokenizer))
        plm = plm.to(args.device)
        plm = PeftModel.from_pretrained(plm, model_dir)

        mod_sd = torch.load(modules_bin, map_location=args.device)
        # embed_timestep is index 1 in modules_except_plm
        embed_ts_w = mod_sd.get("1.weight", None)
        if embed_ts_w is None:
            raise KeyError("modules_except_plm.bin missing '1.weight' (embed_timestep).")
        max_ep_len = int(embed_ts_w.shape[0] - 1)
    else:
        if not os.path.isfile(model_bin):
            raise FileNotFoundError(
                f"Missing {model_bin}. For LoRA checkpoints expect adapter_config.json and modules_except_plm.bin in --model-dir."
            )
        sd = torch.load(model_bin, map_location=args.device)
        embed_ts_w = sd.get("embed_timestep.weight", None)
        if embed_ts_w is None:
            raise KeyError("Checkpoint missing 'embed_timestep.weight' (unexpected model.bin format).")
        max_ep_len = int(embed_ts_w.shape[0] - 1)

        tokenizer = AutoTokenizer.from_pretrained(plm_path, use_fast=True)
        plm = AutoModel.from_pretrained(plm_path)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            plm.resize_token_embeddings(len(tokenizer))
        plm = plm.to(args.device)

    plm_embed_size = getattr(getattr(plm, "config", None), "hidden_size", None)
    if plm_embed_size is None:
        raise ValueError("Cannot infer PLM hidden_size from model config.")

    state_encoder = CyberStateEncoder(state_dim=args.state_dim, state_feature_dim=args.state_feature_dim).to(args.device)
    model = CyberOfflineRLPolicy(
        state_feature_dim=args.state_feature_dim,
        action_dim=args.action_dim,
        state_encoder=state_encoder,
        plm=plm,
        plm_embed_size=plm_embed_size,
        max_length=args.w,
        max_ep_len=max_ep_len,
        device=args.device,
        device_out=args.device_out,
        which_layer=args.which_layer,
    )

    if use_lora:
        model.modules_except_plm.load_state_dict(mod_sd, strict=True)
    else:
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def evaluate_on_env_cyber(
    args,
    model,
    target_return: float,
    max_ep_num: int,
    eval_max_steps: int | None = None,
    process_reward_fn: Callable[[float], float] | None = None,
    env_id: str | None = None,
):
    """
    Run the given policy in the cyber sim for max_ep_num episodes (ABR-style).
    Used during training to select best model by evaluation return.

    :param args: object with device, state_dim, action_dim, w, which_layer, seed,
                 step_cost, winning_reward, ownership_goal, maximum_node_count.
    :param model: CyberOfflineRLPolicy already on device (e.g. from run_plm_cyber).
    :param target_return: return-conditioning target in **training return space**
        (e.g. max_return * target_return_scale), same as ``ExperienceDataset`` returns.
    :param max_ep_num: number of episodes to run.
    :param eval_max_steps: max steps per episode; if None, use args.eval_max_steps or 600.
    :param process_reward_fn: maps raw env reward to training step reward (clip + min-max + /scale),
        matching ``adaptive_bitrate_streaming/run_plm.py``. If None, uses raw reward (legacy; mismatches training).
    :return: dict with ``episodes_return`` (sum of raw env rewards), ``mean_return_per_episode``,
        ``episodes_done_early``, ``episodes_len`` (sum of step counts), ``time/evaluation``,
        and per-episode ``episode_returns`` / ``episode_lens`` / ``episode_done``.
    """
    apt_cyber_sim.ensure_registered()
    max_steps = eval_max_steps if eval_max_steps is not None else getattr(args, "eval_max_steps", 600)

    # =========================
    # DEBUG: done 前是否达到占领阈值
    # 说明：默认关闭，不影响程序运行。需要时把下面 False 改成 True（或把整段注释掉）。
    # =========================
    DEBUG_DONE_GOAL_STATS = False

    eval_seed = getattr(args, "eval_seed", args.seed)
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)

    env = gym.make(
        env_id or cfg.env_id,
        disable_env_checker=True,
        attacker_goal=apt_cyber_sim.AttackerGoal(own_atleast_percent=args.ownership_goal),
        step_cost=args.step_cost,
        winning_reward=args.winning_reward,
        maximum_node_count=args.maximum_node_count,
    )
    ep = EnvironmentBounds.of_identifiers(
        maximum_total_credentials=5,
        maximum_node_count=args.maximum_node_count,
        identifiers=env.identifiers,
    )
    sa_model = CyberBattleStateActionModel(ep)
    mask_helper = AbstractAttackerModel(sa_model)

    policy = model
    policy.eval()
    t0 = time.time()
    episodes_return = 0.0
    episodes_len = 0
    episodes_done_early = 0  # count of episodes that ended with done=True before max_steps

    if DEBUG_DONE_GOAL_STATS:
        dbg_done_episodes = 0
        dbg_done_goal_before = 0
        dbg_done_goal_after = 0
        dbg_done_goal_never = 0
        dbg_done_step_indices: list[int] = []
        dbg_episode_steps: list[int] = []

        def _ownership_ratio_from_obs(obs) -> float:
            """Compute owned-node ratio from observation.
            Note: attacker_goal in env uses owned_count / node_count with node_count ~= maximum_node_count for toyctf-v0.
            """
            levels = np.asarray(obs.get("nodes_privilegelevel", []), dtype=np.float32)
            if levels.size == 0:
                return 0.0
            owned_count = int(np.sum(levels > 0))
            return owned_count / float(args.maximum_node_count)

    episode_returns_list: list[float] = []
    episode_lens_list: list[int] = []
    episode_done_list: list[bool] = []

    with torch.no_grad():
        for ep_i in range(max_ep_num):
            wrapped = AgentWrapper(env, ActionTrackingStateAugmentation(ep, env.reset()))
            _ = wrapped.reset()
            # Clear rollout token history so the next episode uses only its own context.
            if hasattr(policy, "clear_dq"):
                policy.clear_dq()
            total_reward = 0.0
            done = False
            steps = 0
            tgt = float(target_return)

            while not done and steps < max_steps:
                if DEBUG_DONE_GOAL_STATS:
                    ratio_before = _ownership_ratio_from_obs(wrapped.state.observation)

                state_vec = np.array(sa_model.global_features.get(wrapped.state, node=None), dtype=np.float32)
                state = torch.from_numpy(state_vec).reshape(1, 1, -1).to(args.device)
                obs = wrapped.state.observation
                action_mask = mask_helper.compute_action_mask(obs)
                act = policy.sample(state, target_return=tgt, timestep=steps, action_mask=action_mask)
                _, gym_action, _ = sa_model.implement_action(wrapped, np.int32(act))
                if gym_action is None:
                    break
                _, reward, done, _info = wrapped.step(gym_action)

                if DEBUG_DONE_GOAL_STATS and done:
                    ratio_after = _ownership_ratio_from_obs(wrapped.state.observation)
                    goal_before = ratio_before >= float(args.ownership_goal)
                    goal_after = ratio_after >= float(args.ownership_goal)
                    dbg_done_episodes += 1
                    dbg_done_step_indices.append(steps)
                    dbg_done_goal_before += int(goal_before)
                    dbg_done_goal_after += int(goal_after)
                    dbg_done_goal_never += int(not goal_after)

                r_raw = float(reward)
                r_for_tgt = process_reward_fn(r_raw) if process_reward_fn is not None else r_raw
                total_reward += r_raw
                tgt -= r_for_tgt
                steps += 1

            episodes_return += total_reward
            episodes_len += steps
            episode_returns_list.append(float(total_reward))
            episode_lens_list.append(int(steps))
            episode_done_list.append(bool(done))
            if done and steps < max_steps:
                episodes_done_early += 1

            if DEBUG_DONE_GOAL_STATS:
                dbg_episode_steps.append(steps)

    if DEBUG_DONE_GOAL_STATS and max_ep_num > 0:
        avg_steps = float(np.mean(dbg_episode_steps)) if dbg_episode_steps else 0.0
        print(
            "[DEBUG_DONE_GOAL_STATS]",
            f"done_episodes={dbg_done_episodes}/{max_ep_num}",
            f"goal_hit_before_done={dbg_done_goal_before}",
            f"goal_hit_after_done={dbg_done_goal_after}",
            f"goal_never_reached_after_done={dbg_done_goal_never}",
            f"avg_episode_steps={avg_steps:.2f}",
        )
        if dbg_done_step_indices:
            print(
                "[DEBUG_DONE_GOAL_STATS] done_step_index(min/max)="
                f"{min(dbg_done_step_indices)}/{max(dbg_done_step_indices)}"
            )

    elapsed = time.time() - t0
    mean_return_per_episode = episodes_return / max_ep_num if max_ep_num else 0.0
    return {
        "episodes_return": episodes_return,
        "mean_return_per_episode": mean_return_per_episode,
        "episodes_done_early": episodes_done_early,
        "episodes_len": episodes_len,
        "time/evaluation": elapsed,
        "episode_returns": episode_returns_list,
        "episode_lens": episode_lens_list,
        "episode_done": episode_done_list,
    }


def evaluate(args):
    """Load checkpoint and run evaluation via ``evaluate_on_env_cyber`` (same code path as ``run_plm_cyber``)."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Eval alignment (run_plm_cyber parity):", getattr(args, "_eval_alignment", {}))
    print("Using target_return:", float(args.target_return), "dt_process_reward:", getattr(args, "_dt_process_reward_fn") is not None)

    policy = _load_policy(args)

    # Align attribute names with run_plm_cyber (``--eval-max-steps``) for shared eval function.
    args.eval_max_steps = args.max_steps
    if not hasattr(args, "eval_seed"):
        args.eval_seed = args.seed

    env_ids = _parse_env_ids(args)
    per_env = {}
    for eid in env_ids:
        eval_logs = evaluate_on_env_cyber(
            args,
            policy,
            target_return=float(args.target_return),
            max_ep_num=args.eval_episodes,
            eval_max_steps=args.max_steps,
            process_reward_fn=getattr(args, "_dt_process_reward_fn", None),
            env_id=eid,
        )

        er = eval_logs["episode_returns"]
        el = eval_logs["episode_lens"]
        ed = eval_logs["episode_done"]
        results = [{"episode": i, "return": er[i], "len": el[i], "done": ed[i]} for i in range(len(er))]
        summary = {
            "time_sec": eval_logs["time/evaluation"],
            "episodes": args.eval_episodes,
            "mean_return": float(np.mean(er)) if er else 0.0,
            "std_return": float(np.std(er)) if er else 0.0,
            "mean_len": float(np.mean(el)) if el else 0.0,
            "episodes_return_sum": eval_logs["episodes_return"],
            "mean_return_per_episode": eval_logs["mean_return_per_episode"],
            "episodes_done_early": eval_logs["episodes_done_early"],
            "config": {
                "env_id": eid,
                "step_cost": args.step_cost,
                "winning_reward": args.winning_reward,
                "ownership_goal": args.ownership_goal,
                "maximum_node_count": args.maximum_node_count,
                "target_return": float(args.target_return),
                "dt_process_reward": getattr(args, "_dt_process_reward_fn") is not None,
                "alignment": getattr(args, "_eval_alignment", None),
            },
        }
        per_env[eid] = {"summary": summary, "episodes": results}
        print(f"[env={eid}] summary:", summary)

    aggregate = _aggregate_multi_env_eval(per_env)

    out_dir = cfg.results_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cyber_eval_seed_{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "aggregate": aggregate,
                "per_env": per_env,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Evaluation done.")
    print("Aggregate:", aggregate)
    print("Saved to:", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint with the same eval logic as run_plm_cyber (evaluate_on_env_cyber). "
        "Prefer --exp-pool-path or --exp-pool (same pool as training) so target_return and DT reward "
        "processing match training automatically."
    )
    exp_group = p.add_mutually_exclusive_group(required=False)
    exp_group.add_argument(
        "--exp-pool-path",
        default=None,
        help="Path to exp_pool.pkl (same as training). With this, target_return and DT process_reward match run_plm_cyber.",
    )
    exp_group.add_argument(
        "--exp-pool",
        default=None,
        metavar="NAME",
        help=f"Named pool from config (keys: {', '.join(sorted(cfg.exp_pool_paths.keys()))}). Same as run_plm_cyber.",
    )
    p.add_argument("--sample-step", type=int, default=None, help="Same as run_plm_cyber when building dataset stats for alignment.")
    p.add_argument("--gamma", type=float, default=1.0, help="Same as run_plm_cyber (dataset discount).")
    p.add_argument("--scale", type=int, default=1000, help="Same as run_plm_cyber (DT reward scale / dataset).")
    p.add_argument(
        "--target-return-scale",
        type=float,
        default=1.0,
        help="Same as run_plm_cyber: target_return = max_return * this when using --exp-pool / --exp-pool-path.",
    )

    p.add_argument("--model-dir", required=True, help="Path to best_model dir: either model.bin (full) or adapter_config.json + modules_except_plm.bin (LoRA)")
    p.add_argument("--plm-path", default=None)
    p.add_argument("--plm-hf-id", default=None)

    p.add_argument("--device", default="cpu")
    p.add_argument("--device-out", dest="device_out", default=None)
    p.add_argument("--which-layer", type=int, default=-1)

    p.add_argument("--action-dim", type=int, default=cfg.action_dim)
    p.add_argument("--state-dim", type=int, default=cfg.state_dim)
    p.add_argument("--state-feature-dim", type=int, default=cfg.state_feature_dim)
    p.add_argument("--w", type=int, default=20)

    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument(
        "--max-steps",
        "--eval-max-steps",
        type=int,
        default=cfg.eval_max_steps_default,
        dest="max_steps",
        help="Max steps per episode (same as run_plm_cyber --eval-max-steps).",
    )
    p.add_argument("--seed", type=int, default=100003)
    p.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="RNG seed inside evaluate_on_env_cyber (default: same as --seed; matches run_plm_cyber when unset).",
    )

    p.add_argument("--step-cost", type=float, default=cfg.step_cost)
    p.add_argument("--winning-reward", type=int, default=cfg.winning_reward)
    p.add_argument("--ownership-goal", type=float, default=cfg.ownership_goal)
    p.add_argument("--maximum-node-count", type=int, default=cfg.maximum_node_count)
    p.add_argument(
        "--env-id",
        type=str,
        default=cfg.env_id,
        help="Single environment id for evaluation (e.g. AptCyberBattleToyCtf-v0). Ignored when --env-ids is set.",
    )
    p.add_argument(
        "--env-ids",
        type=str,
        default=None,
        help="Comma-separated environment ids for multi-env evaluation. Example: AptCyberBattleToyCtf-v0,AptCyberBattleNode10V1-v0",
    )

    p.add_argument(
        "--target-return",
        type=float,
        default=None,
        help="Initial return-to-go (training space). Required only if you do NOT pass --exp-pool / --exp-pool-path; "
        "otherwise computed from the pool (max_return * target-return-scale).",
    )
    p.add_argument(
        "--dt-reward-min",
        type=float,
        default=None,
        help="Only when no exp pool: raw min_reward (must use with --dt-reward-max and --dt-scale).",
    )
    p.add_argument(
        "--dt-reward-max",
        type=float,
        default=None,
        help="Only when no exp pool: raw max_reward.",
    )
    p.add_argument(
        "--dt-scale",
        type=float,
        default=None,
        help="Only when no exp pool: same as run_plm_cyber --scale for make_dt_process_reward.",
    )

    args = p.parse_args()
    apply_run_plm_cyber_eval_alignment(args)
    if args.device_out is None:
        args.device_out = args.device
    if args.eval_seed is None:
        args.eval_seed = args.seed
    evaluate(args)

