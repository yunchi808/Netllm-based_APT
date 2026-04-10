import os
import sys
import pickle
import numpy as np
import torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from plm_special.utils.console_logger import ConsoleLogger  # noqa: E402

from config import cfg  # noqa: E402

from plm_special.data.dataset_cyber import CyberExperienceDataset
from plm_special.models.state_encoder_cyber import CyberStateEncoder
from plm_special.models.rl_policy_cyber import CyberOfflineRLPolicy
from plm_special.trainer_cyber import CyberTrainer

from evaluate_plm_cyber_env import evaluate_on_env_cyber
from plm_special.utils.dt_reward import make_dt_process_reward


def _infer_plm_type(plm_path: str, plm) -> str:
    """Infer plm_type from path or config for LoRA target_modules. Lazy-imports low_rank."""
    from plm_special.models.low_rank import TARGET_MODULES

    cfg = getattr(plm, "config", None)
    model_type = (getattr(cfg, "model_type", None) or "").lower() if cfg else ""
    path_lower = str(plm_path).lower()

    if model_type and model_type in TARGET_MODULES:
        # HF Llama 3 often still reports model_type='llama'; use path to pick llama3 key (same LoRA modules).
        if model_type == "llama" and ("llama-3" in path_lower or "llama3" in path_lower):
            return "llama3"
        return model_type

    # Path heuristics (order: more specific substrings first)
    path_rules = (
        ("qwen3", "qwen3"),
        ("qwen2.5", "qwen2"),
        ("qwen2", "qwen2"),
        ("qwen", "qwen2"),
        ("gemma-2", "gemma2"),
        ("gemma2", "gemma2"),
        ("gemma", "gemma"),
        ("deepseek-v3", "deepseek_v3"),
        ("deepseek_v3", "deepseek_v3"),
        ("deepseek-v2", "deepseek_v2"),
        ("deepseek_v2", "deepseek_v2"),
        ("deepseek", "deepseek"),
        ("llama-3", "llama3"),
        ("llama3", "llama3"),
        ("llava", "llava"),
        ("mistral", "mistral"),
        ("mixtral", "mistral"),
        ("opt-", "opt"),
        ("/opt/", "opt"),
        ("gpt2", "gpt2"),
        ("llama", "llama"),
    )
    for needle, key in path_rules:
        if needle in path_lower and key in TARGET_MODULES:
            return key

    for k in ("mistral", "gpt2", "opt", "t5"):
        if k in path_lower or (model_type and k in model_type):
            if k == "t5":
                return "t5-lm"
            return k
    return "gpt2"


def save_model(args, model, save_dir):
    if args.rank > 0:
        model.plm.save_pretrained(save_dir)
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, "modules_except_plm.bin"))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "model.bin"))


def _resolve_exp_pool_path(args) -> str:
    """Use --exp-pool-path or a named --exp-pool from config."""
    if getattr(args, "exp_pool_path", None):
        return args.exp_pool_path
    name = getattr(args, "exp_pool", None)
    if name:
        paths = cfg.exp_pool_paths
        if name not in paths:
            raise ValueError(
                f"Unknown --exp-pool {name!r}. Valid keys: {sorted(paths.keys())}. "
                "Add names in config.py or pass --exp-pool-path."
            )
        return paths[name]
    raise ValueError("Specify --exp-pool-path or --exp-pool (see config.cfg.exp_pool_paths).")


def _parse_eval_env_ids(args) -> list[str]:
    """Parse multi-env evaluation ids for training-time validation."""
    raw = getattr(args, "eval_env_ids", None)
    if not raw:
        return [str(getattr(args, "env_id", cfg.env_id)).strip()]
    env_ids = [x.strip() for x in str(raw).split(",") if x.strip()]
    return env_ids or [str(getattr(args, "env_id", cfg.env_id)).strip()]


def _aggregate_eval_returns(per_env_logs: dict) -> dict:
    """Aggregate returns across environments for checkpoint selection."""
    env_ids = list(per_env_logs.keys())
    sums = [float(per_env_logs[e]["episodes_return"]) for e in env_ids]
    means = [float(per_env_logs[e]["mean_return_per_episode"]) for e in env_ids]
    return {
        "env_ids": env_ids,
        "episodes_return_mean": float(np.mean(sums)) if sums else float("-inf"),
        "episodes_return_min": float(np.min(sums)) if sums else float("-inf"),
        "mean_return_per_episode_mean": float(np.mean(means)) if means else float("-inf"),
    }


def run(args):
    exp_pool = pickle.load(open(args.exp_pool_path, "rb"))

    # dataset uses dones to split episodes and compute returns
    exp_dataset = CyberExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
    print("Dataset size:", len(exp_dataset))
    print("Dataset info:")
    pprint(exp_dataset.exp_dataset_info)

    plm_path = args.plm_path
    if plm_path is None:
        plm_path = args.plm_hf_id
    if plm_path is None:
        raise ValueError("Please specify --plm-path (local) or --plm-hf-id (HF model id).")

    from transformers import AutoModel, AutoTokenizer

    load_kw = {}
    if getattr(args, "device_map", None):
        load_kw["device_map"] = args.device_map
    plm = AutoModel.from_pretrained(plm_path, **load_kw)
    tokenizer = AutoTokenizer.from_pretrained(plm_path, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        plm.resize_token_embeddings(len(tokenizer))

    if not load_kw:
        plm = plm.to(args.device)

    if args.rank > 0:
        from plm_special.models.low_rank import peft_model
        plm_type = args.plm_type if args.plm_type != "auto" else _infer_plm_type(plm_path, plm)
        plm = peft_model(
            plm,
            plm_type,
            args.rank,
            print_trainable=True,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_preset=args.lora_target,
        )

    # Prefer the actual loaded model hidden size to avoid cfg/model mismatches
    plm_embed_size = getattr(getattr(plm, "config", None), "hidden_size", None)
    if plm_embed_size is None:
        raise ValueError("Cannot infer PLM hidden_size from model config.")
    max_ep_len = int(exp_dataset.exp_dataset_info["max_timestep"]) + 1

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / args.warmup_steps, 1))
    trainer = CyberTrainer(
        args,
        model=model,
        optimizer=optimizer,
        exp_dataset=exp_dataset,
        device=args.device,
        action_dim=args.action_dim,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr_scheduler=lr_scheduler,
    )

    ft_dir = cfg.ft_plms_dir
    _bs_prefix = f"bs{args.batch_size}_" if args.batch_size > 1 else ""
    models_dir = os.path.join(
        ft_dir,
        f"cyber_{os.path.basename(str(plm_path))}",
        f"action_dim_{args.action_dim}_state_dim_{args.state_dim}_sfd_{args.state_feature_dim}",
        f"{_bs_prefix}rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_epochs_{args.num_epochs}_seed_{args.seed}",
    )
    checkpoint_dir = os.path.join(models_dir, "checkpoint")
    best_model_dir = os.path.join(models_dir, "best_model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    console_log = open(os.path.join(models_dir, "console.log"), "w")
    sys.stdout = ConsoleLogger(sys.__stdout__, console_log)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # ABR-style: best model by evaluation return (simulation), not by train loss
    target_return = float(exp_dataset.exp_dataset_info["max_return"]) * args.target_return_scale
    dt_process_reward = make_dt_process_reward(
        exp_dataset.exp_dataset_info["min_reward"],
        exp_dataset.exp_dataset_info["max_reward"],
        args.scale,
    )
    best_eval_return = float("-inf")
    all_losses = []

    for epoch in range(args.num_epochs):
        logs, losses = trainer.train_epoch()
        all_losses.extend(losses)
        print("=" * 20, f"Epoch #{epoch}", "=" * 20)
        print(">" * 10, "Training Information:")
        pprint(logs)

        if epoch % args.save_checkpoint_per_epoch == 0:
            ckpt_dir = os.path.join(checkpoint_dir, str(epoch))
            os.makedirs(ckpt_dir, exist_ok=True)
            save_model(args, model, ckpt_dir)
            print("Checkpoint saved at:", ckpt_dir)

        if epoch % args.eval_per_epoch == 0:
            per_env_logs = {}
            for env_id in args._eval_env_ids:
                env_logs = evaluate_on_env_cyber(
                    args,
                    model,
                    target_return=target_return,
                    max_ep_num=args.eval_episodes,
                    eval_max_steps=args.eval_max_steps,
                    process_reward_fn=dt_process_reward,
                    env_id=env_id,
                )
                per_env_logs[env_id] = env_logs
                print(">" * 10, f"Evaluation Information [env={env_id}]")
                pprint(env_logs)

            agg = _aggregate_eval_returns(per_env_logs)
            if args.model_select_metric == "mean":
                score = agg["episodes_return_mean"]
            else:
                score = agg["episodes_return_min"]
            if best_eval_return < score:
                best_eval_return = score
                save_model(args, model, best_model_dir)
                print("Best model saved at:", best_model_dir)

            print(">" * 10, "Evaluation Aggregate")
            pprint({"metric": args.model_select_metric, "score": score, "best_score": best_eval_return, **agg})

    np.savetxt(os.path.join(models_dir, "train_losses.txt"), np.array(all_losses, dtype=np.float32), fmt="%.6f")
    print("Done. Best eval return:", best_eval_return)
    print("Model dir:", models_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument(
        "--exp-pool-path",
        default=None,
        help="Path to exp_pool.pkl (from csv_to_exp_pool_cyber.py). Mutually exclusive with --exp-pool.",
    )
    exp_group.add_argument(
        "--exp-pool",
        default=None,
        metavar="NAME",
        help=f"Named pool from config (keys: {', '.join(sorted(cfg.exp_pool_paths.keys()))})",
    )
    parser.add_argument("--sample-step", type=int, default=None, help="Stride when sampling windows")

    parser.add_argument("--action-dim", type=int, default=cfg.action_dim)
    parser.add_argument("--state-dim", type=int, default=cfg.state_dim)
    parser.add_argument("--state-feature-dim", type=int, default=cfg.state_feature_dim)

    parser.add_argument("--w", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--scale", type=int, default=1000)

    parser.add_argument("--plm-type", type=str, default="auto", help="For LoRA: gpt2, llama, mistral, opt, t5-lm, etc. Use 'auto' to infer from model.")
    parser.add_argument("--plm-size", type=str, default="base", help="Kept for compatibility; not used.")
    parser.add_argument(
        "--plm-path",
        type=str,
        default=None,
        help="Local path to PLM weights/tokenizer",
    )
    parser.add_argument(
        "--plm-hf-id",
        type=str,
        default=None,
        help="HuggingFace model id to download/cache when --plm-path is missing",
    )
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--which-layer", type=int, default=-1)
    parser.add_argument(
        "--lora-target",
        type=str,
        default="default",
        choices=("default", "attn_qv", "attn_qkvo", "attn_qkvo_mlp"),
        help="LoRA target preset when --rank > 0. 'default' uses architecture mapping in low_rank.py.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha. If omitted, defaults to rank.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--save-checkpoint-per-epoch", type=int, default=1)
    parser.add_argument("--eval-per-epoch", type=int, default=1, help="Run sim evaluation every N epochs (ABR-style)")
    parser.add_argument("--target-return-scale", type=float, default=1.0, help="target_return = max_return * scale for eval")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes per evaluation")
    parser.add_argument("--eval-max-steps", type=int, default=cfg.eval_max_steps_default, help="Max steps per episode in evaluation")
    parser.add_argument("--grad-accum-steps", dest="grad_accum_steps", type=int, default=32)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training micro-batch size (B>1 increases VRAM; grad-accum still merges B*steps for optimizer).",
    )
    parser.add_argument("--seed", type=int, default=100003)

    parser.add_argument("--step-cost", type=float, default=cfg.step_cost, help="Env step cost for evaluation")
    parser.add_argument("--winning-reward", type=int, default=cfg.winning_reward, help="Env winning reward for evaluation")
    parser.add_argument("--ownership-goal", type=float, default=cfg.ownership_goal, help="Env ownership goal for evaluation")
    parser.add_argument("--maximum-node-count", type=int, default=cfg.maximum_node_count, help="Env max node count for evaluation")
    parser.add_argument(
        "--env-id",
        type=str,
        default=cfg.env_id,
        help="Single evaluation environment id during training.",
    )
    parser.add_argument(
        "--eval-env-ids",
        type=str,
        default=None,
        help="Comma-separated env ids for multi-env validation during training.",
    )
    parser.add_argument(
        "--model-select-metric",
        type=str,
        default="mean",
        choices=("mean", "min"),
        help="Best-model selection over multiple envs: mean or min of episodes_return.",
    )

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-out", dest="device_out", default=None)
    parser.add_argument("--device-mid", dest="device_mid", default=None)
    parser.add_argument("--device-map", default=None, help="e.g. 'auto' for large PLMs across devices (used in from_pretrained)")

    args = parser.parse_args()
    if args.plm_type != "auto" and args.plm_type not in cfg.plm_types:
        parser.error(f"--plm-type must be 'auto' or one of {cfg.plm_types}, got {args.plm_type!r}")
    if args.device_out is None:
        args.device_out = args.device
    args.exp_pool_path = _resolve_exp_pool_path(args)
    args._eval_env_ids = _parse_eval_env_ids(args)
    print("Arguments:")
    pprint(args)
    run(args)

