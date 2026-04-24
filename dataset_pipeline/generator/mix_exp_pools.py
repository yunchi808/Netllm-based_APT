import argparse
import pickle
from pathlib import Path
from typing import List, Tuple
import sys

# Allow running as a script:
# python dataset_pipeline/generator/mix_exp_pools.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plm_special.data.exp_pool_cyber import CyberExperiencePool


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _normalize_ratios(values: List[float]) -> List[float]:
    s = float(sum(values))
    if s <= 0:
        raise ValueError("ratio sum must be positive")
    return [float(v) / s for v in values]


def _load_pool(path: Path) -> CyberExperiencePool:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, CyberExperiencePool):
        raise TypeError(f"{path} is not CyberExperiencePool")
    return obj


def _take_prefix(pool: CyberExperiencePool, n: int):
    states = list(pool.states[:n])
    actions = list(pool.actions[:n])
    rewards = list(pool.rewards[:n])
    dones = list(pool.dones[:n])
    masks = None
    if getattr(pool, "action_masks", None) is not None:
        masks = list(pool.action_masks[:n])
    if len(dones) > 0 and not bool(dones[-1]):
        dones[-1] = True
    return states, actions, rewards, dones, masks


def _compute_target_total(lengths: List[int], mode: str) -> int:
    if mode == "min":
        return int(min(lengths))
    if mode == "max":
        return int(max(lengths))
    if mode == "mean":
        return int(round(sum(lengths) / len(lengths)))
    raise ValueError(f"unknown control_total mode: {mode}")


def _merge(
    pools: List[CyberExperiencePool],
    ratios: List[float],
    target_total: int,
) -> Tuple[CyberExperiencePool, List[int]]:
    picks = [int(round(target_total * r)) for r in ratios]
    diff = target_total - sum(picks)
    picks[-1] += diff

    for i, pool in enumerate(pools):
        picks[i] = min(picks[i], len(pool.states))

    masks_available = [getattr(pool, "action_masks", None) is not None for pool in pools]
    if len(set(masks_available)) != 1:
        raise ValueError("action_masks availability mismatch between pools")

    all_states, all_actions, all_rewards, all_dones = [], [], [], []
    all_masks = [] if masks_available[0] else None

    for pool, n in zip(pools, picks):
        s, a, r, d, m = _take_prefix(pool, n)
        all_states.extend(s)
        all_actions.extend(a)
        all_rewards.extend(r)
        all_dones.extend(d)
        if all_masks is not None and m is not None:
            all_masks.extend(m)

    mix = CyberExperiencePool(
        states=all_states,
        actions=all_actions,
        rewards=all_rewards,
        dones=all_dones,
        action_masks=all_masks,
    )
    return mix, picks


def main():
    parser = argparse.ArgumentParser(description="Mix multiple CyberExperiencePool pkl files by ratio.")
    parser.add_argument("--input-paths", required=True, help="Comma-separated pkl paths")
    parser.add_argument("--ratios", required=True, help="Comma-separated ratios, e.g. 0.5,0.5 or 7,3")
    parser.add_argument(
        "--out-path",
        required=True,
        help="Output mixed pkl path. Relative paths are saved under artifacts/exp_pools/",
    )
    parser.add_argument(
        "--control-total",
        default="min",
        choices=["min", "max", "mean"],
        help="How to choose target total size from source pool sizes",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=0,
        help="Optional explicit target total steps (overrides --control-total when > 0)",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in _parse_csv_list(args.input_paths)]
    ratio_values = [float(x) for x in _parse_csv_list(args.ratios)]
    if len(input_paths) < 2:
        raise ValueError("need at least 2 input pools")
    if len(input_paths) != len(ratio_values):
        raise ValueError("number of input paths must equal number of ratios")

    ratios = _normalize_ratios(ratio_values)
    pools = [_load_pool(p) for p in input_paths]
    lengths = [len(p.states) for p in pools]
    target_total = int(args.target_total) if args.target_total > 0 else _compute_target_total(lengths, args.control_total)

    mix, picks = _merge(pools, ratios, target_total)
    out = Path(args.out_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / "artifacts" / "exp_pools" / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(mix, f)

    print("Mixed pool generated")
    for i, p in enumerate(input_paths):
        print(f"  src{i}: {p} total={lengths[i]} picked={picks[i]} ratio={ratios[i]:.6f}")
    print(f"  mixed_total={len(mix.states)}")
    if len(mix.states) > 0:
        print(f"  state_dim={len(mix.states[0])}")
    actions = [int(a) for a in mix.actions] if len(mix.actions) > 0 else []
    if actions:
        print(f"  action_min={min(actions)} action_max={max(actions)} action_dim~={max(actions) + 1}")
    print(f"  done_count={sum(bool(x) for x in mix.dones)}")
    print(f"  output={out.resolve()}")


if __name__ == "__main__":
    main()
