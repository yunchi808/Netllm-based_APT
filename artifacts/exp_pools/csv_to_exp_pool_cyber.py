import argparse
import csv
import os
import pickle
import sys
from typing import List, Optional

import numpy as np

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from config import cfg  # noqa: E402
from plm_special.data.exp_pool_cyber import CyberExperiencePool  # noqa: E402


def _parse_state_vector(state_str: str) -> np.ndarray:
    # Example state string contains brackets and newlines:
    # "[1. ,0.1,0. , ...,\n 0. , ...]"
    s = state_str.replace("\n", " ").replace("[", "").replace("]", "").strip()
    vec = np.fromstring(s, sep=",", dtype=np.float32)
    if vec.size == 0:
        raise ValueError(f"Failed parsing state vector from: {state_str[:120]!r}...")
    return vec


def _parse_mask(mask_str: str) -> np.ndarray:
    # mask is stored like: "[1,0,0,...]"
    s = mask_str.strip()
    # fast path: remove brackets and parse ints
    s2 = s.replace("[", "").replace("]", "").replace("\n", " ").strip()
    arr = np.fromstring(s2, sep=",", dtype=np.int8)
    if arr.size == 0:
        raise ValueError(f"Failed parsing action_mask from: {mask_str[:120]!r}...")
    return arr


def convert_csv_to_exp_pool(
    csv_path: str,
    out_path: str,
    max_rows: int = -1,
    keep_action_mask: bool = True,
) -> dict:
    states: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []
    action_masks: Optional[List[np.ndarray]] = [] if keep_action_mask else None

    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        required = {"state", "action", "reward", "done"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}. Found: {reader.fieldnames}")

        for i, row in enumerate(reader):
            if max_rows != -1 and i >= max_rows:
                break

            st = _parse_state_vector(row["state"])
            act = int(float(row["action"]))
            rew = float(row["reward"])
            dn = bool(int(float(row["done"])))

            states.append(st)
            actions.append(act)
            rewards.append(rew)
            dones.append(dn)

            if action_masks is not None:
                if "action_mask" not in row or row["action_mask"] is None:
                    raise ValueError("keep_action_mask=True but CSV has no 'action_mask' column.")
                action_masks.append(_parse_mask(row["action_mask"]))

    pool = CyberExperiencePool(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        action_masks=action_masks,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pool, f)

    state_dim = int(pool.states[0].shape[0]) if len(pool) else 0
    action_dim = int(pool.action_masks[0].shape[0]) if (pool.action_masks and len(pool.action_masks)) else 0
    info = {
        "csv_path": csv_path,
        "out_path": out_path,
        "num_steps": len(pool),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "min_action": int(min(pool.actions)) if len(pool) else None,
        "max_action": int(max(pool.actions)) if len(pool) else None,
        "done_count": int(sum(1 for d in pool.dones if d)),
        "keep_action_mask": keep_action_mask,
    }
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True, help="Path to cyber dataset.csv")
    parser.add_argument(
        "--out-path",
        default=cfg.exp_pool_paths["cyber_default"],
        help="Output pkl path (default: config exp_pool_paths['cyber_default'])",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Max rows to convert (use -1 for all rows)",
    )
    parser.add_argument(
        "--no-action-mask",
        action="store_true",
        help="Do not store action_mask in pkl (not recommended for cyber)",
    )

    args = parser.parse_args()
    info = convert_csv_to_exp_pool(
        csv_path=args.csv_path,
        out_path=args.out_path,
        max_rows=args.max_rows,
        keep_action_mask=not args.no_action_mask,
    )
    print("Converted CSV -> exp_pool.pkl")
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

