"""Per-step reward transform for Decision Transformer return-conditioning (ABR-aligned).

Training uses ``ExperienceDataset``: rewards are min-max normalized using pool-wide
``min_reward`` / ``max_reward``, then discounted and divided by ``scale``.
Evaluation must apply the same per-step mapping before updating ``target_return``.

Matches ``adaptive_bitrate_streaming/run_plm.py::process_reward``.
"""


def make_dt_process_reward(min_reward: float, max_reward: float, scale: float):
    """
    Return a callable ``process_reward(raw_reward) -> float`` for use in
    ``target_return -= process_reward(env_reward)``.

    If ``max_reward <= min_reward`` (constant-reward pool), returns always 0.0
    (no division by zero).
    """
    mn = float(min_reward)
    mx = float(max_reward)
    sc = float(scale)
    if sc == 0.0:
        raise ValueError("scale must be non-zero for DT reward processing.")

    def process_reward(reward: float) -> float:
        r = float(reward)
        if mx <= mn:
            return 0.0
        r = min(mx, max(mn, r))
        return (r - mn) / (mx - mn) / sc

    return process_reward
