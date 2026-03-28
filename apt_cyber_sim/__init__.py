# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""APT vendored CyberBattleSim (minimal).

This package is a self-contained copy of the CyberBattleSim environment code
needed by the automated_penetration_test task. It intentionally uses the
package name `apt_cyber_sim` to avoid conflicts with any installed `cyberbattle`
package and to avoid importing from other folders.
"""

import os
import sys

# Resolve APT config when `apt_cyber_sim` is on sys.path as a top-level package.
_APT_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APT_PARENT not in sys.path:
    sys.path.insert(0, _APT_PARENT)

from gym.envs.registration import registry, register as gym_register

from config import cfg  # noqa: E402

from ._env.cyberbattle_env import AttackerGoal, DefenderGoal
from .samples.toyctf import toy_ctf
def _is_registered(env_id: str) -> bool:
    # gym<=0.21 had registry.env_specs, gym>=0.26 uses registry as dict-like
    if hasattr(registry, "env_specs"):  # pragma: no cover
        return env_id in registry.env_specs  # type: ignore[attr-defined]
    return env_id in registry  # type: ignore[operator]


def ensure_registered():
    """Register the ToyCTF env used by APT evaluation."""
    if _is_registered(cfg.env_id):
        return

    # Defaults from automated_penetration_test.config (aligned with offline CSV generation).
    gym_register(
        id=cfg.env_id,
        entry_point="apt_cyber_sim._env.cyberbattle_toyctf:CyberBattleToyCtf",
        kwargs={
            "defender_agent": None,
            "attacker_goal": AttackerGoal(own_atleast_percent=cfg.ownership_goal),
            "defender_goal": DefenderGoal(eviction=True),
            "step_cost": cfg.step_cost,
            "winning_reward": cfg.winning_reward,
            "maximum_node_count": cfg.maximum_node_count,
        },
    )


# Register on import for convenience (mirrors upstream behavior)
ensure_registered()

