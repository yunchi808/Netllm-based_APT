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
def _is_registered(env_id: str) -> bool:
    # gym<=0.21 had registry.env_specs, gym>=0.26 uses registry as dict-like
    if hasattr(registry, "env_specs"):  # pragma: no cover
        return env_id in registry.env_specs  # type: ignore[attr-defined]
    return env_id in registry  # type: ignore[operator]


def ensure_registered():
    """Register APT cyber environments used by training/evaluation."""
    common_kwargs = {
        "defender_agent": None,
        "attacker_goal": AttackerGoal(own_atleast_percent=cfg.ownership_goal),
        "defender_goal": DefenderGoal(eviction=True),
        "step_cost": cfg.step_cost,
        "winning_reward": cfg.winning_reward,
        "maximum_node_count": cfg.maximum_node_count,
    }

    env_specs = {
        "AptCyberBattleToyCtf-v0": "apt_cyber_sim._env.cyberbattle_toyctf:CyberBattleToyCtf",
        "AptCyberBattleNode10V1-v0": "apt_cyber_sim._env.cyberbattle_node10_v1:CyberBattleNode10V1",
        "AptCyberBattleNode10V2-v0": "apt_cyber_sim._env.cyberbattle_node10_v2:CyberBattleNode10V2",
        "AptCyberBattleNode10V3-v0": "apt_cyber_sim._env.cyberbattle_node10_v3:CyberBattleNode10V3",
    }
    for env_id, entry_point in env_specs.items():
        if _is_registered(env_id):
            continue
        gym_register(
            id=env_id,
            entry_point=entry_point,
            kwargs=common_kwargs,
        )


# Register on import for convenience (mirrors upstream behavior)
ensure_registered()

