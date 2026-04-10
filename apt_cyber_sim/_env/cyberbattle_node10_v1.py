# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .._env import cyberbattle_env
from ..samples.node10_v1 import node10_v1


class CyberBattleNode10V1(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation based on node10_v1."""

    def __init__(self, **kwargs):
        super().__init__(initial_environment=node10_v1.new_environment(), **kwargs)

