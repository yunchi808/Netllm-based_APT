# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .._env import cyberbattle_env
from ..samples.node10_v2 import node10_v2


class CyberBattleNode10V2(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation based on node10_v2."""

    def __init__(self, **kwargs):
        super().__init__(initial_environment=node10_v2.new_environment(), **kwargs)

