from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class CyberExperiencePool:
    """
    ABR-compatible experience pool for cyber trajectories.

    Keeps the same core field names as ABR's ExperiencePool so that generic dataset
    logic can reuse them: states/actions/rewards/dones.
    """

    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    action_masks: Optional[List[np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.states)

