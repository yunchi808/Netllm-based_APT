import numpy as np
from .state_action_tools import AbstractAttackerModel


class RAND_attacker(AbstractAttackerModel):
    def __init__(self, stateaction_model):
        super().__init__(stateaction_model)

    def choose_abstract_action(self, current_state, action_mask, test_flag=False):
        valid_actions = [i for i in range(self.action_dim) if action_mask[i]]
        action = np.random.choice(valid_actions)
        return action, None
