import numpy as np
import torch

from plm_special.data.dataset import ExperienceDataset


class CyberExperienceDataset(ExperienceDataset):
    """
    ExperienceDataset + action_mask support.

    Expects exp_pool to have:
    - states: list[np.ndarray] or list[torch.Tensor]
    - actions: list[int]
    - rewards: list[float]
    - dones: list[bool]
    - action_masks: list[np.ndarray] (optional but recommended)
    """

    @property
    def action_masks(self):
        return getattr(self.exp_pool, "action_masks", None)

    def __getitem__(self, index):
        start = self.dataset_indices[index]
        end = start + self.max_length

        states = self.states[start:end]
        if isinstance(states[0], np.ndarray):
            # -> (seq_len, state_dim)
            states = torch.as_tensor(np.stack(states, axis=0), dtype=torch.float32)
        elif torch.is_tensor(states[0]):
            states = torch.stack(states, dim=0).float()
        else:
            raise TypeError(f"Unsupported state element type: {type(states[0])}")

        actions = self.actions[start:end]
        returns = self.returns[start:end]
        timesteps = self.timesteps[start:end]

        masks = None
        if self.action_masks is not None:
            ms = self.action_masks[start:end]
            if isinstance(ms[0], np.ndarray):
                masks = torch.as_tensor(np.stack(ms, axis=0), dtype=torch.float32)
            elif torch.is_tensor(ms[0]):
                masks = torch.stack(ms, dim=0).float()
            else:
                raise TypeError(f"Unsupported action_mask element type: {type(ms[0])}")

        return states, actions, returns, timesteps, masks

