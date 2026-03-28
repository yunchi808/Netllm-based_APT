import numpy as np
import torch


def process_batch_cyber(batch, device: str, action_dim: int):
    """
    Batch format from CyberExperienceDataset:
      states: torch.FloatTensor (seq_len, state_dim)
      actions: list[int] length seq_len
      returns: list[float] length seq_len
      timesteps: list[int] length seq_len
      masks: torch.FloatTensor (seq_len, action_dim) or None

    Returns shapes aligned with CyberOfflineRLPolicy:
      states: (1, seq_len, state_dim)
      actions_in: (1, seq_len, 1) normalized to [0,1]
      returns: (1, seq_len, 1)
      timesteps: (1, seq_len)
      labels: (1, seq_len) long
      masks: (1, seq_len, action_dim) float or None
    """
    states, actions, returns, timesteps, masks = batch

    # DataLoader with batch_size=1 will add an outer batch dim:
    # states: (1, T, D), masks: (1, T, A)
    if states.dim() == 3 and states.shape[0] == 1:
        states = states.to(device)
    elif states.dim() == 2:
        states = states.unsqueeze(0).to(device)
    else:
        raise ValueError(f"Expected states with shape (T, D) or (1, T, D), got {tuple(states.shape)}")

    actions = torch.as_tensor(actions, dtype=torch.long, device=device).reshape(1, -1)
    labels = actions
    actions_in = (actions.float() / float(max(action_dim - 1, 1))).unsqueeze(2)  # (1, T, 1)

    returns = torch.as_tensor(returns, dtype=torch.float32, device=device).reshape(1, -1, 1)
    timesteps = torch.as_tensor(timesteps, dtype=torch.int32, device=device).reshape(1, -1)

    if masks is not None:
        if masks.dim() == 3 and masks.shape[0] == 1:
            pass
        elif masks.dim() == 2:
            masks = masks.unsqueeze(0)
        else:
            raise ValueError(f"Expected masks with shape (T, A) or (1, T, A), got {tuple(masks.shape)}")
        if masks.shape[-1] != action_dim:
            raise ValueError(f"mask last dim {masks.shape[-1]} != action_dim {action_dim}")
        masks = masks.to(device)
        masks = (masks > 0.5).to(torch.float32)

    return states, actions_in, returns, timesteps, labels, masks


def masked_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None):
    """
    logits: (B, T, A)
    labels: (B, T)
    mask: (B, T, A) where 1=valid, 0=invalid
    """
    if mask is not None:
        # set invalid actions to a very negative number
        logits = logits.masked_fill(mask <= 0.0, -1e9)

    # flatten for CE
    B, T, A = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, A),
        labels.reshape(B * T),
    )
    return loss

