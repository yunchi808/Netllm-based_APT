import torch


def collate_cyber_experience(batch: list):
    """
    Stack samples from CyberExperienceDataset __getitem__ into a training batch.

    Each sample: (states, actions, returns, timesteps, masks)
      states: FloatTensor (T, state_dim)
      actions: list[int] length T
      returns: list[float] length T
      timesteps: list[int] length T
      masks: FloatTensor (T, action_dim) or None (all None or all tensor)
    """
    if not batch:
        raise ValueError("collate_cyber_experience: empty batch")
    states = torch.stack([torch.as_tensor(b[0], dtype=torch.float32) for b in batch], dim=0)
    actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
    returns = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    timesteps = torch.tensor([b[3] for b in batch], dtype=torch.long)
    masks0 = batch[0][4]
    if masks0 is None:
        masks = None
    else:
        masks = torch.stack([torch.as_tensor(b[4], dtype=torch.float32) for b in batch], dim=0)
    return states, actions, returns, timesteps, masks


def process_batch_cyber(batch, device: str, action_dim: int):
    """
    After ``collate_cyber_experience`` (or legacy single-sample batch), move to device
    and build tensors for ``CyberOfflineRLPolicy.forward``.

    states:   (B, T, state_dim)
    actions:  (B, T) long
    returns:  (B, T) float -> (B, T, 1)
    timesteps: (B, T) int32
    masks:    (B, T, action_dim) or None

    Returns:
      states: (B, T, D)
      actions_in: (B, T, 1) normalized to [0,1]
      returns: (B, T, 1)
      timesteps: (B, T)
      labels: (B, T) long
      masks: (B, T, A) float or None
    """
    states, actions, returns, timesteps, masks = batch

    if states.dim() == 2:
        states = states.unsqueeze(0)
    if states.dim() != 3:
        raise ValueError(f"Expected states (B, T, D), got {tuple(states.shape)}")
    states = states.to(device, non_blocking=True)

    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    if actions.dim() == 1:
        actions = actions.unsqueeze(0)
    if actions.dim() != 2:
        raise ValueError(f"Expected actions (B, T), got {tuple(actions.shape)}")
    labels = actions
    actions_in = (actions.float() / float(max(action_dim - 1, 1))).unsqueeze(-1)

    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
    if returns.dim() == 2:
        returns = returns.unsqueeze(-1)
    elif returns.dim() == 3:
        pass
    else:
        raise ValueError(f"Expected returns (B, T) or (B, T, 1), got {tuple(returns.shape)}")

    timesteps = torch.as_tensor(timesteps, dtype=torch.long, device=device)
    if timesteps.dim() == 1:
        timesteps = timesteps.unsqueeze(0)

    if masks is not None:
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        if masks.dim() != 3:
            raise ValueError(f"Expected masks (B, T, A), got {tuple(masks.shape)}")
        if masks.shape[-1] != action_dim:
            raise ValueError(f"mask last dim {masks.shape[-1]} != action_dim {action_dim}")
        masks = masks.to(device, non_blocking=True)
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

