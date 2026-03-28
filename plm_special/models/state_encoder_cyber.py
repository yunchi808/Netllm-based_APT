import torch
import torch.nn as nn


class CyberStateEncoder(nn.Module):
    """
    Simple MLP encoder for vector state (state_dim -> state_feature_dim).
    Input:  (B, T, state_dim)
    Output: (B, T, state_feature_dim)
    """

    def __init__(self, state_dim: int, state_feature_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.state_feature_dim = state_feature_dim

        hidden = max(128, state_feature_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_feature_dim),
            nn.GELU(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 3:
            raise ValueError(f"Expected states (B,T,D), got {tuple(states.shape)}")
        return self.net(states)

