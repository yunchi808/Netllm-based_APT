from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CyberOfflineRLPolicy(nn.Module):
    """
    Cyber variant of OfflineRLPolicy:
    - state is a vector (state_dim)
    - optional action_mask is applied in loss (handled outside) and can be used for sampling
    """

    def __init__(
        self,
        state_feature_dim: int,
        action_dim: int,
        state_encoder: nn.Module,
        plm: nn.Module,
        plm_embed_size: int,
        max_length: int,
        max_ep_len: int,
        device: str,
        device_out: str | None = None,
        which_layer: int = -1,
    ):
        super().__init__()
        if device_out is None:
            device_out = device

        self.action_dim = action_dim
        self.max_length = max_length
        self.device = device
        self.device_out = device_out
        self.which_layer = which_layer

        self.plm = plm
        self.plm_embed_size = plm_embed_size
        self.state_encoder = state_encoder

        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)
        self.embed_state = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)

        self.action_head = nn.Linear(plm_embed_size, action_dim).to(device)

        self.modules_except_plm = nn.ModuleList(
            [self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, self.embed_state, self.embed_ln, self.action_head]
        )

        # --- rollout context (DT-style) ---
        # During evaluation we need to condition action prediction on previous actions (a_{<i}).
        # We therefore keep (R,S,A) token embeddings in deques, and build the same token order as in `forward`:
        #   [R_0, S_0, A_0, R_1, S_1, A_1, ...] then for current step we append [R_t, S_t] and predict from S_t.
        self.states_dq: deque[torch.Tensor] = deque(maxlen=max_length)
        self.returns_dq: deque[torch.Tensor] = deque(maxlen=max_length)
        self.actions_dq: deque[torch.Tensor] = deque(maxlen=max_length)

    def _plm_input_dtype(self) -> torch.dtype:
        """Get the base PLM compute dtype (robust with PEFT wrappers)."""
        if hasattr(self.plm, "get_input_embeddings"):
            try:
                emb = self.plm.get_input_embeddings()
                if emb is not None and hasattr(emb, "weight"):
                    return emb.weight.dtype
            except Exception:
                pass
        if hasattr(self.plm, "get_base_model"):
            try:
                base = self.plm.get_base_model()
                if hasattr(base, "get_input_embeddings"):
                    emb = base.get_input_embeddings()
                    if emb is not None and hasattr(emb, "weight"):
                        return emb.weight.dtype
                return next(base.parameters()).dtype
            except Exception:
                pass
        return next(self.plm.parameters()).dtype

    def clear_dq(self) -> None:
        """Clear rollout token history (call at each env.reset)."""
        self.states_dq.clear()
        self.returns_dq.clear()
        self.actions_dq.clear()

    def forward(self, states, actions_in, returns, timesteps, attention_mask=None):
        """
        states:     (B, T, state_dim)
        actions_in: (B, T, 1) normalized
        returns:    (B, T, 1)
        timesteps:  (B, T)
        output: logits (B, T, action_dim)
        """
        states = states.to(self.device)
        actions_in = actions_in.to(self.device)
        returns = returns.to(self.device)
        timesteps = timesteps.to(self.device)

        time_emb = self.embed_timestep(timesteps)  # (B,T,E)
        r_emb = self.embed_return(returns) + time_emb
        a_emb = self.embed_action(actions_in) + time_emb

        s_feat = self.state_encoder(states)  # (B,T,F)
        s_emb = self.embed_state(s_feat) + time_emb

        # Flatten (R_t, S_t, A_t) per timestep -> (B, 3T, E); order matches prior loop:
        # R_0,S_0,A_0, R_1,S_1,A_1, ...
        stacked = torch.stack((r_emb, s_emb, a_emb), dim=2).reshape(states.shape[0], 3 * r_emb.shape[1], -1)
        stacked = stacked[:, -self.plm_embed_size :, :]
        stacked = self.embed_ln(stacked)
        plm_dtype = self._plm_input_dtype()
        stacked = stacked.to(dtype=plm_dtype)

        if attention_mask is None:
            attention_mask = torch.ones((stacked.shape[0], stacked.shape[1]), dtype=torch.long, device=self.device)

        plm_kwargs = dict(
            inputs_embeds=stacked,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        use_autocast = self.device.startswith("cuda") and plm_dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=plm_dtype, enabled=use_autocast):
            try:
                outputs = self.plm(**plm_kwargs, stop_layer_idx=self.which_layer)
            except TypeError:
                outputs = self.plm(**plm_kwargs)
        h = outputs["last_hidden_state"]  # (B, L, E)

        T = r_emb.shape[1]
        full_L = 3 * T
        # index of S_i in the full (pre-truncate) token sequence: (i+1)*3 - 2
        action_positions = torch.arange(T, device=self.device, dtype=torch.long) * 3 + 1
        L = h.shape[1]
        offset = max(full_L - L, 0)
        pos = action_positions - offset
        pos = torch.clamp(pos, 0, L - 1)
        logits_used = h[:, pos, :]  # (B,T,E)
        logits_used = logits_used.to(dtype=self.action_head.weight.dtype)
        logits = self.action_head(logits_used)  # (B,T,A)
        return logits

    def sample(self, state, target_return, timestep, action_mask=None):
        """
        Minimal sampling helper (optional). For now we keep training-only entrypoint,
        but this is useful once you add cyber env evaluation.
        """
        state = state.to(self.device)  # (1,1,D)
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)

        time_emb = self.embed_timestep(timestep)
        r_emb = self.embed_return(target_return) + time_emb

        s_feat = self.state_encoder(state)
        s_emb = self.embed_state(s_feat) + time_emb

        # Build DT token sequence with previous (R,S,A) blocks:
        #   prev: [R_0, S_0, A_0, ..., R_{t-1}, S_{t-1}, A_{t-1}]
        #   now:  [R_t, S_t]  (predict from S_t; do NOT include A_t)
        prev_blocks = []
        for i in range(len(self.states_dq)):
            prev_blocks.append(torch.cat((self.returns_dq[i], self.states_dq[i], self.actions_dq[i]), dim=1))  # (1,3,E)
        if prev_blocks:
            prev_seq = torch.cat(prev_blocks, dim=1)  # (1,3H,E)
            stacked = torch.cat((prev_seq, torch.cat((r_emb, s_emb), dim=1)), dim=1)  # (1,3H+2,E)
        else:
            stacked = torch.cat((r_emb, s_emb), dim=1)  # (1,2,E)

        stacked = self.embed_ln(stacked)
        plm_dtype = self._plm_input_dtype()
        stacked = stacked.to(dtype=plm_dtype)
        attn = torch.ones((1, stacked.shape[1]), dtype=torch.long, device=self.device)
        plm_kwargs = dict(
            inputs_embeds=stacked,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
        )
        use_autocast = self.device.startswith("cuda") and plm_dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=plm_dtype, enabled=use_autocast):
            try:
                out = self.plm(**plm_kwargs, stop_layer_idx=self.which_layer)
            except TypeError:
                out = self.plm(**plm_kwargs)
        h = out["last_hidden_state"][:, -1:]  # (1,1,E)
        h = h.to(dtype=self.action_head.weight.dtype)
        logits = self.action_head(h).reshape(-1)  # (A,)

        if action_mask is not None:
            m = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).reshape(-1)
            logits = logits.masked_fill(m <= 0.0, -1e9)

        pi = F.softmax(logits, dim=0).detach().cpu().numpy()
        idx = int(np.random.choice(np.arange(pi.size), p=pi))

        # After sampling A_t, push (R_t,S_t,A_t) into deque for future steps.
        denom = float(max(self.action_dim - 1, 1))
        a_in = torch.tensor([[[idx / denom]]], dtype=torch.float32, device=self.device)  # (1,1,1) normalized
        a_emb = self.embed_action(a_in) + time_emb  # (1,1,E)
        self.returns_dq.append(r_emb)
        self.states_dq.append(s_emb)
        self.actions_dq.append(a_emb)

        return idx

