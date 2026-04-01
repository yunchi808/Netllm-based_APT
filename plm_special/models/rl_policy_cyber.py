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
        if hasattr(self.plm, "get_base_model"):
            try:
                base = self.plm.get_base_model()
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
        states:     (1, T, state_dim)
        actions_in: (1, T, 1) normalized
        returns:    (1, T, 1)
        timesteps:  (1, T)
        output: logits (1, T, action_dim)
        """
        assert states.shape[0] == 1, "batch size should be 1 to avoid CUDA memory exceed"

        states = states.to(self.device)
        actions_in = actions_in.to(self.device)
        returns = returns.to(self.device)
        timesteps = timesteps.to(self.device)

        time_emb = self.embed_timestep(timesteps)  # (1,T,E)
        r_emb = self.embed_return(returns) + time_emb
        a_emb = self.embed_action(actions_in) + time_emb

        s_feat = self.state_encoder(states)  # (1,T,F)
        s_emb = self.embed_state(s_feat) + time_emb

        # stack as (R_t, S_t, A_t) for each t
        stacked = []
        action_positions = np.zeros(r_emb.shape[1], dtype=np.int64)
        for i in range(r_emb.shape[1]):
            block = torch.cat((r_emb[0, i : i + 1], s_emb[0, i : i + 1], a_emb[0, i : i + 1]), dim=0)
            stacked.append(block)
            # positions in each triple: [0]=R, [1]=S, [2]=A
            # IMPORTANT: predict A_i from S_i representation (ABR-style),
            # so the model cannot "peek" at the current action token A_i.
            action_positions[i] = (i + 1) * 3 - 2  # index of S in flattened sequence
        stacked = torch.cat(stacked, dim=0).unsqueeze(0)  # (1, 3T, E)
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
        try:
            outputs = self.plm(**plm_kwargs, stop_layer_idx=self.which_layer)
        except TypeError:
            outputs = self.plm(**plm_kwargs)
        h = outputs["last_hidden_state"]  # (1, L, E)

        # map action_positions into truncated sequence coordinates
        L = h.shape[1]
        full_L = 3 * r_emb.shape[1]
        offset = max(full_L - L, 0)
        pos = torch.as_tensor(action_positions - offset, device=self.device)
        pos = torch.clamp(pos, 0, L - 1)
        logits_used = h[:, pos]  # (1,T,E)
        logits_used = logits_used.to(dtype=self.action_head.weight.dtype)
        logits = self.action_head(logits_used)  # (1,T,A)
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

