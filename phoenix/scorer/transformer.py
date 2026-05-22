from __future__ import annotations

import torch
import torch.nn as nn

from data.models import ActionType, Topic

N_TOPICS = len(Topic)        # 10
N_ACTIONS = len(ActionType)  # 12
POST_FEAT_DIM = N_TOPICS + 3  # topic vec + has_media + has_link + age_norm


def build_attention_mask(n_history: int, n_candidates: int) -> torch.Tensor:
    """
    Returns additive mask of shape [seq_len, seq_len].
    0.0 = allowed to attend, -inf = blocked.

    History tokens  : can attend to other history tokens only
    Candidate tokens: can attend to all history + itself only (isolation)
    """
    seq_len = n_history + n_candidates
    # Start fully blocked
    mask = torch.full((seq_len, seq_len), float("-inf"))

    # History attends to all history
    mask[:n_history, :n_history] = 0.0

    # Each candidate attends to all history + itself only
    for i in range(n_candidates):
        ci = n_history + i
        mask[ci, :n_history] = 0.0  # can see history
        mask[ci, ci] = 0.0          # can see itself

    return mask


class PhoenixScorer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_actions: int = N_ACTIONS,
        max_history: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_history = max_history

        # Project raw features into d_model space
        # History token: action one-hot (12) + post features (13) = 25
        self.history_proj = nn.Linear(N_ACTIONS + POST_FEAT_DIM, d_model)
        # Candidate token: post features only (13)
        self.candidate_proj = nn.Linear(POST_FEAT_DIM, d_model)

        self.position_embedding = nn.Embedding(max_history + 1, d_model)  # +1 for candidate slot

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # One binary output head per action
        self.action_heads = nn.Linear(d_model, n_actions)

    def forward(
        self,
        history_features: torch.Tensor,   # [batch, n_history, 25]
        candidate_features: torch.Tensor,  # [batch, 13]
    ) -> torch.Tensor:
        """
        Returns logits of shape [batch, n_actions].
        Apply sigmoid externally for probabilities.
        """
        batch = history_features.size(0)
        n_history = history_features.size(1)

        # Project tokens
        hist_tokens = self.history_proj(history_features)   # [batch, n_history, d_model]
        cand_token = self.candidate_proj(candidate_features).unsqueeze(1)  # [batch, 1, d_model]

        # Positional embeddings
        hist_pos = torch.arange(n_history, device=hist_tokens.device)
        cand_pos = torch.tensor([n_history], device=cand_token.device)
        hist_tokens = hist_tokens + self.position_embedding(hist_pos)
        cand_token = cand_token + self.position_embedding(cand_pos)

        # Concatenate: [history..., candidate]
        seq = torch.cat([hist_tokens, cand_token], dim=1)  # [batch, n_history+1, d_model]

        # Candidate isolation mask
        mask = build_attention_mask(n_history, n_candidates=1).to(seq.device)

        out = self.transformer(seq, mask=mask)  # [batch, n_history+1, d_model]

        # Use only the candidate token's output for prediction
        candidate_out = out[:, -1, :]           # [batch, d_model]
        return self.action_heads(candidate_out)  # [batch, n_actions]

    def predict_probs(
        self,
        history_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> dict[ActionType, float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                history_features.unsqueeze(0),
                candidate_features.unsqueeze(0),
            )
            probs = torch.sigmoid(logits).squeeze(0)
        return {action: probs[i].item() for i, action in enumerate(ActionType)}
