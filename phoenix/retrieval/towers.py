from __future__ import annotations

import torch
import torch.nn as nn


class UserTower(nn.Module):
    def __init__(self, input_dim: int = 10, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [batch, embedding_dim]


class CandidateTower(nn.Module):
    def __init__(self, input_dim: int = 13, embedding_dim: int = 64):
        super().__init__()
        # input: 10 topic floats + has_media + has_link + post_age_norm
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [batch, embedding_dim]


class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim: int = 10, post_input_dim: int = 13, embedding_dim: int = 64):
        super().__init__()
        self.user_tower = UserTower(user_input_dim, embedding_dim)
        self.candidate_tower = CandidateTower(post_input_dim, embedding_dim)

    def forward(self, user_features: torch.Tensor, post_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_tower(user_features)      # [batch, embedding_dim]
        post_emb = self.candidate_tower(post_features)  # [batch, embedding_dim]
        return (user_emb * post_emb).sum(-1)            # dot product → [batch]

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_features)

    def encode_post(self, post_features: torch.Tensor) -> torch.Tensor:
        return self.candidate_tower(post_features)
