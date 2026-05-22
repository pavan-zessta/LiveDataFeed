from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.generator import DataGenerator
from data.models import ActionType, Topic, User, Post
from phoenix.scorer.transformer import PhoenixScorer

N_ACTIONS = len(ActionType)
ACTION_LIST = list(ActionType)
MAX_HISTORY = 20
MAX_AGE_HOURS = 72.0


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def post_to_features(post: Post) -> list[float]:
    topic_vec = [post.topic_vector.get(t, 0.0) for t in Topic]
    age_h = (datetime.utcnow() - post.created_at).total_seconds() / 3600
    return topic_vec + [
        float(post.has_media),
        float(post.has_link),
        float(min(age_h / MAX_AGE_HOURS, 1.0)),
    ]


def action_to_onehot(action: ActionType) -> list[float]:
    vec = [0.0] * N_ACTIONS
    vec[ACTION_LIST.index(action)] = 1.0
    return vec


def history_token(action: ActionType, post: Post) -> list[float]:
    return action_to_onehot(action) + post_to_features(post)  # 12 + 13 = 25


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ScorerDataset(Dataset):
    def __init__(self, data_path: str | Path):
        gen = DataGenerator.load_from_json(data_path)

        user_map = {u.user_id: u for u in gen.users}
        post_map = {p.post_id: p for p in gen.posts}

        # Group engagements by user
        from collections import defaultdict
        user_engagements: dict[str, list] = defaultdict(list)
        for eng in gen.engagements:
            if eng.user_id in user_map and eng.post_id in post_map:
                user_engagements[eng.user_id].append(eng)

        self._history_feats: list[torch.Tensor] = []
        self._candidate_feats: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

        for user_id, engs in user_engagements.items():
            if len(engs) < 2:
                continue

            # Sort by timestamp so history is chronological
            engs.sort(key=lambda e: e.timestamp)

            for i in range(1, len(engs)):
                # History = all prior engagements (capped at MAX_HISTORY)
                history = engs[max(0, i - MAX_HISTORY):i]
                target_eng = engs[i]
                candidate_post = post_map[target_eng.post_id]

                # Pad or truncate history to MAX_HISTORY
                hist_tokens = [
                    history_token(e.action, post_map[e.post_id])
                    for e in history
                ]
                while len(hist_tokens) < MAX_HISTORY:
                    hist_tokens.insert(0, [0.0] * 25)  # zero-pad at front

                # Build multi-hot label from all actions this user took on this post
                label = [0.0] * N_ACTIONS
                for eng in engs:
                    if eng.post_id == target_eng.post_id:
                        label[ACTION_LIST.index(eng.action)] = 1.0

                self._history_feats.append(torch.tensor(hist_tokens, dtype=torch.float32))
                self._candidate_feats.append(
                    torch.tensor(post_to_features(candidate_post), dtype=torch.float32)
                )
                self._labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        return self._history_feats[idx], self._candidate_feats[idx], self._labels[idx]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: PhoenixScorer, loader: DataLoader) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for hist_f, cand_f, labels in loader:
            logits = model(hist_f, cand_f)
            total_loss += criterion(logits, labels).item() * len(labels)
            n += len(labels)

    return {"val_loss": total_loss / n}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_scorer(
    data_path: str | Path,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 256,
    save_path: str | Path = "models/scorer.pt",
) -> PhoenixScorer:
    dataset = ScorerDataset(data_path)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty — run generate_data.py first.")

    split = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = PhoenixScorer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training scorer on {len(train_ds)} samples, validating on {len(val_ds)}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for hist_f, cand_f, labels in train_loader:
            optimizer.zero_grad()
            logits = model(hist_f, cand_f)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_loader)
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"train_loss={total_loss/len(train_ds):.4f}  "
                f"val_loss={metrics['val_loss']:.4f}"
            )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    return model


if __name__ == "__main__":
    train_scorer("data/generated/dataset.json")
