from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.generator import DataGenerator
from data.models import Topic, User, Post
from phoenix.retrieval.towers import TwoTowerModel

MAX_AGE_HOURS = 72.0


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def user_to_features(user: User) -> list[float]:
    return [user.interest_vector.get(t, 0.0) for t in Topic]


def post_to_features(post: Post) -> list[float]:
    topic_vec = [post.topic_vector.get(t, 0.0) for t in Topic]
    from datetime import datetime, timezone
    age_h = (datetime.utcnow() - post.created_at).total_seconds() / 3600
    extras = [
        float(post.has_media),
        float(post.has_link),
        float(min(age_h / MAX_AGE_HOURS, 1.0)),
    ]
    return topic_vec + extras


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RetrievalDataset(Dataset):
    def __init__(self, data_path: str | Path, negatives_per_positive: int = 4):
        gen = DataGenerator.load_from_json(data_path)

        user_map: dict[str, User] = {u.user_id: u for u in gen.users}
        post_map: dict[str, Post] = {p.post_id: p for p in gen.posts}
        all_posts = gen.posts

        samples: list[tuple[list[float], list[float], float]] = []

        for eng in gen.engagements:
            user = user_map.get(eng.user_id)
            post = post_map.get(eng.post_id)
            if user is None or post is None:
                continue

            samples.append((user_to_features(user), post_to_features(post), 1.0))

            for _ in range(negatives_per_positive):
                neg_post = random.choice(all_posts)
                samples.append((user_to_features(user), post_to_features(neg_post), 0.0))

        random.shuffle(samples)
        self._user_feats = torch.tensor([s[0] for s in samples], dtype=torch.float32)
        self._post_feats = torch.tensor([s[1] for s in samples], dtype=torch.float32)
        self._labels = torch.tensor([s[2] for s in samples], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        return self._user_feats[idx], self._post_feats[idx], self._labels[idx]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: TwoTowerModel, loader: DataLoader) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for user_f, post_f, labels in loader:
            scores = model(user_f, post_f)
            total_loss += criterion(scores, labels).item() * len(labels)
            preds = (scores > 0).float()
            correct += (preds == labels).sum().item()
            n += len(labels)

    return {"loss": total_loss / n, "accuracy": correct / n}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_retrieval_model(
    data_path: str | Path,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    save_path: str | Path = "models/retrieval.pt",
) -> TwoTowerModel:
    dataset = RetrievalDataset(data_path)
    split = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TwoTowerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for user_f, post_f, labels in train_loader:
            optimizer.zero_grad()
            scores = model(user_f, post_f)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, val_loader)
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"train_loss={total_loss/len(train_ds):.4f}  "
                f"val_loss={metrics['loss']:.4f}  "
                f"val_acc={metrics['accuracy']:.3f}"
            )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    return model


if __name__ == "__main__":
    train_retrieval_model("data/generated/dataset.json")
