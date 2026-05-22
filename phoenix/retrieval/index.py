from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
import torch

from data.models import Post, Topic
from phoenix.retrieval.towers import CandidateTower

MAX_AGE_HOURS = 72.0  # normalisation constant matches the generator


def post_to_features(post: Post) -> np.ndarray:
    topic_vec = [post.topic_vector.get(t, 0.0) for t in Topic]
    age_hours = (
        (np.datetime64("now") - np.datetime64(post.created_at, "ns")).astype(float)
        / 3_600_000_000_000
    )
    extras = [
        float(post.has_media),
        float(post.has_link),
        float(np.clip(age_hours / MAX_AGE_HOURS, 0.0, 1.0)),
    ]
    return np.array(topic_vec + extras, dtype=np.float32)


class PostIndex:
    def __init__(self, embedding_dim: int = 64):
        self._dim = embedding_dim
        self._index = faiss.IndexFlatIP(embedding_dim)  # inner product = dot product
        self._id_map: list[str] = []                    # FAISS int index → post_id

    # ------------------------------------------------------------------

    def build_index(self, posts: list[Post], candidate_tower: CandidateTower) -> None:
        candidate_tower.eval()
        features = np.stack([post_to_features(p) for p in posts])  # [N, 13]

        with torch.no_grad():
            embeddings = candidate_tower(
                torch.tensor(features, dtype=torch.float32)
            ).numpy()  # [N, embedding_dim]

        faiss.normalize_L2(embeddings)
        self._index.reset()
        self._index.add(embeddings)
        self._id_map = [p.post_id for p in posts]

    def search(self, user_embedding: np.ndarray, top_k: int = 500) -> list[tuple[str, float]]:
        emb = user_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(emb)
        scores, indices = self._index.search(emb, min(top_k, len(self._id_map)))
        return [
            (self._id_map[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".meta"), "wb") as f:
            pickle.dump({"id_map": self._id_map, "dim": self._dim}, f)

    @classmethod
    def load(cls, path: str | Path) -> "PostIndex":
        path = Path(path)
        index = cls.__new__(cls)
        index._index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".meta"), "rb") as f:
            meta = pickle.load(f)
        index._id_map = meta["id_map"]
        index._dim = meta["dim"]
        return index
