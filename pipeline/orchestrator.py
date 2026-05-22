from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import torch

from data.models import ActionType, Post, ScoredCandidate, Topic, User
from phoenix.retrieval.index import PostIndex, post_to_features
from phoenix.retrieval.towers import TwoTowerModel
from phoenix.scorer.transformer import PhoenixScorer
from phoenix.scorer.train import post_to_features as scorer_post_features
from phoenix.scorer.train import history_token, MAX_HISTORY, ACTION_LIST
from phoenix.scorer.weights import ACTION_WEIGHTS
from pipeline.filters import apply_pre_scoring_filters, filter_author_diversity
from pipeline.scorers import apply_author_diversity, compute_weighted_score, select_top_k
from thunder.store import ThunderStore

MAX_AGE_HOURS = 72.0
N_ACTIONS = len(ActionType)


def _user_to_tensor(user: User) -> torch.Tensor:
    vec = [user.interest_vector.get(t, 0.0) for t in Topic]
    return torch.tensor(vec, dtype=torch.float32)


def _build_history_tensor(
    user: User,
    post_map: dict[str, Post],
    engagement_history: list[tuple[ActionType, str]],  # [(action, post_id), ...]
) -> torch.Tensor:
    tokens = []
    for action, post_id in engagement_history[-MAX_HISTORY:]:
        post = post_map.get(post_id)
        if post:
            tokens.append(history_token(action, post))

    # Zero-pad to MAX_HISTORY
    while len(tokens) < MAX_HISTORY:
        tokens.insert(0, [0.0] * (N_ACTIONS + 13))

    return torch.tensor(tokens, dtype=torch.float32)


class FeedOrchestrator:
    def __init__(
        self,
        thunder_store: ThunderStore,
        retrieval_model: TwoTowerModel,
        faiss_index: PostIndex,
        scorer_model: PhoenixScorer,
        post_map: dict[str, Post],
    ):
        self._thunder = thunder_store
        self._retrieval = retrieval_model
        self._index = faiss_index
        self._scorer = scorer_model
        self._post_map = post_map

        # In-memory engagement history per user: {user_id: [(action, post_id), ...]}
        self._engagement_history: dict[str, list[tuple[ActionType, str]]] = {}
        # Posts the user has already been served: {user_id: set[post_id]}
        self._seen: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Engagement recording
    # ------------------------------------------------------------------

    def record_engagement(self, user_id: str, post_id: str, action: ActionType) -> None:
        self._engagement_history.setdefault(user_id, []).append((action, post_id))

    # ------------------------------------------------------------------
    # Feed generation
    # ------------------------------------------------------------------

    async def generate_feed(
        self,
        user: User,
        k: int = 50,
    ) -> list[ScoredCandidate]:

        # Step 1 — Source candidates in parallel
        thunder_task = asyncio.to_thread(self._thunder.get_in_network_candidates, user)

        user_tensor = _user_to_tensor(user)
        self._retrieval.eval()
        with torch.no_grad():
            user_emb = self._retrieval.encode_user(user_tensor.unsqueeze(0)).squeeze(0).numpy()

        phoenix_task = asyncio.to_thread(self._index.search, user_emb, 500)

        thunder_posts, phoenix_results = await asyncio.gather(thunder_task, phoenix_task)

        # Step 2 — Merge into ScoredCandidate list
        candidates: list[ScoredCandidate] = []

        for post in thunder_posts:
            candidates.append(ScoredCandidate(post=post, source="thunder"))

        for post_id, _sim in phoenix_results:
            post = self._post_map.get(post_id)
            if post:
                candidates.append(ScoredCandidate(post=post, source="phoenix_retrieval"))

        # Step 3 — Pre-scoring filters
        seen = self._seen.get(user.user_id, set())
        candidates = apply_pre_scoring_filters(
            candidates,
            user_id=user.user_id,
            blocked_ids=user.blocked,
            muted_keywords=user.muted_keywords,
            seen_post_ids=seen,
        )

        if not candidates:
            return []

        # Step 4 & 5 — Score with Phoenix transformer + weighted score
        history = self._engagement_history.get(user.user_id, [])
        history_tensor = _build_history_tensor(user, self._post_map, history)

        self._scorer.eval()
        with torch.no_grad():
            for c in candidates:
                cand_feat = torch.tensor(
                    scorer_post_features(c.post), dtype=torch.float32
                )
                logits = self._scorer(
                    history_tensor.unsqueeze(0),
                    cand_feat.unsqueeze(0),
                )
                probs = torch.sigmoid(logits).squeeze(0)
                c.action_predictions = {
                    action: probs[i].item() for i, action in enumerate(ActionType)
                }
                c.final_score = compute_weighted_score(c.action_predictions, ACTION_WEIGHTS)

        # Step 6 — Author diversity + Step 7 — Top K
        candidates = apply_author_diversity(candidates)
        top = select_top_k(candidates, k)

        # Track seen posts
        self._seen.setdefault(user.user_id, set()).update(c.post.post_id for c in top)

        return top
