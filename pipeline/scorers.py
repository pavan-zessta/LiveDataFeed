from __future__ import annotations

from data.models import ActionType, ScoredCandidate
from phoenix.scorer.weights import ACTION_WEIGHTS


def compute_weighted_score(
    predictions: dict[ActionType, float],
    weights: dict[ActionType, float] = ACTION_WEIGHTS,
) -> float:
    """Final Score = Σ (weight_i × P(action_i))"""
    return sum(weights.get(action, 0.0) * prob for action, prob in predictions.items())


def apply_author_diversity(
    candidates: list[ScoredCandidate],
    decay_factor: float = 0.7,
) -> list[ScoredCandidate]:
    """
    Penalise repeated authors so no single account floods the feed.
    1st post from author: full score
    2nd post:             score × 0.7
    3rd post:             score × 0.49
    ...
    """
    sorted_candidates = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    author_count: dict[str, int] = {}
    out: list[ScoredCandidate] = []

    for c in sorted_candidates:
        n = author_count.get(c.post.author_id, 0)
        if n > 0:
            c.final_score *= decay_factor ** n
        author_count[c.post.author_id] = n + 1
        out.append(c)

    # Re-sort after decay adjustments
    out.sort(key=lambda c: c.final_score, reverse=True)
    return out


def select_top_k(candidates: list[ScoredCandidate], k: int = 50) -> list[ScoredCandidate]:
    return sorted(candidates, key=lambda c: c.final_score, reverse=True)[:k]
