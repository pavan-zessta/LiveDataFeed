from __future__ import annotations

from datetime import datetime, timedelta

from data.models import ScoredCandidate


def filter_duplicates(candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
    seen: set[str] = set()
    out: list[ScoredCandidate] = []
    for c in candidates:
        if c.post.post_id not in seen:
            seen.add(c.post.post_id)
            out.append(c)
    return out


def filter_old_posts(
    candidates: list[ScoredCandidate],
    max_age_hours: int = 48,
) -> list[ScoredCandidate]:
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    return [c for c in candidates if c.post.created_at >= cutoff]


def filter_self_posts(
    candidates: list[ScoredCandidate],
    user_id: str,
) -> list[ScoredCandidate]:
    return [c for c in candidates if c.post.author_id != user_id]


def filter_blocked_authors(
    candidates: list[ScoredCandidate],
    blocked_ids: list[str],
) -> list[ScoredCandidate]:
    blocked = set(blocked_ids)
    return [c for c in candidates if c.post.author_id not in blocked]


def filter_muted_keywords(
    candidates: list[ScoredCandidate],
    muted_keywords: list[str],
) -> list[ScoredCandidate]:
    if not muted_keywords:
        return candidates
    lowered = [kw.lower() for kw in muted_keywords]
    return [
        c for c in candidates
        if not any(kw in c.post.text.lower() for kw in lowered)
    ]


def filter_already_seen(
    candidates: list[ScoredCandidate],
    seen_post_ids: set[str],
) -> list[ScoredCandidate]:
    return [c for c in candidates if c.post.post_id not in seen_post_ids]


def filter_author_diversity(
    candidates: list[ScoredCandidate],
    max_per_author: int = 3,
) -> list[ScoredCandidate]:
    # Sort by score so we keep the best posts per author
    sorted_candidates = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    author_count: dict[str, int] = {}
    out: list[ScoredCandidate] = []
    for c in sorted_candidates:
        count = author_count.get(c.post.author_id, 0)
        if count < max_per_author:
            out.append(c)
            author_count[c.post.author_id] = count + 1
    return out


def apply_pre_scoring_filters(
    candidates: list[ScoredCandidate],
    user_id: str,
    blocked_ids: list[str],
    muted_keywords: list[str],
    seen_post_ids: set[str],
    max_age_hours: int = 48,
) -> list[ScoredCandidate]:
    candidates = filter_duplicates(candidates)
    candidates = filter_old_posts(candidates, max_age_hours)
    candidates = filter_self_posts(candidates, user_id)
    candidates = filter_blocked_authors(candidates, blocked_ids)
    candidates = filter_muted_keywords(candidates, muted_keywords)
    candidates = filter_already_seen(candidates, seen_post_ids)
    return candidates
