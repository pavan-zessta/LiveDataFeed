from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from data.models import ActionType, Engagement, Post, ScoredCandidate, Topic, User

# ---------------------------------------------------------------------------
# Archetype interest profiles — base weights per topic (0-1)
# ---------------------------------------------------------------------------

USER_ARCHETYPES: dict[str, dict[Topic, float]] = {
    "techie": {
        Topic.TECH: 0.95, Topic.SCIENCE: 0.60, Topic.FINANCE: 0.45,
        Topic.GAMING: 0.40, Topic.ART: 0.10, Topic.SPORTS: 0.05,
        Topic.POLITICS: 0.10, Topic.MUSIC: 0.15, Topic.FOOD: 0.10, Topic.TRAVEL: 0.20,
    },
    "sports_fan": {
        Topic.SPORTS: 0.95, Topic.FINANCE: 0.20, Topic.FOOD: 0.30,
        Topic.MUSIC: 0.25, Topic.TECH: 0.15, Topic.POLITICS: 0.20,
        Topic.SCIENCE: 0.10, Topic.GAMING: 0.30, Topic.ART: 0.10, Topic.TRAVEL: 0.25,
    },
    "science_nerd": {
        Topic.SCIENCE: 0.95, Topic.TECH: 0.65, Topic.FINANCE: 0.25,
        Topic.POLITICS: 0.30, Topic.ART: 0.20, Topic.SPORTS: 0.05,
        Topic.MUSIC: 0.10, Topic.FOOD: 0.15, Topic.GAMING: 0.10, Topic.TRAVEL: 0.30,
    },
    "creative": {
        Topic.ART: 0.95, Topic.MUSIC: 0.85, Topic.TRAVEL: 0.60,
        Topic.FOOD: 0.50, Topic.TECH: 0.20, Topic.SCIENCE: 0.15,
        Topic.SPORTS: 0.10, Topic.POLITICS: 0.15, Topic.GAMING: 0.25, Topic.FINANCE: 0.10,
    },
    "news_junkie": {
        Topic.POLITICS: 0.90, Topic.FINANCE: 0.75, Topic.SCIENCE: 0.40,
        Topic.TECH: 0.35, Topic.SPORTS: 0.30, Topic.ART: 0.15,
        Topic.MUSIC: 0.10, Topic.FOOD: 0.20, Topic.GAMING: 0.05, Topic.TRAVEL: 0.25,
    },
    "foodie": {
        Topic.FOOD: 0.95, Topic.TRAVEL: 0.70, Topic.ART: 0.40,
        Topic.MUSIC: 0.35, Topic.SCIENCE: 0.20, Topic.SPORTS: 0.15,
        Topic.TECH: 0.10, Topic.POLITICS: 0.10, Topic.GAMING: 0.05, Topic.FINANCE: 0.15,
    },
}

# Engagement probabilities by similarity bucket
_POSITIVE_ACTIONS = [ActionType.LIKE, ActionType.REPLY, ActionType.REPOST,
                     ActionType.QUOTE, ActionType.CLICK, ActionType.SHARE,
                     ActionType.FOLLOW_AUTHOR, ActionType.VIDEO_VIEW]
_NEGATIVE_ACTIONS = [ActionType.NOT_INTERESTED, ActionType.BLOCK_AUTHOR,
                     ActionType.MUTE_AUTHOR, ActionType.REPORT]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _dot(a: dict[Topic, float], b: dict[Topic, float]) -> float:
    return sum(a.get(t, 0.0) * b.get(t, 0.0) for t in Topic)


def _topic_values_to_str(d: dict[Topic, float]) -> dict[str, float]:
    return {k.value: v for k, v in d.items()}


def _topic_str_to_enum(d: dict[str, float]) -> dict[Topic, float]:
    return {Topic(k): v for k, v in d.items()}


class DataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.users: list[User] = []
        self.posts: list[Post] = []
        self.engagements: list[Engagement] = []

    # ------------------------------------------------------------------
    # 1. Users
    # ------------------------------------------------------------------

    def generate_users(self, n: int = 100) -> list[User]:
        archetypes = list(USER_ARCHETYPES.keys())
        users: list[User] = []

        for i in range(n):
            archetype = random.choice(archetypes)
            base = USER_ARCHETYPES[archetype]

            # Add Gaussian noise ±0.15, then clamp to [0, 1]
            interest_vector = {
                topic: _clamp(base[topic] + random.gauss(0, 0.15))
                for topic in Topic
            }

            users.append(User(
                user_id=str(uuid.uuid4()),
                username=f"{archetype}_{i:04d}",
                interest_vector=interest_vector,
            ))

        # Assign following lists: each user follows 10–30% of others
        user_ids = [u.user_id for u in users]
        for user in users:
            others = [uid for uid in user_ids if uid != user.user_id]
            k = random.randint(int(0.10 * len(others)), int(0.30 * len(others)))
            user.following = random.sample(others, k)

        self.users = users
        return users

    # ------------------------------------------------------------------
    # 2. Posts
    # ------------------------------------------------------------------

    def generate_posts(self, posts_per_user: int = 20) -> list[Post]:
        if not self.users:
            raise RuntimeError("Call generate_users() first.")

        posts: list[Post] = []
        now = datetime.utcnow()

        for user in self.users:
            # Weight topics by user interest to pick primary topic
            topics = list(Topic)
            weights = [user.interest_vector.get(t, 0.0) + 0.05 for t in topics]  # +0.05 floor

            for _ in range(posts_per_user):
                primary = random.choices(topics, weights=weights, k=1)[0]

                # topic_vector: primary topic high (0.6–1.0), others low (0–0.3)
                topic_vector: dict[Topic, float] = {}
                for t in Topic:
                    if t == primary:
                        topic_vector[t] = random.uniform(0.6, 1.0)
                    else:
                        topic_vector[t] = random.uniform(0.0, 0.3)

                age_hours = random.uniform(0, 72)
                posts.append(Post(
                    post_id=str(uuid.uuid4()),
                    author_id=user.user_id,
                    text=f"Post about {primary.value} by {user.username}",
                    topic_vector=topic_vector,
                    primary_topic=primary,
                    has_media=random.random() < 0.3,
                    has_link=random.random() < 0.25,
                    like_count=random.randint(0, 500),
                    reply_count=random.randint(0, 100),
                    repost_count=random.randint(0, 200),
                    created_at=now - timedelta(hours=age_hours),
                ))

        self.posts = posts
        return posts

    # ------------------------------------------------------------------
    # 3. Engagements
    # ------------------------------------------------------------------

    def generate_engagements(self, posts_per_user_sample: int = 50) -> list[Engagement]:
        if not self.users or not self.posts:
            raise RuntimeError("Call generate_users() and generate_posts() first.")

        engagements: list[Engagement] = []
        now = datetime.utcnow()

        for user in self.users:
            sample = random.sample(self.posts, min(posts_per_user_sample, len(self.posts)))

            for post in sample:
                if post.author_id == user.user_id:
                    continue  # skip own posts

                similarity = _dot(user.interest_vector, post.topic_vector)
                # similarity is in ~[0, 3] for unit-ish vectors; normalise to [0,1]
                sim_norm = _clamp(similarity / 3.0)

                # Positive engagement: probability scales with similarity
                if random.random() < sim_norm * 0.6:
                    action = random.choice(_POSITIVE_ACTIONS)
                    engagements.append(Engagement(
                        user_id=user.user_id,
                        post_id=post.post_id,
                        action=action,
                        timestamp=now - timedelta(minutes=random.randint(0, 1440)),
                    ))

                # Negative engagement: probability scales with (1 - similarity)
                elif random.random() < (1 - sim_norm) * 0.15:
                    action = random.choice(_NEGATIVE_ACTIONS)
                    engagements.append(Engagement(
                        user_id=user.user_id,
                        post_id=post.post_id,
                        action=action,
                        timestamp=now - timedelta(minutes=random.randint(0, 1440)),
                    ))

        self.engagements = engagements
        return engagements

    # ------------------------------------------------------------------
    # 4. Serialisation
    # ------------------------------------------------------------------

    def save_to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _user(u: User) -> dict:
            return {
                "user_id": u.user_id,
                "username": u.username,
                "interest_vector": _topic_values_to_str(u.interest_vector),
                "following": u.following,
                "blocked": u.blocked,
                "muted_keywords": u.muted_keywords,
            }

        def _post(p: Post) -> dict:
            return {
                "post_id": p.post_id,
                "author_id": p.author_id,
                "text": p.text,
                "topic_vector": _topic_values_to_str(p.topic_vector),
                "primary_topic": p.primary_topic.value,
                "has_media": p.has_media,
                "has_link": p.has_link,
                "like_count": p.like_count,
                "reply_count": p.reply_count,
                "repost_count": p.repost_count,
                "created_at": p.created_at.isoformat(),
            }

        def _eng(e: Engagement) -> dict:
            return {
                "user_id": e.user_id,
                "post_id": e.post_id,
                "action": e.action.value,
                "timestamp": e.timestamp.isoformat(),
            }

        payload = {
            "users": [_user(u) for u in self.users],
            "posts": [_post(p) for p in self.posts],
            "engagements": [_eng(e) for e in self.engagements],
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_from_json(cls, path: str | Path) -> "DataGenerator":
        with open(path) as f:
            data = json.load(f)

        gen = cls.__new__(cls)
        gen.users = [
            User(
                user_id=u["user_id"],
                username=u["username"],
                interest_vector=_topic_str_to_enum(u["interest_vector"]),
                following=u["following"],
                blocked=u["blocked"],
                muted_keywords=u["muted_keywords"],
            )
            for u in data["users"]
        ]
        gen.posts = [
            Post(
                post_id=p["post_id"],
                author_id=p["author_id"],
                text=p["text"],
                topic_vector=_topic_str_to_enum(p["topic_vector"]),
                primary_topic=Topic(p["primary_topic"]),
                has_media=p["has_media"],
                has_link=p["has_link"],
                like_count=p["like_count"],
                reply_count=p["reply_count"],
                repost_count=p["repost_count"],
                created_at=datetime.fromisoformat(p["created_at"]),
            )
            for p in data["posts"]
        ]
        gen.engagements = [
            Engagement(
                user_id=e["user_id"],
                post_id=e["post_id"],
                action=ActionType(e["action"]),
                timestamp=datetime.fromisoformat(e["timestamp"]),
            )
            for e in data["engagements"]
        ]
        return gen
