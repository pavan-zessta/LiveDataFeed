from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Optional

import redis

from data.models import Post, Topic, User


class ThunderStore:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        ttl_hours: int = 48,
    ):
        self._r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self._ttl = ttl_hours * 3600

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_key(self, author_id: str, post_id: str) -> str:
        return f"posts:{author_id}:{post_id}"

    def _author_set_key(self, author_id: str) -> str:
        return f"author_posts:{author_id}"

    def _serialize(self, post: Post) -> str:
        return json.dumps({
            "post_id": post.post_id,
            "author_id": post.author_id,
            "text": post.text,
            "topic_vector": {t.value: v for t, v in post.topic_vector.items()},
            "primary_topic": post.primary_topic.value,
            "has_media": post.has_media,
            "has_link": post.has_link,
            "like_count": post.like_count,
            "reply_count": post.reply_count,
            "repost_count": post.repost_count,
            "created_at": post.created_at.isoformat(),
        })

    def _deserialize(self, raw: str) -> Post:
        d = json.loads(raw)
        return Post(
            post_id=d["post_id"],
            author_id=d["author_id"],
            text=d["text"],
            topic_vector={Topic(k): v for k, v in d["topic_vector"].items()},
            primary_topic=Topic(d["primary_topic"]),
            has_media=d["has_media"],
            has_link=d["has_link"],
            like_count=d["like_count"],
            reply_count=d["reply_count"],
            repost_count=d["repost_count"],
            created_at=datetime.fromisoformat(d["created_at"]),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_post(self, post: Post) -> None:
        key = self._post_key(post.author_id, post.post_id)
        score = post.created_at.timestamp()

        pipe = self._r.pipeline()
        pipe.set(key, self._serialize(post), ex=self._ttl)
        pipe.zadd(self._author_set_key(post.author_id), {post.post_id: score})
        pipe.expire(self._author_set_key(post.author_id), self._ttl)
        pipe.execute()

    def get_in_network_candidates(self, user: User) -> list[Post]:
        cutoff = time.time() - self._ttl
        posts: list[Post] = []

        for author_id in user.following:
            # Fetch post IDs in the TTL window, newest first
            post_ids = self._r.zrangebyscore(
                self._author_set_key(author_id),
                min=cutoff,
                max="+inf",
            )
            for post_id in post_ids:
                raw = self._r.get(self._post_key(author_id, post_id))
                if raw:
                    posts.append(self._deserialize(raw))

        return posts

    def get_post(self, post_id: str, author_id: str) -> Optional[Post]:
        raw = self._r.get(self._post_key(author_id, post_id))
        return self._deserialize(raw) if raw else None

    def delete_post(self, post_id: str, author_id: str) -> None:
        pipe = self._r.pipeline()
        pipe.delete(self._post_key(author_id, post_id))
        pipe.zrem(self._author_set_key(author_id), post_id)
        pipe.execute()

    def flush(self) -> None:
        self._r.flushdb()
