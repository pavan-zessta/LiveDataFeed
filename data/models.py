from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Topic(Enum):
    TECH = "tech"
    SCIENCE = "science"
    SPORTS = "sports"
    POLITICS = "politics"
    MUSIC = "music"
    FOOD = "food"
    GAMING = "gaming"
    ART = "art"
    FINANCE = "finance"
    TRAVEL = "travel"


class ActionType(Enum):
    # Positive signals
    LIKE = "like"
    REPLY = "reply"
    REPOST = "repost"
    QUOTE = "quote"
    CLICK = "click"
    SHARE = "share"
    FOLLOW_AUTHOR = "follow_author"
    VIDEO_VIEW = "video_view"

    # Negative signals
    NOT_INTERESTED = "not_interested"
    BLOCK_AUTHOR = "block_author"
    MUTE_AUTHOR = "mute_author"
    REPORT = "report"


@dataclass
class User:
    user_id: str
    username: str
    interest_vector: dict[Topic, float] = field(default_factory=dict)  # 0–1 per topic
    following: list[str] = field(default_factory=list)   # user_ids
    blocked: list[str] = field(default_factory=list)     # user_ids
    muted_keywords: list[str] = field(default_factory=list)


@dataclass
class Post:
    post_id: str
    author_id: str
    text: str
    topic_vector: dict[Topic, float] = field(default_factory=dict)  # 0–1 per topic
    primary_topic: Topic = Topic.TECH
    has_media: bool = False
    has_link: bool = False
    like_count: int = 0
    reply_count: int = 0
    repost_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Engagement:
    user_id: str
    post_id: str
    action: ActionType
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScoredCandidate:
    post: Post
    source: str                                    # "thunder" or "phoenix_retrieval"
    action_predictions: dict[ActionType, float] = field(default_factory=dict)  # model outputs
    final_score: float = 0.0                       # weighted combination
