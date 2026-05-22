from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.generator import DataGenerator
from data.models import ActionType, Post, Topic, User
from phoenix.retrieval.index import PostIndex
from phoenix.retrieval.towers import TwoTowerModel
from phoenix.scorer.transformer import PhoenixScorer
from pipeline.orchestrator import FeedOrchestrator
from thunder.store import ThunderStore

# ---------------------------------------------------------------------------
# State shared across requests
# ---------------------------------------------------------------------------

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load dataset
    data_path = Path("data/generated/dataset.json")
    gen = DataGenerator.load_from_json(data_path)
    user_map: dict[str, User] = {u.user_id: u for u in gen.users}
    post_map: dict[str, Post] = {p.post_id: p for p in gen.posts}

    # Redis / Thunder
    thunder = ThunderStore()
    for post in gen.posts:
        thunder.ingest_post(post)

    # Two-tower retrieval model
    retrieval_model = TwoTowerModel()
    retrieval_ckpt = Path("models/retrieval.pt")
    if retrieval_ckpt.exists():
        retrieval_model.load_state_dict(torch.load(retrieval_ckpt, weights_only=True))
    retrieval_model.eval()

    # FAISS index
    faiss_index = PostIndex()
    faiss_meta = Path("models/faiss.meta")
    if faiss_meta.exists():
        faiss_index = PostIndex.load(Path("models/faiss"))
    else:
        faiss_index.build_index(gen.posts, retrieval_model.candidate_tower)

    # Scorer model
    scorer = PhoenixScorer()
    scorer_ckpt = Path("models/scorer.pt")
    if scorer_ckpt.exists():
        scorer.load_state_dict(torch.load(scorer_ckpt, weights_only=True))
    scorer.eval()

    # Orchestrator
    orchestrator = FeedOrchestrator(
        thunder_store=thunder,
        retrieval_model=retrieval_model,
        faiss_index=faiss_index,
        scorer_model=scorer,
        post_map=post_map,
    )

    state["orchestrator"] = orchestrator
    state["user_map"] = user_map
    state["post_map"] = post_map
    state["thunder"] = thunder

    yield

    state.clear()


app = FastAPI(title="MiniForYou", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CreatePostRequest(BaseModel):
    author_id: str
    text: str
    primary_topic: str
    has_media: bool = False
    has_link: bool = False


class CreateUserRequest(BaseModel):
    username: str
    interests: dict[str, float]  # topic_name -> weight


class EngageRequest(BaseModel):
    user_id: str
    post_id: str
    action: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "users": len(state.get("user_map", {})), "posts": len(state.get("post_map", {}))}


@app.post("/users", status_code=201)
async def create_user(req: CreateUserRequest):
    try:
        interest_vector = {Topic(k): v for k, v in req.interests.items()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user = User(
        user_id=str(uuid.uuid4()),
        username=req.username,
        interest_vector=interest_vector,
    )
    state["user_map"][user.user_id] = user
    return {"user_id": user.user_id, "username": user.username}


@app.get("/users/{user_id}")
async def get_user(user_id: str):
    user = state["user_map"].get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "user_id": user.user_id,
        "username": user.username,
        "interest_vector": {t.value: v for t, v in user.interest_vector.items()},
        "following_count": len(user.following),
    }


@app.post("/posts", status_code=201)
async def create_post(req: CreatePostRequest):
    user_map = state["user_map"]
    if req.author_id not in user_map:
        raise HTTPException(status_code=404, detail="Author not found")

    try:
        primary_topic = Topic(req.primary_topic)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid topic: {req.primary_topic}")

    topic_vector = {t: 0.1 for t in Topic}
    topic_vector[primary_topic] = 0.9

    post = Post(
        post_id=str(uuid.uuid4()),
        author_id=req.author_id,
        text=req.text,
        topic_vector=topic_vector,
        primary_topic=primary_topic,
        has_media=req.has_media,
        has_link=req.has_link,
        created_at=datetime.utcnow(),
    )

    state["post_map"][post.post_id] = post
    state["thunder"].ingest_post(post)
    return {"post_id": post.post_id}


@app.get("/feed/{user_id}")
async def get_feed(user_id: str, k: int = 20):
    user = state["user_map"].get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    orchestrator: FeedOrchestrator = state["orchestrator"]
    feed = await orchestrator.generate_feed(user, k=k)

    return {
        "user_id": user_id,
        "feed": [
            {
                "post_id": c.post.post_id,
                "author_id": c.post.author_id,
                "text": c.post.text,
                "primary_topic": c.post.primary_topic.value,
                "has_media": c.post.has_media,
                "like_count": c.post.like_count,
                "reply_count": c.post.reply_count,
                "repost_count": c.post.repost_count,
                "source": c.source,
                "final_score": round(c.final_score, 4),
                "action_predictions": {
                    k: round(v, 4) for k, v in c.action_predictions.items()
                    if isinstance(k, ActionType)
                },
            }
            for c in feed
        ],
    }


@app.post("/engage")
async def engage(req: EngageRequest):
    user_map = state["user_map"]
    post_map = state["post_map"]

    if req.user_id not in user_map:
        raise HTTPException(status_code=404, detail="User not found")
    if req.post_id not in post_map:
        raise HTTPException(status_code=404, detail="Post not found")

    try:
        action = ActionType(req.action)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action: {req.action}")

    orchestrator: FeedOrchestrator = state["orchestrator"]
    orchestrator.record_engagement(req.user_id, req.post_id, action)

    return {"status": "recorded", "action": action.value}
