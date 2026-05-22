"""
MODULE 6: FastAPI Application
==============================

TODO: Implement these endpoints:

POST /posts
  → Create a new post, ingest into Thunder
  → Body: {author_id, text, primary_topic, has_media}

GET /feed/{user_id}
  → Generate personalized feed for user
  → Calls FeedOrchestrator.generate_feed()
  → Returns ranked list of ScoredCandidate.to_dict()

POST /engage
  → Record an engagement (like, reply, etc.)
  → Body: {user_id, post_id, action}
  → Updates the user's engagement history

GET /users/{user_id}
  → Get user profile and interest vector

POST /users
  → Create a new user
  → Body: {username, interests}

GET /health
  → Health check

STARTUP:
  - Load trained models from models/
  - Connect to Redis
  - Build FAISS index
  - Initialize FeedOrchestrator
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MiniForYou", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "message": "Start implementing modules!"}
