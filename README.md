# MiniForYou — Build X's "For You" Feed Algorithm

A learning project that mirrors X's recommendation pipeline:
FastAPI + PyTorch + Redis + React

## Architecture

```
[React Frontend]
        │
        ▼
[FastAPI Orchestrator]         ← api/main.py
        │
   ┌────┴────┐
   ▼         ▼
[Thunder]  [Phoenix Retrieval] ← thunder/store.py, phoenix/retrieval/
   └────┬────┘
        ▼
  [Filters]                    ← pipeline/filters.py
        ▼
  [Phoenix Scorer]             ← phoenix/scorer/transformer.py
        ▼
  [Weighted Scorer]            ← pipeline/scorers.py
        ▼
  [Top-K → Feed]
```

## Build Order

| Module | Files to implement | What you'll learn |
|--------|--------------------|-------------------|
| 1. Data Models | `data/models.py`, `data/generator.py` | Recommendation data structures |
| 2. Thunder | `thunder/store.py` | Redis, real-time ingestion |
| 3. Phoenix Retrieval | `phoenix/retrieval/towers.py`, `index.py`, `train.py` | Embeddings, FAISS, PyTorch |
| 4. Phoenix Scorer | `phoenix/scorer/transformer.py`, `train.py` | Transformers, attention |
| 5. Filters & Scorers | `pipeline/filters.py`, `pipeline/scorers.py` | Scoring design |
| 6. Orchestrator | `pipeline/orchestrator.py`, `api/main.py` | Async Python, API design |
| 7. Frontend | `frontend/` | Connecting ML to UI |

## Setup

```bash
pip install -r requirements.txt
# Start Redis: docker run -d -p 6379:6379 redis:latest
# Run API: uvicorn api.main:app --reload --port 8000
```
