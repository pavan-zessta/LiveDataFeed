# MiniForYou — Project Context

## What this is
A learning project that mirrors X's "For You" feed recommendation algorithm.
Built with FastAPI + PyTorch + Redis + React.

## Architecture
- **Home Mixer** (api/ + pipeline/) → FastAPI orchestrator
- **Thunder** (thunder/) → Redis in-memory post store for in-network posts
- **Phoenix Retrieval** (phoenix/retrieval/) → Two-tower PyTorch model + FAISS
- **Phoenix Scorer** (phoenix/scorer/) → Transformer with candidate isolation
- **Weighted Scorer** (pipeline/scorers.py) → Final = Σ(weight × P(action))
- **Filters** (pipeline/filters.py) → Pre/post scoring filters

## Build order
1. data/models.py → Enums + dataclasses
2. data/generator.py → Fake data simulator
3. thunder/store.py → Redis store
4. phoenix/retrieval/towers.py, index.py, train.py → Two-tower + FAISS
5. phoenix/scorer/transformer.py, train.py → Transformer scorer
6. pipeline/filters.py, scorers.py → Filters + weighted scoring
7. pipeline/orchestrator.py, api/main.py → Wire everything
8. frontend/ → React UI

## Key design decisions
- Candidate isolation: candidates can't attend to each other in transformer
- Multi-action prediction: P(like), P(reply), ..., P(block), P(report)
- Negative weights are extreme: P(block)×-500 dwarfs P(like)×1.0
- Author diversity: decay_factor=0.7 per repeated author
- Dot product similarity for retrieval, sigmoid outputs for scoring

## Tech: Python 3.10+, FastAPI, PyTorch, FAISS, Redis