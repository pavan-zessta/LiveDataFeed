"""
MODULE 3: Phoenix Retrieval — FAISS Index
==========================================
After training the two-tower model, we pre-compute embeddings for all posts
and store them in a FAISS index for fast similarity search.

TODO: Implement PostIndex class:

1. __init__(embedding_dim=64)
   → Create a FAISS IndexFlatIP (inner product / dot product index)

2. build_index(posts: list[Post], candidate_tower: CandidateTower)
   → Encode all posts through the CandidateTower
   → Add embeddings to the FAISS index
   → Store post_id mapping (FAISS returns integer indices, not post_ids)

3. search(user_embedding: np.ndarray, top_k=500) -> list[tuple[str, float]]
   → Query FAISS for top_k nearest posts by dot product
   → Return list of (post_id, similarity_score)

4. save(path) / load(path)
   → Persist the index to disk

WHY FAISS:
  Brute force dot product over 1M posts = slow.
  FAISS uses optimized BLAS routines and optional quantization
  to search millions of vectors in milliseconds.
  IndexFlatIP = exact search (good for learning).
  IndexIVFFlat = approximate search (good for production scale).
"""
