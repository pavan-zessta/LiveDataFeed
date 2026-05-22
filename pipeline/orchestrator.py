"""
MODULE 6: Pipeline Orchestrator (Home Mixer)
=============================================
Wires everything together into a single feed generation call.

TODO: Implement FeedOrchestrator class:

1. __init__(thunder_store, retrieval_model, faiss_index,
            scorer_model, weights_config)

2. async generate_feed(user: User, k=50) -> list[ScoredCandidate]:

   Step 1 — Source candidates (run in parallel):
     thunder_candidates = thunder_store.get_in_network_candidates(user)
     phoenix_candidates = faiss_index.search(user_embedding, top_k=500)

   Step 2 — Merge:
     all_candidates = thunder_candidates + phoenix_candidates

   Step 3 — Pre-scoring filters:
     filtered = apply_all_pre_filters(all_candidates, user)

   Step 4 — Score with Phoenix transformer:
     for candidate in filtered:
       candidate.action_predictions = scorer_model.predict(user, candidate)

   Step 5 — Compute weighted scores:
     for candidate in filtered:
       candidate.final_score = compute_weighted_score(predictions, weights)

   Step 6 — Author diversity:
     diversified = apply_author_diversity(filtered)

   Step 7 — Select top K:
     return select_top_k(diversified, k)

PARALLEL EXECUTION NOTE:
  Thunder and Phoenix Retrieval are independent — run them with
  asyncio.gather() so they execute concurrently. This cuts latency
  roughly in half compared to sequential execution.
"""
