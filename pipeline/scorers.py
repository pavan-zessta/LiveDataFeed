"""
MODULE 5: Weighted Scorer & Author Diversity
=============================================

TODO: Implement:

1. compute_weighted_score(predictions: dict[ActionType, float],
                          weights: dict[ActionType, float]) -> float
   → Final Score = Σ (weight_i × P(action_i))
   → This is the core formula from X's system

2. apply_author_diversity(candidates: list[ScoredCandidate],
                          decay_factor=0.7) -> list[ScoredCandidate]
   → Sort by score descending
   → For each author, track how many posts we've seen
   → Each additional post from same author: score *= decay_factor
   → 1st post: full score, 2nd: ×0.7, 3rd: ×0.49, 4th: ×0.343...
   → Re-sort after applying decay

3. select_top_k(candidates: list[ScoredCandidate], k=50) -> list
   → Sort by final_score descending, return top k
"""
