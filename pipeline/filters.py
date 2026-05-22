"""
MODULE 5: Pre/Post Scoring Filters
====================================
Cheap checks that remove obviously ineligible posts before
the expensive ML scoring step.

TODO: Implement these filter functions:

PRE-SCORING FILTERS (run before Phoenix Scorer):

1. filter_duplicates(candidates) -> list[ScoredCandidate]
   → Remove posts with the same post_id

2. filter_old_posts(candidates, max_age_hours=48) -> list
   → Remove posts older than threshold

3. filter_self_posts(candidates, user_id) -> list
   → Remove posts authored by the requesting user

4. filter_blocked_authors(candidates, blocked_ids) -> list
   → Remove posts from blocked accounts

5. filter_muted_keywords(candidates, muted_keywords) -> list
   → Remove posts containing any muted keyword (case-insensitive)

6. filter_already_seen(candidates, seen_post_ids: set) -> list
   → Remove posts the user has already been shown

POST-SCORING FILTERS (run after scoring + selection):

7. filter_author_diversity(candidates, max_per_author=3) -> list
   → If more than max_per_author posts from same author made it
     into the final set, keep only the top-scored ones

DESIGN NOTE:
  Filters are intentionally simple — no ML, just business logic.
  They run in microseconds. The goal is to shrink the candidate set
  before the expensive transformer inference.
"""
