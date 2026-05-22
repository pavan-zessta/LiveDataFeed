"""
MODULE 4: Phoenix Scorer — Transformer Model
==============================================
The core ranking model. Takes a user's engagement history + one candidate
post and predicts probabilities for every possible action.

TODO: Implement:

1. PhoenixScorer(nn.Module)
   - __init__(d_model=64, n_heads=4, n_layers=2, n_actions=12)

   - Embedding layers:
     → action_embedding: maps ActionType to d_model vector
     → topic_embedding: maps topic vector to d_model vector
     → position_embedding: learned positional encoding

   - Transformer encoder with CANDIDATE ISOLATION:
     → User's engagement history tokens CAN attend to each other
     → Candidate post token can attend to user history
     → Candidate post CANNOT attend to other candidates
     → This is done via a custom attention mask

   - Output heads:
     → One sigmoid output per action type
     → P(like), P(reply), P(repost), ..., P(block), P(report)

   - forward(user_history, candidate_features) -> dict[ActionType, float]

2. build_attention_mask(seq_len, n_candidates)
   → Returns mask where candidates can see user context but not each other
   → Shape: [seq_len, seq_len], True = can attend, False = blocked

CANDIDATE ISOLATION (key design choice):
  Without isolation: post A's score depends on what other posts are in the batch.
  With isolation: post A's score is always the same regardless of batch composition.
  This makes scores consistent and cacheable — critical at X's scale.

  Attention mask example (3 history tokens, 2 candidates):
                  h1  h2  h3  c1  c2
  history_1  h1 [ ✓   ✓   ✓   ✗   ✗ ]
  history_2  h2 [ ✓   ✓   ✓   ✗   ✗ ]
  history_3  h3 [ ✓   ✓   ✓   ✗   ✗ ]
  candidate1 c1 [ ✓   ✓   ✓   ✓   ✗ ]  ← sees history + itself only
  candidate2 c2 [ ✓   ✓   ✓   ✗   ✓ ]  ← sees history + itself only
"""
