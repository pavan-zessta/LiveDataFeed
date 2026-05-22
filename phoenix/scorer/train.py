"""
MODULE 4: Transformer Scorer Training
=======================================

TODO: Implement:

1. ScorerDataset(Dataset)
   - Each sample: (user_engagement_sequence, candidate_post, action_labels)
   - user_engagement_sequence: last N engagements as feature vectors
   - candidate_post: feature vector for the post being scored
   - action_labels: multi-hot vector [1,0,0,1,0,...] for which actions occurred

2. train_scorer(data_path, epochs=30, lr=0.0001)
   - Use BCEWithLogitsLoss (multi-label, not multi-class!)
   - Each action head is an independent binary prediction
   - Optimizer: AdamW with weight decay
   - Save to models/scorer.pt

3. Multi-label loss explained:
   Unlike classification (pick ONE class), we predict INDEPENDENT probabilities.
   A user can both like AND reply to the same post.
   So we use binary cross entropy on EACH action separately:
     loss = Σ BCE(predicted_P(action_i), actual_label_i)
"""
