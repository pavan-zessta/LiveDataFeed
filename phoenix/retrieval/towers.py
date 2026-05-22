"""
MODULE 3: Phoenix Retrieval — Two-Tower Model
===============================================
PyTorch model that encodes users and posts into the same embedding space.

TODO: Implement:

1. UserTower(nn.Module)
   - Input: user interest vector (10 floats) + engagement history features
   - Architecture: Linear(input_dim, 128) → ReLU → Linear(128, embedding_dim)
   - Output: user embedding [embedding_dim]

2. CandidateTower(nn.Module)
   - Input: post topic vector (10 floats) + post features (has_media, has_link, age...)
   - Architecture: Linear(input_dim, 128) → ReLU → Linear(128, embedding_dim)
   - Output: post embedding [embedding_dim]

3. TwoTowerModel(nn.Module)
   - Combines UserTower + CandidateTower
   - forward(user_features, post_features) → similarity score
   - Similarity = dot product of the two embeddings
   - Training: positive pairs (user engaged with post) score high,
     negative pairs (random user-post) score low

MATH RECAP:
  user_emb  = UserTower(user_features)       # shape: [batch, 64]
  post_emb  = CandidateTower(post_features)  # shape: [batch, 64]
  score     = (user_emb * post_emb).sum(-1)  # dot product per pair
  loss      = BCEWithLogitsLoss(score, label) # 1 if engaged, 0 if not
"""
