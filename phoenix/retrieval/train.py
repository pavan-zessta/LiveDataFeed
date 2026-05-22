"""
MODULE 3: Two-Tower Training Loop
===================================

TODO: Implement:

1. RetrievalDataset(Dataset)
   - Load engagements from JSON
   - Each sample: (user_features, post_features, label)
   - Positive pairs: actual engagements (label=1)
   - Negative pairs: random user-post combinations (label=0)
   - Ratio: ~4 negatives per positive

2. train_retrieval_model(data_path, epochs=20, lr=0.001)
   - Create DataLoader with RetrievalDataset
   - Initialize TwoTowerModel
   - Loss: BCEWithLogitsLoss
   - Optimizer: Adam
   - Train loop: forward → loss → backward → step
   - Save model checkpoint to models/retrieval.pt

3. evaluate(model, test_loader)
   - Compute recall@K: of the posts a user actually engaged with,
     how many appear in the model's top-K predictions?
"""
