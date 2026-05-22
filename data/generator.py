"""
MODULE 1: Fake Data Generator
==============================
Generates realistic users, posts, and engagements for training.

TODO: Implement:

1. USER_ARCHETYPES dict
   - 6 profiles (techie, sports_fan, science_nerd, creative, news_junkie, foodie)
   - Each maps Topic -> float (0-1) representing interest strength

2. DataGenerator class
   - generate_users(n=100)
     → Create users by sampling archetype + adding noise (±0.15)
     → Assign random following lists (10-30% of other users)

   - generate_posts(posts_per_user=20)
     → Each user creates posts weighted toward their interests
     → Build topic_vector for each post (primary topic high, others low)

   - generate_engagements()
     → For each user, show them a sample of posts
     → Engagement probability = dot_product(user.interest_vector, post.topic_vector)
     → Higher similarity → more likely to like/reply/repost
     → Low similarity → more likely to get not_interested/block

   - save_to_json(path) / load_from_json(path)

KEY INSIGHT:
  The dot product between user interests and post topics determines
  engagement probability. This is EXACTLY what the two-tower model
  will learn to predict from raw features.
"""
