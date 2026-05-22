"""
MODULE 2: Thunder — In-Memory Post Store
=========================================
Redis-backed store for real-time post ingestion and retrieval.
This is X's "Thunder" — sub-millisecond lookups for in-network content.

TODO: Implement ThunderStore class:

1. __init__(redis_host, redis_port, ttl_hours=48)
   → Connect to Redis

2. ingest_post(post: Post)
   → Store post in Redis with TTL (auto-expires old posts)
   → Key pattern: "posts:{author_id}:{post_id}"
   → Also maintain a sorted set per author: "author_posts:{author_id}"
     scored by timestamp for fast range queries

3. get_in_network_candidates(user: User) -> list[Post]
   → For each user_id in user.following:
     → Fetch their recent posts from Redis
   → Return combined list (these are the "Thunder" candidates)

4. delete_post(post_id, author_id)
   → Remove from both the post key and the author sorted set

5. get_post(post_id) -> Optional[Post]
   → Fetch a single post by ID

DESIGN NOTES:
  - Redis sorted sets let you query "posts by author X in last 48 hours"
    in O(log N) time — this is why Thunder is fast
  - TTL handles cleanup automatically — no cron job needed
  - In X's real system, Thunder consumes Kafka events; we'll use
    direct method calls (same concept, simpler setup)
"""
