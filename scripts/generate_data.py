"""
Run this first: generates fake users, posts, and engagements.
Saves to data/generated/dataset.json

Usage: python scripts/generate_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generator import DataGenerator

if __name__ == "__main__":
    gen = DataGenerator(seed=42)

    print("Generating users...")
    users = gen.generate_users(n=100)
    print(f"  {len(users)} users created")

    print("Generating posts...")
    posts = gen.generate_posts(posts_per_user=20)
    print(f"  {len(posts)} posts created")

    print("Generating engagements...")
    engagements = gen.generate_engagements(posts_per_user_sample=50)
    print(f"  {len(engagements)} engagements created")

    out = Path("data/generated/dataset.json")
    gen.save_to_json(out)
    print(f"\nSaved to {out}")
