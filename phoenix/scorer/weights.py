"""
MODULE 5: Action Weights Configuration
========================================
These weights determine how much each predicted action contributes
to a post's final score.

TODO: Define ACTION_WEIGHTS dict mapping ActionType -> float

Suggested starting weights (tune these!):
  LIKE:             1.0
  REPLY:           27.0      ← replies = high engagement signal
  REPOST:          10.0
  QUOTE:           20.0
  CLICK:            0.5
  SHARE:           15.0
  FOLLOW_AUTHOR:  100.0      ← strongest positive signal
  VIDEO_VIEW:       0.3
  NOT_INTERESTED: -100.0     ← strong negative
  BLOCK_AUTHOR:  -500.0      ← very strong negative
  MUTE_AUTHOR:   -300.0
  REPORT:       -1000.0      ← nuclear negative

WHY THESE VALUES:
  The system is risk-averse. Even a 5% chance of block (-500 × 0.05 = -25)
  wipes out a 50% chance of like (1.0 × 0.5 = 0.5). This is intentional:
  showing content someone would block damages trust in the whole feed.
"""
