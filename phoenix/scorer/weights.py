from data.models import ActionType

ACTION_WEIGHTS: dict[ActionType, float] = {
    ActionType.LIKE:            1.0,
    ActionType.REPLY:          27.0,
    ActionType.REPOST:         10.0,
    ActionType.QUOTE:          20.0,
    ActionType.CLICK:           0.5,
    ActionType.SHARE:          15.0,
    ActionType.FOLLOW_AUTHOR: 100.0,
    ActionType.VIDEO_VIEW:      0.3,
    ActionType.NOT_INTERESTED: -100.0,
    ActionType.BLOCK_AUTHOR:   -500.0,
    ActionType.MUTE_AUTHOR:    -300.0,
    ActionType.REPORT:        -1000.0,
}
