import numpy as np

RUSH_HOURS = {8, 9, 10, 12, 13, 18, 19, 20}


def default_eta(km, hour: int):
    """Naive baseline ETA (minutes): constant speed, slower in rush hours."""
    speed = 14.0 if hour in RUSH_HOURS else 22.0
    return np.asarray(km, dtype=float) / speed * 60.0
