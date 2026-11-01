import numpy as np


def gini(values) -> float:
    """0 = perfectly equal earnings, 1 = one rider earns everything."""
    v = np.sort(np.asarray(values, dtype=float))
    n, total = len(v), v.sum()
    if n == 0 or total <= 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, n + 1)) * v)) / (n * total) - (n + 1) / n)
