import numpy as np


def ensure_signed_polarity(p):
    p = np.asarray(p)
    unique_values = np.unique(p)
    if np.all(np.isin(unique_values, [0, 1])):
        return np.where(p == 1, 1, -1).astype(np.int8)
    return np.where(p > 0, 1, -1).astype(np.int8)


def select_events_by_polarity(events, polarity_mode="positive"):
    xs = events["x"].astype(np.int32)
    ys = events["y"].astype(np.int32)
    ps = ensure_signed_polarity(events["p"])
    ts = events["t"].astype(np.int64)

    if polarity_mode == "positive":
        mask = ps > 0
    elif polarity_mode == "negative":
        mask = ps < 0
    elif polarity_mode == "all":
        mask = np.ones(len(ps), dtype=bool)
    else:
        raise ValueError("polarity_mode must be one of: 'positive', 'negative', 'all'")

    return xs[mask], ys[mask], ps[mask], ts[mask]
