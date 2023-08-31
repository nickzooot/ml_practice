import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.maximum(
        np.linalg.norm(x, axis=-1)[:, None] ** 2 +
        np.linalg.norm(y, axis=-1)[None, :] ** 2 -
        (2 * x) @ y.T, 0))


def cosine_distance(x, y):
    return (1 - (x @ y.T) /
            np.linalg.norm(x, axis=-1)[:, None] /
            np.linalg.norm(y, axis=-1)[None, :])
