import numpy as np


def cosine(v1: np.array, v2: np.array) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v1))


def l2(v1: np.array, v2: np.array) -> float:
    return np.linalg.norm(v1 - v2)


def l2_norm(v1: np.array) -> np.array:
    return v1 / np.linalg.norm(v1)


def match(
    enc1: np.array, enc2: np.array, thresholds: dict[str, float], metrics: str
) -> str:

    if metrics == "cosine":
        return cosine(enc1, enc2) <= thresholds["cosine"]

    elif metrics == "l2":
        return l2(enc1, enc2) <= thresholds["l2"]

    elif metrics == "l2_norm":

        return l2(l2_norm(enc1), l2_norm(enc2)) <= thresholds["l2_norm"]

    else:
        raise ValueError("Distance Metrics not found")
