import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0

    return float(np.dot(a, b) / denom)

def normalize_similarity_to_score(similarity: float) -> float:
    similarity = max(-1.0, min(1.0, similarity))
    return ((similarity + 1.0) / 2.0) * 100.0