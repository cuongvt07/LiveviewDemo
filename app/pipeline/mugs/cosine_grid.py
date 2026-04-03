import numpy as np
from functools import lru_cache


@lru_cache(maxsize=64)
def cosine_spacing(n: int) -> tuple[float, ...]:
    return tuple((1.0 - float(np.cos(np.pi * i / n))) / 2.0 for i in range(n + 1))
