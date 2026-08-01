import math
from typing import Sequence


def topk_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy, in nats, of an already-renormalized top-K distribution.

    H = -sum(p_i * log(p_i))   (paper Eq. 1, applied to the re-normalized
    p^(K) from Eq. 2, i.e. the `probs` you get out of TopKResult).

    Pure function: no model, no tokens, no I/O. Same formula whether the
    probabilities came from a local model or an API's logprobs field.
    """
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h
