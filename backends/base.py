from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class TopKResult:
    token_ids: List[int]
    probs: List[float]  # re-normalized top-K probs; sums to 1.0 (paper Eq. 2)


class LMBackend(ABC):
    """Minimal interface the EAGER engine needs from a language model.

    next_token_topk takes the FULL token sequence generated so far, not just
    the newest token. This is deliberate: it's the only contract that both a
    local transformers model (which could internally cache) and a stateless
    HTTP API (OpenAI-compatible chat completions) can satisfy identically.
    The engine code that computes entropy and decides whether to branch never
    needs to know which kind of backend it's talking to.
    """

    @abstractmethod
    def encode(self, text: str) -> List[int]: ...

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str: ...

    @abstractmethod
    def next_token_topk(self, token_ids: List[int], k: int, temperature: float) -> TopKResult: ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int: ...
