from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LMBackend, TopKResult


class LocalTransformersBackend(LMBackend):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def apply_chat_template(self, user_message: str) -> List[int]:
        messages = [{"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @torch.no_grad()
    def next_token_topk(self, token_ids: List[int], k: int = 20, temperature: float = 1.0) -> TopKResult:
        input_ids = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_ids).logits[0, -1, :]  # last position, full vocab
        logits = logits / max(temperature, 1e-5)
        probs_full = torch.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs_full, k)
        topk_probs = topk_probs / topk_probs.sum()  # re-normalize -> Eq. 2
        return TopKResult(token_ids=topk_ids.tolist(), probs=topk_probs.tolist())
