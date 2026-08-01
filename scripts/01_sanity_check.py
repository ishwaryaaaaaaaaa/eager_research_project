"""Phase 1 sanity check: no EAGER logic yet.

Just prove the pipe works end to end: load the model on the GPU, generate
plain greedy text, and pull a top-K entropy value at every single step.
If this script's entropy numbers look sane (low most of the time, occasional
spikes) then engine/entropy.py and backends/local_transformers.py are both
verified against a real model before we build anything on top of them.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.local_transformers import LocalTransformersBackend
from engine.entropy import topk_entropy

PROMPT = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Think step by step."
MAX_NEW_TOKENS = 60
TOP_K = 20

print("Loading Qwen2.5-1.5B-Instruct onto GPU (first run downloads ~3GB)...")
backend = LocalTransformersBackend()
print(f"Loaded on device: {backend.device}\n")

token_ids = backend.apply_chat_template(PROMPT)
prompt_len = len(token_ids)

print(f"Prompt tokens: {prompt_len}")
print("Generating (greedy) with per-token entropy:\n")
print(f"{'step':>4} {'token':<20} {'entropy':>8}")
print("-" * 36)

for step in range(MAX_NEW_TOKENS):
    result = backend.next_token_topk(token_ids, k=TOP_K, temperature=1.0)
    entropy = topk_entropy(result.probs)

    next_token_id = result.token_ids[0]  # greedy = highest-probability token
    token_ids.append(next_token_id)

    token_text = backend.decode([next_token_id]).replace("\n", "\\n")
    marker = " <-- spike" if entropy > 1.5 else ""
    print(f"{step:>4} {token_text:<20} {entropy:>8.3f}{marker}")

    if next_token_id == backend.eos_token_id:
        print("\n(hit EOS)")
        break

print("\nFull generated text:")
print(backend.decode(token_ids[prompt_len:]))
