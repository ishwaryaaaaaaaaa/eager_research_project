"""Task 4 test: do all 4 modes actually behave differently in the way they should?"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.local_transformers import LocalTransformersBackend
from engine.modes import run_mode

PROMPT = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Think step by step."
M = 4
MAX_STEPS = 60
THETA = 1.2
INTERVAL = 15

print("Loading model...")
backend = LocalTransformersBackend()
token_ids = backend.apply_chat_template(PROMPT)

for mode, kwargs in [
    ("greedy", {}),
    ("full_parallel", {}),
    ("branch_fixed", {"interval": INTERVAL}),
    ("eager", {"theta": THETA}),
]:
    result = run_mode(mode, token_ids, backend, M=M, max_steps=MAX_STEPS, **kwargs)
    naive_upper_bound = len(result.sequences) * MAX_STEPS
    savings = 100 * (1 - result.total_tokens / naive_upper_bound) if naive_upper_bound else 0
    print(f"{mode:>15}: {len(result.sequences)} sequences, {result.total_tokens} tokens, {savings:.1f}% savings vs naive")
