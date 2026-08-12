"""Task 3 test: does branching actually happen, and does prefix-sharing save tokens?"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.local_transformers import LocalTransformersBackend
from engine.branch_policy import entropy_gated
from engine.branch_tree import BranchTree, reconstruct_sequence

PROMPT = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Think step by step."
THETA = 1.2
M = 4
MAX_STEPS = 80

print("Loading model...")
backend = LocalTransformersBackend()
token_ids = backend.apply_chat_template(PROMPT)

tree = BranchTree(backend=backend, policy_fn=entropy_gated(THETA), M=M, max_steps=MAX_STEPS)
result = tree.run(token_ids)

naive_upper_bound = M * MAX_STEPS
print(f"\nCompleted leaves: {len(result.completed_leaves)}")
print(f"Total unique generated tokens (tree nodes): {result.total_nodes}")
print(f"Naive upper bound with no sharing (M x max_steps): {naive_upper_bound}")
print(f"Savings from prefix sharing: {100 * (1 - result.total_nodes / naive_upper_bound):.1f}%\n")

for i, leaf in enumerate(result.completed_leaves):
    seq = reconstruct_sequence(leaf)
    text = backend.decode(seq)
    print(f"--- Leaf {i} ({len(seq)} tokens) ---")
    print(text[:300].replace("\n", " "))
    print()
