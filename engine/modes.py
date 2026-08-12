from dataclasses import dataclass
from typing import List, Optional

from backends.base import LMBackend
from engine.branch_policy import BranchPolicy, entropy_gated, fixed_interval, never
from engine.branch_tree import BranchTree, reconstruct_sequence

GREEDY_TEMPERATURE = 0.01  # near-zero -> next_token_topk's softmax collapses close to argmax


@dataclass
class ModeResult:
    sequences: List[List[int]]
    total_tokens: int


def _run_single_tree(
    policy: BranchPolicy,
    prompt_token_ids: List[int],
    backend: LMBackend,
    M: int,
    max_steps: int,
    k: int,
    temperature: float,
) -> ModeResult:
    tree = BranchTree(backend, policy, M=M, max_steps=max_steps, k=k, temperature=temperature)
    result = tree.run(prompt_token_ids)
    return ModeResult(
        sequences=[reconstruct_sequence(leaf) for leaf in result.completed_leaves],
        total_tokens=result.total_nodes,
    )


def run_mode(
    mode: str,
    prompt_token_ids: List[int],
    backend: LMBackend,
    M: int,
    max_steps: int,
    theta: Optional[float] = None,
    interval: Optional[int] = None,
    k: int = 20,
    temperature: float = 1.0,
) -> ModeResult:
    if mode == "greedy":
        # M=1: a single BranchTree call, one path, no forking possible.
        return _run_single_tree(never(), prompt_token_ids, backend, M=1, max_steps=max_steps, k=k, temperature=GREEDY_TEMPERATURE)

    if mode == "full_parallel":
        # M separate BranchTree calls, each its own root -- zero sharing between them,
        # unlike branch_fixed/eager which share one root across all their leaves.
        sequences: List[List[int]] = []
        total_tokens = 0
        for _ in range(M):
            single = _run_single_tree(never(), prompt_token_ids, backend, M=1, max_steps=max_steps, k=k, temperature=temperature)
            sequences.extend(single.sequences)
            total_tokens += single.total_tokens
        return ModeResult(sequences=sequences, total_tokens=total_tokens)

    if mode == "branch_fixed":
        assert interval is not None, "branch_fixed requires `interval`"
        return _run_single_tree(fixed_interval(interval), prompt_token_ids, backend, M=M, max_steps=max_steps, k=k, temperature=temperature)

    if mode == "eager":
        assert theta is not None, "eager requires `theta`"
        return _run_single_tree(entropy_gated(theta), prompt_token_ids, backend, M=M, max_steps=max_steps, k=k, temperature=temperature)

    raise ValueError(f"unknown mode: {mode!r}")
