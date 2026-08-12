from typing import Protocol


class BranchPolicy(Protocol):
    def __call__(self, entropy: float, num_active_leaves: int, M: int, steps_since_last_branch: int) -> bool: ...


def entropy_gated(theta: float) -> BranchPolicy:
    """The real EAGER rule (Algorithm 1): branch iff entropy is high enough
    AND we still have room under the sequence cap M."""

    def policy(entropy: float, num_active_leaves: int, M: int, steps_since_last_branch: int) -> bool:
        return entropy >= theta and num_active_leaves < M

    return policy


def never() -> BranchPolicy:
    """Used by greedy/full_parallel: a single BranchTree call should walk
    exactly one path and never fork. Paired with M=1 in the caller as a
    belt-and-suspenders guarantee, not because this alone is load-bearing."""

    def policy(entropy: float, num_active_leaves: int, M: int, steps_since_last_branch: int) -> bool:
        return False

    return policy


def fixed_interval(n: int) -> BranchPolicy:
    """Ablation baseline: ignore entropy, branch every n steps, still capped
    at M. Isolates whether entropy-gating specifically matters, or whether
    any periodic branching would do."""

    def policy(entropy: float, num_active_leaves: int, M: int, steps_since_last_branch: int) -> bool:
        return steps_since_last_branch >= n and num_active_leaves < M

    return policy
