import random
from dataclasses import dataclass, field
from typing import List, Optional

from backends.base import LMBackend
from engine.branch_policy import BranchPolicy
from engine.entropy import topk_entropy


@dataclass(eq=False)  # identity-based equality: needed for active.remove(leaf) to work correctly
class Node:
    token_id: Optional[int]
    parent: Optional["Node"]
    entropy: Optional[float] = None
    steps_since_last_branch: int = 0
    children: List["Node"] = field(default_factory=list)


@dataclass
class TreeResult:
    completed_leaves: List[Node]
    total_nodes: int  # generated tokens only; root (the prompt) is never counted


def reconstruct_sequence(leaf: Node) -> List[int]:
    """Walk parent pointers back to the root, return generated token ids in
    root-to-leaf order. The root is a sentinel (token_id=None) and is never
    included -- this returns only what was generated, not the prompt."""
    tokens: List[int] = []
    node = leaf
    while node.parent is not None:
        assert node.token_id is not None  # only the root sentinel has token_id=None
        tokens.append(node.token_id)
        node = node.parent
    tokens.reverse()
    return tokens


class BranchTree:
    def __init__(
        self,
        backend: LMBackend,
        policy_fn: BranchPolicy,
        M: int,
        max_steps: int,
        k: int = 20,
        temperature: float = 1.0,
    ):
        self.backend = backend
        self.policy_fn = policy_fn
        self.M = M
        self.max_steps = max_steps
        self.k = k
        self.temperature = temperature

    def run(self, prompt_token_ids: List[int]) -> TreeResult:
        root = Node(token_id=None, parent=None)
        active: List[Node] = [root]
        completed: List[Node] = []
        total_nodes = 0

        for _ in range(self.max_steps):
            if not active:
                break
            leaves_this_step = list(active)  # fixed snapshot: each leaf advances exactly once this timestep

            for leaf in leaves_this_step:
                context = prompt_token_ids + reconstruct_sequence(leaf)
                result = self.backend.next_token_topk(context, k=self.k, temperature=self.temperature)
                entropy = topk_entropy(result.probs)

                # len(active) is live, already reflects any branching earlier this
                # same timestep -- matches Algorithm 1's mutate-while-scanning semantics.
                if self.policy_fn(entropy, len(active), self.M, leaf.steps_since_last_branch):
                    new_leaves = [
                        self._add_child(leaf, result.token_ids[0], entropy, 0),
                        self._add_child(leaf, result.token_ids[1], entropy, 0),
                    ]
                    total_nodes += 2
                else:
                    token_id = random.choices(result.token_ids, weights=result.probs, k=1)[0]
                    new_leaves = [self._add_child(leaf, token_id, entropy, leaf.steps_since_last_branch + 1)]
                    total_nodes += 1

                active.remove(leaf)
                for node in new_leaves:
                    if node.token_id == self.backend.eos_token_id:
                        completed.append(node)
                    else:
                        active.append(node)

        completed.extend(active)  # anything still active hit max_steps without EOS
        return TreeResult(completed_leaves=completed, total_nodes=total_nodes)

    @staticmethod
    def _add_child(parent: Node, token_id: int, entropy: float, steps_since_last_branch: int) -> Node:
        child = Node(token_id=token_id, parent=parent, entropy=entropy, steps_since_last_branch=steps_since_last_branch)
        parent.children.append(child)
        return child
