# EAGER — Entropy-Gated Branching (local replication)

A from-scratch, learning-focused implementation of the core mechanism from
**["EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling"](https://cohere.com/research/papers/eager-entropy-aware-generation-for-adaptive-inference-time-scaling-2025-10-16)**
(Scalena, Zotos, Fersini, Nissim, Üstün — Cohere Labs / U. Groningen / U. Milan-Bicocca, 2025).

This is **not** a full paper replication — no budget-reallocation stage
(the paper's Algorithm 2 / EAGER-adapt), no benchmark-scale evaluation. It's
a simplified, from-first-principles build of the paper's central idea, run
entirely locally, with every design decision made deliberately rather than
copied from the paper without understanding it.

## The idea, in one paragraph

Standard "parallel sampling" generates `M` fully independent guesses per
prompt and picks the best — wasteful on easy prompts, where most guesses
converge to nearly the same answer. EAGER instead generates *one* sequence
and only **forks it into multiple candidates at tokens where the model is
genuinely uncertain** (measured via top-K token entropy), sharing every
token before the fork instead of regenerating it per branch.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12.10 | — |
| ML runtime | [PyTorch](https://pytorch.org/) 2.13.0, **CUDA 12.6 build** | Stock `pip install torch` on this machine resolved to a CPU-only wheel — had to explicitly install from `https://download.pytorch.org/whl/cu126` to get GPU support. |
| Model runtime | [Transformers](https://huggingface.co/docs/transformers) 5.14.1 + Accelerate 1.14.0 | Loads and runs the model; gives raw per-step logits, which the entropy computation needs. |
| Model | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Same family as the paper's Qwen3-4B, small enough to run comfortably on 6GB VRAM. |
| Hardware | NVIDIA RTX 3050 Laptop GPU, 6GB VRAM | Local, no API costs — every generation happens on-device. |
| Editor tooling | VS Code + `debugpy` | `.vscode/launch.json` ships pre-configured debug targets for each script. |
| *(planned, Task 6-7)* | pandas | Eval harness CSV logging + aggregation. |
| *(planned, Task 8)* | Gradio | Interactive visualizer for entropy traces and mode comparisons. |

No external API calls anywhere in the pipeline — the whole thing runs
against local model weights.

## Project structure

```
backends/
  base.py                 LMBackend interface: encode/decode/next_token_topk.
                           Deliberately stateless per call (full token
                           sequence in, distribution out) so a future
                           OpenAI-compatible backend can drop in without
                           the engine code changing.
  local_transformers.py   Concrete backend: Qwen2.5-1.5B-Instruct on the
                           local GPU. One forward pass per call, returns
                           top-K=20 renormalized next-token probabilities.
                           Known limitation: no KV-cache reuse yet (each
                           call recomputes the full sequence).

engine/
  entropy.py               Pure top-K Shannon entropy function (paper Eq. 1).
                           No model/backend knowledge at all.
  branch_policy.py          Three interchangeable "should I branch right
                           now?" functions: entropy_gated(theta) (the real
                           EAGER rule), fixed_interval(n) (ablation
                           baseline - branch on a schedule, ignore
                           entropy), never() (used by greedy/full_parallel).
  branch_tree.py            The core engine. Node (token + parent pointer +
                           entropy + branch counter), BranchTree.run()
                           grows a real tree in lockstep, sharing every
                           token before a fork. reconstruct_sequence and
                           reconstruct_trace walk the tree back to build
                           full sequences / per-token entropy traces.
  modes.py                  run_mode() dispatcher: greedy, full_parallel,
                           branch_fixed, eager - all four are really just
                           BranchTree called with a different policy
                           and/or a different number of times.

scripts/
  01_sanity_check.py       Phase 1 proof: model loads on GPU, entropy is
                           computable at every generation step.
  02_branch_tree_test.py   Proves the tree shares prefixes for real (not
                           just in theory) and that entropy correctly
                           discriminates branch points from non-branch
                           points.
  03_modes_test.py         Runs all 4 modes on the same prompt, confirms
                           full_parallel shows exactly 0% token savings
                           (by design - it shares nothing) while the two
                           tree-based modes show real savings.

data/                      (empty - reserved for Task 5's evaluation prompts)
eval/                      (empty - reserved for Task 6/7's harness + scorer)

.vscode/
  settings.json            Points VS Code at the interpreter with
                           torch/transformers installed.
  launch.json               Pre-built debug configs for each script, so
                           breakpoints in engine/ files hit correctly
                           regardless of which tab is focused.
```

## Setup

```bash
# 1. CUDA-enabled torch (must be installed explicitly - the default
#    `pip install torch` resolves to CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. everything else
pip install -r requirements.txt
```

Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Running things

```bash
python scripts/01_sanity_check.py       # per-token entropy on one prompt, no branching
python scripts/02_branch_tree_test.py   # branching + prefix-sharing, entropy-gated
python scripts/03_modes_test.py         # all 4 modes compared on the same prompt
```

Or in VS Code: `Run and Debug` panel (`Ctrl+Shift+D`) → pick the matching
config from the dropdown → set breakpoints in any `engine/*.py` file → run.

## The three knobs

Three independent parameters, checked in different places, never compared
to each other:

| Knob | Question it answers | Where it lives |
|---|---|---|
| `theta` (θ) | "Is *this specific token* uncertain enough to branch on?" | `entropy_gated(theta)` in `branch_policy.py` |
| `M` | "How many total sequences am I allowed, ever, for this prompt?" | checked inside every policy function, enforced live by `branch_tree.py` |
| `max_steps` | "How long does any single run go before a forced stop?" | passed into `BranchTree(...)`, caps the whole lockstep loop |

**Not yet calibrated.** Current values (`theta≈1.2`, small `M`/`max_steps`
in the test scripts) are informed guesses from eyeballing real entropy
traces, not the paper's actual calibration procedure (sweep θ against a
labeled validation set). Real calibration needs an accuracy signal, which
doesn't exist until the eval harness (Task 6) is built.

## The four generation modes

| Mode | Sequences | Shares prefixes? |
|---|---|---|
| `greedy` | 1 | n/a - nothing to share |
| `full_parallel` | `M` | **No** - paper's original baseline, M fully independent generations from scratch |
| `branch_fixed` | up to `M` | Yes - forks on a fixed token interval, ignoring entropy (ablation: isolates whether entropy-gating specifically matters) |
| `eager` | up to `M` | Yes - forks only where entropy crosses `theta` (the real method) |

Measured on one GSM8K-style prompt (`M=4`, `max_steps=60`): `full_parallel`
used exactly `4×60=240` tokens (proving zero sharing, by design), while
`branch_fixed` and `eager` used 163 and 104 tokens respectively.

## Status

- [x] Task 1 — Backend + local model on GPU
- [x] Task 2 — Top-K entropy computation
- [x] Task 3 — Shared-prefix branch tree
- [x] Task 4 — Four generation modes
- [ ] Task 5 — Evaluation prompt set
- [ ] Task 6 — Eval harness with CSV logging
- [ ] Task 7 — Self-consistency voting scorer (Pass@k, Cons@k, Pass Rate)
- [ ] Task 8 — Gradio visualizer

## Known limitations / open decisions

- **No KV-cache reuse.** `local_transformers.py` reruns the full forward
  pass every step. Correct, but O(n²) per sequence — a candidate
  optimization once correctness is fully verified end-to-end.
- **Sampling approximation.** The paper samples the non-branch token from
  the full vocabulary distribution; this implementation samples from the
  top-K=20 renormalized distribution instead (simpler, and the paper's own
  footnote notes top-20 already captures most of the probability mass).
- **θ/M/max_steps are placeholders**, not calibrated (see above).
- Budget reallocation (paper's Algorithm 2 / EAGER-adapt) is explicitly
  out of scope for this project.

## Citation

```
Scalena, D., Zotos, L., Fersini, E., Nissim, M., & Üstün, A. (2025).
EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling.
Proceedings of the 43rd International Conference on Machine Learning (ICML).
```
