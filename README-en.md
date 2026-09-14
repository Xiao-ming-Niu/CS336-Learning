# CS336 Learning

[中文](README.md) | **English**

> A study repository for Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) —
> starting from PyTorch tensor thinking and building up the full LLM training stack:
> Tokenizer → Transformer → Training → GPU Systems → Scaling Laws → Data → Alignment.

**Current progress: Week 3 / 20 (Phase 0 done, Phase 1 in progress)**

---

## Table of Contents

- [Overview](#overview)
- [Learning Roadmap](#learning-roadmap-20-weeks)
- [Repository Layout](#repository-layout)
- [Notes Index](#notes-index)
- [Study Methodology](#study-methodology)
- [Environment](#environment)
- [References](#references)
- [Commit Conventions](#commit-conventions)

---

## Overview

This repository is not a record of "watching the CS336 lectures". It is an engineering-oriented set of notes built along one capability chain:

```text
Python / PyTorch Fundamentals
        ↓
Tensor Shape Reasoning
        ↓
BPE Tokenizer
        ↓
Transformer From Scratch
        ↓
Language Model Training
        ↓
GPU / Profiling / Triton
        ↓
Distributed Training
        ↓
Scaling Laws
        ↓
Data Pipeline
        ↓
SFT / DPO / GRPO
        ↓
Mini LLM Training Stack
```

There are three target levels:

| Level | Goal |
| --- | --- |
| **Principles** | Explain why BPE works, the shape of every tensor in a Transformer, the math of attention, the role of RoPE / RMSNorm / SwiGLU, why AdamW ≠ Adam + L2, why FlashAttention is fast, what DDP / FSDP actually solve, and how scaling laws drive compute-optimal training |
| **Implementation** | Write from scratch: tokenizer, Linear / Embedding / Softmax / Cross Entropy, attention, a full Transformer, the training loop, Triton kernels, distributed training and alignment pipelines |
| **Presentation** | Turn the work into publicly readable notes, reproducible experiments, and a GitHub / Jekyll portfolio |

Notes are written primarily in **Chinese**, keeping English terminology, formulas and API names so they stay aligned with official materials, papers and interview language.

---

## Learning Roadmap (20 Weeks)

The full plan lives in **[cs336_learning_roadmap.md](cs336_learning_roadmap.md)** (environment setup, weekly tasks, milestones, interview review).

| Phase | Weeks | Focus | Main Deliverable | Status |
| --- | ---: | --- | --- | :---: |
| Phase 0 | Week 1–2 | Linux + Python / PyTorch Bootcamp | PyTorch fundamentals notes | ✅ |
| Phase 1 | Week 3–7 | Assignment 1: Basics | Transformer + TinyStories LM | 🚧 |
| Phase 2 | Week 8–11 | Assignment 2: Systems | Triton / profiling / distributed | ⬜ |
| Phase 3 | Week 12–13 | Assignment 3: Scaling | Scaling-law experiment | ⬜ |
| Phase 4 | Week 14–16 | Assignment 4: Data | Data pipeline | ⬜ |
| Phase 5 | Week 17–19 | Assignment 5: Alignment | SFT / DPO / GRPO experiments | ⬜ |
| Phase 6 | Week 20 | Portfolio | GitHub + Jekyll project page | ⬜ |

Default budget is roughly 15 hours per week:

```text
Lectures / theory     3–4 h
Implementation        6–8 h
Testing / debugging   2–3 h
Notes / GitHub        1–2 h
Interview review      1 h
```

---

## Repository Layout

```text
cs336-from-scratch/
├── README.md                    # Chinese version (default)
├── README-en.md                 # English version (this file)
├── cs336_learning_roadmap.md    # 20-week roadmap (environment / weekly plan / milestones / references)
└── notes/
    ├── week01_Tensor_thinking/  # Lesson 1–5: tensor thinking & autograd
    ├── week02_Pytorch_nn/       # Lesson 6–12: PyTorch NN fundamentals + mini project
    └── week03_Tokenizer/        # Tokenizer basics / toy BPE / pre-tokenization
```

Later phases will grow `src/`, `tests/`, `benchmarks/`, `configs/`, and `experiments/` directories as assignments progress.

> Note: `checkpoints/`, `*.pt`, `datasets/`, `.venv/` and similar artifacts are excluded via [.gitignore](.gitignore); notebooks generate them when run.

---

## Notes Index

### Week 1 — Tensor Thinking

| # | Lesson | Notebook | Highlights |
| --- | --- | --- | --- |
| 1 | Tensor & Shape Thinking | [01_tensor_basics.ipynb](notes/week01_Tensor_thinking/01_tensor_basics.ipynb) | Shape semantics, indexing / slicing, per-dimension accounting, reshape / transpose / permute, shapes in attention |
| 2 | PyTorch Broadcasting | [02_broadcasting.ipynb](notes/week01_Tensor_thinking/02_broadcasting.ipynb) | Broadcasting rules, expanding size-1 dims, reduction and `keepdim`, broadcasting in RMSNorm and attention masks |
| 3 | Matrix Multiplication & Tensor Shapes | [03_matrix_multiplication.ipynb](notes/week01_Tensor_thinking/03_matrix_multiplication.ipynb) | 2-D and high-dimensional matmul, linear projection, Q/K/V, the full self-attention shape pipeline |
| 4 | `einsum` & Tensor Contractions | [04_einsum.ipynb](notes/week01_Tensor_thinking/04_einsum.ipynb) | Einstein summation, index semantics, contraction / reorder / combine, rewriting attention with einsum, `einsum` vs `matmul` |
| 5 | Autograd & Computational Graph | [05_autograd.ipynb](notes/week01_Tensor_thinking/05_autograd.ipynb) | Computational graph, chain rule, VJP, gradient accumulation, leaf tensors, `no_grad` / `detach`, the minimal training loop |

### Week 2 — PyTorch NN Fundamentals

| # | Lesson | Notebook | Highlights |
| --- | --- | --- | --- |
| 6 | `nn.Module` & `Parameter` | [06_nn_module_parameters.ipynb](notes/week02_Pytorch_nn/06_nn_module_parameters.ipynb) | Parameter registration, module tree, `named_parameters` / `state_dict`, Parameter vs Buffer, ModuleList, train / eval |
| 7 | Linear Layer From Scratch | [07_linear_layer.ipynb](notes/week02_Pytorch_nn/07_linear_layer.ipynb) | Batched linear, PyTorch weight layout, initialization (Xavier), forward / backward reference tests, parameter count |
| 8 | Embedding From Scratch | [08_embedding.ipynb](notes/week02_Pytorch_nn/08_embedding.ipynb) | Token IDs to vectors, row lookup, gradient accumulation for repeated tokens, one-hot equivalence, weight tying |
| 9 | Softmax & Cross Entropy | [09_softmax_cross_entropy.ipynb](notes/week02_Pytorch_nn/09_softmax_cross_entropy.ipynb) | Numerically stable softmax, logsumexp, cross-entropy gradient derivation, next-token shift, perplexity, the `log V` baseline |
| 10 | SGD / Momentum / Adam / AdamW | [10_optimizers.ipynb](notes/week02_Pytorch_nn/10_optimizers.ipynb) | Hand-written SGD, first / second moments, bias correction, weight decay, Adam vs AdamW, optimizer state |
| 11 | Training Loop & DataLoader | [11_training_loop.ipynb](notes/week02_Pytorch_nn/11_training_loop.ipynb) | Dataset / DataLoader / mini-batch, epoch vs step, validation loop, gradient norm and clipping, checkpoints |
| 12 | Mini Project: Tiny Network | [12_tiny_neural_netwrk.ipynb](notes/week02_Pytorch_nn/12_tiny_neural_netwrk.ipynb) | End-to-end training of a small network: reproducibility, overfit-one-batch, accuracy, training curves, save / reload checkpoints, inference pipeline |

### Week 3 — Tokenizer

| Topic | File | Highlights |
| --- | --- | --- |
| Tokenizer Basics | [01_Tokenizer_Basic.ipynb](notes/week03_Tokenizer/01_Tokenizer_Basic.ipynb) | Four core concepts, `ord()` vs bytes, completeness of all 256 bytes, UTF-8 round trip, compression ratio |
| Toy BPE Trainer | [02_Toy_BPE_Trainer.ipynb](notes/week03_Tokenizer/02_Toy_BPE_Trainer.ipynb) | Pair counting, tie-breaking, merge implementation, vocabulary growth, toy BPE training loop, training vs encoding |
| Pre-tokenization | [03_Pre-Tokenization.ipynb](notes/week03_Tokenizer/03_Pre-Tokenization.ipynb) | GPT-2 style pre-tokenization regex, contractions / numbers / emoji, pre-token frequency vs pair frequency, the formal BPE data structures |
| Special Tokens | [04_Special_Tokens.ipynb](notes/week03_Tokenizer/04_Special_Tokens.ipynb) | Atomicity of special tokens, correct processing order, the special-token pattern (length-descending sort / capturing groups), isolating ordinary spans from special spans |
| Tokenizer Theory Notes | [Tokenizer.md](notes/week03_Tokenizer/Tokenizer.md) | Unicode / UTF-8 / bytes, the two phases of BPE, pair counting and tie-breaking, merges, encode / decode flows, training performance |

---

## Study Methodology

Every unit (lesson) follows the same template so that the loop "principles → implementation → verification → presentation" stays closed:

```text
1. Why                       Why is this technique needed
2. Math                      Formulas and derivations
3. Shapes                    All important tensor shapes
4. Naive implementation      The most understandable hand-written version
5. Reference implementation  Compare against the PyTorch reference
6. Unit tests                Correctness tests (shape / forward / backward)
7. Numerical issues          Numerical stability
8. Performance               Time / memory / kernel behavior
9. CS336 mapping             Which lecture / assignment it maps to
10. Interview questions      Questions commonly asked in ML engineer interviews
11. GitHub notes             Rewritten for public readability
```

A primitive is only considered correct when it passes all four checks:

```text
Correct shapes → Forward matches reference → Backward matches reference → Stable on extreme inputs (large logits / uniform logits)
```

---

## Environment

- **Setup**: Windows + WSL2 / Ubuntu 24.04 (installed to a non-C drive, with swap off the C drive too)
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (capability analysis in section 36 of the roadmap)
- **Env management**: mainly `mamba`; `uv` + official lockfiles when reproducing Stanford assignments
- **Editor**: VS Code + WSL Remote

Full setup steps (WSL location, swap configuration, CUDA verification, VS Code workflow) are in sections 2–12 of [cs336_learning_roadmap.md](cs336_learning_roadmap.md).

Notes are Jupyter notebooks and can be run directly. The recommended workflow is: implement it yourself first, then align against the PyTorch reference.

---

## References

- Stanford CS336: <https://cs336.stanford.edu/>
- Stanford CS336 GitHub: <https://github.com/stanford-cs336>
- PyTorch: <https://pytorch.org/>
- Triton: <https://triton-lang.org/>
- WSL docs: <https://learn.microsoft.com/windows/wsl/install>
- Mamba: <https://mamba.readthedocs.io/>

Always prefer the latest official materials.

---

## Commit Conventions

Let the Git history itself become the learning trajectory. Avoid low-information messages like `update` / `fix` / `test`:

```bash
git status
git add
git commit -m "Implement byte-level BPE merge loop"
git push
```

Examples of good messages:

```text
Implement byte-level BPE merge loop
Add RMSNorm reference tests
Benchmark causal attention on RTX 3050 Ti
Fix RoPE broadcasting bug
```

---

## Roadmap Progress

- [x] Week 1 — Tensor Thinking (Lessons 1–5)
- [x] Week 2 — PyTorch NN Fundamentals (Lessons 6–12, including the mini project)
- [x] Week 3 — Tokenizer (basics / toy BPE / pre-tokenization)
- [ ] Week 4 — Neural Network Primitives
- [ ] Week 5 — Attention
- [ ] Week 6 — Transformer + AdamW
- [ ] Week 7 — TinyStories Training
- [ ] Phases 2–6 (Systems / Scaling / Data / Alignment / Portfolio)
