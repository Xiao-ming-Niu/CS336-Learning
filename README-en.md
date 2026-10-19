# CS336 Learning

[中文](README.md) | **English**

> A study repository for Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) —
> starting from PyTorch tensor thinking and building up the full LLM training stack:
> Tokenizer → Transformer → Training → GPU Systems → Scaling Laws → Data → Alignment.

**Current progress: Week 7 / 20 (Weeks 1–3 done; Weeks 4–7 drafted and in progress)**

---

## Table of Contents

- [Overview](#overview)
- [Learning Roadmap](#learning-roadmap-20-weeks)
- [Repository Layout](#repository-layout)
- [Notes Index](#notes-index)
- [Study Methodology](#study-methodology)
- [Environment](#environment)
- [Data Preparation](#data-preparation)
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

The full plan lives in **[cs336_learning_roadmap.md](cs336_learning_roadmap.md)** (environment plan, weekly tasks, final completion criteria, references).

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
CS336-Learning/
├── README.md                    # Chinese version (default)
├── README-en.md                 # English version (this file)
├── cs336_learning_roadmap.md    # 20-week roadmap (environment / weekly plan / completion criteria / references)
├── pyrightconfig.json           # Pyright type-checking config (points at the cs336 conda env)
├── data/                        # Datasets (gitignored, see "Data Preparation")
└── notes/
    ├── week01_Tensor_thinking/  # Lesson 01–05: tensor thinking & autograd
    ├── week02_Pytorch_nn/       # Lesson 06–12: PyTorch NN fundamentals + mini project
    ├── week03_Tokenizer/        # Lesson 01–08: the full tokenizer pipeline (basics → inference)
    ├── week04_Neural_Network_Primitives/  # Lesson 09–10: tensor shapes / linear / embedding / RMSNorm / SwiGLU / residual
    ├── week05_Attention/        # Lesson 11: scaled dot-product / causal / multi-head attention
    ├── week06_Transformer_AdamW/  # Lesson 12–13: RoPE / transformer block / LM / cross entropy / AdamW / LR schedule
    └── week07_TinyStories_Training/  # Lesson 14–15: training loop / checkpoints / sampling / resource accounting
```

Later phases will grow `src/`, `tests/`, `benchmarks/`, `configs/`, and `experiments/` directories as assignments progress.

> Note: `data/`, `checkpoints/`, `*.pt`, `datasets/`, `.venv/` and similar datasets and artifacts are excluded via [.gitignore](.gitignore); re-download them using the [Data Preparation](#data-preparation) section when needed.

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
| Special Tokens | [04_Special_Tokens.ipynb](notes/week03_Tokenizer/04_Special_Tokens.ipynb) | Atomicity of special tokens, correct processing order, the special-token pattern (length-descending sort / capturing groups), round-trip invariants, overlapping special tokens, the three-layer encode flow and a unified decode structure |
| Correct BPE Trainer | [05_Correct_BPE_Trainer.ipynb](notes/week03_Tokenizer/05_Correct_BPE_Trainer.ipynb) | Weighted pair counting, tie-breaking, non-overlapping merge and its propagation, vocabulary initialization and growth, evolution of segmentation, determinism / special-token contamination tests, final validator |
| Profile & Optimize BPE Training | [06_profile_optimize_bpe_training.ipynb](notes/week03_Tokenizer/06_profile_optimize_bpe_training.ipynb) | Keeping the correct-but-slow version as an oracle, profiling to find the real hotspot, merge locality, the `pair_counts` + `pair_to_sequences` inverted index, the incremental-update formula, handling sequences that "collide" after a merge, checking round by round against the slow version |
| File-based BPE Correctness | [07_File_Based_BPE_Correctness.ipynb](notes/week03_Tokenizer/07_File_Based_BPE_Correctness.ipynb) | Safe chunk boundaries for a file (delimiters landing exactly on a read boundary), when $C(T)=\sum_i C(T_i)$ holds, local counters and aggregation, differential testing (whole-file counts == chunked counts), multi-process pre-tokenization, and why BPE merges cannot be trained per chunk |
| Tokenizer Inference | [08_Tokenizer_Inference.ipynb](notes/week03_Tokenizer/08_Tokenizer_Inference.ipynb) | Merge ranks and greedy merging during encode, building a vocab from the 256 base bytes, detecting special tokens before ordinary regex, decode concatenating bytes before UTF-8 decoding, round-trip and edge cases, chunked vs streaming encode, complexity and real-world engineering |
| Tokenizer Theory Notes | [Tokenizer.md](notes/week03_Tokenizer/Tokenizer.md) | Unicode / UTF-8 / bytes, the two phases of BPE, pair counting and tie-breaking, merges, encode / decode flows, training performance |

### Week 4 — Neural Network Primitives 🚧

| # | Lesson | Notebook | Highlights |
| --- | --- | --- | --- |
| 9 | Tensor Shapes / Linear / Embedding | [09_Tensor_Shapes_Linear_Embedding.ipynb](notes/week04_Neural_Network_Primitives/09_Tensor_Shapes_Linear_Embedding.ipynb) | Shape / stride / dtype, broadcasting and batched matmul, the linear formula vs PyTorch's weight layout, matching backward and not just forward, embedding as a table lookup rather than numeric token IDs, gradient accumulation for repeated tokens, first-order parameter and memory estimates, extreme-shape boundaries |
| 10 | RMSNorm / SwiGLU / Residual | [10_RMSNorm_SwiGLU_Residual.ipynb](notes/week04_Neural_Network_Primitives/10_RMSNorm_SwiGLU_Residual.ipynb) | Why normalization is needed, numerical experiments on RMSNorm and the role of ε, why SwiGLU uses three linears, the identity path of a residual connection, the ordering inside a pre-norm block, key correctness checks |

> Note: Softmax and cross entropy for Week 4 were already covered in Week 2 [Lesson 09](notes/week02_Pytorch_nn/09_softmax_cross_entropy.ipynb).

### Week 5 — Attention 🚧

| # | Lesson | Notebook | Highlights |
| --- | --- | --- | --- |
| 11 | Scaled Dot-Product / Causal Attention | [11_Scaled_Dot_Product_Causal_Attention.ipynb](notes/week05_Attention/11_Scaled_Dot_Product_Causal_Attention.ipynb) | Turning `(B,T,D)` into `(B,H,T,D_head)` via reshape / transpose, scaled dot-product, causal masking with `-inf`, matching `F.scaled_dot_product_attention` numerically, a full multi-head attention module, checking causality by perturbing future tokens, gradients and single-token shape boundaries |

### Week 6 — Transformer + AdamW 🚧

| # | Lesson | Notebook | Highlights |
| --- | --- | --- | --- |
| 12 | RoPE / Transformer Block / Transformer LM | [12_RoPE_Transformer_Block_LM.ipynb](notes/week06_Transformer_AdamW/12_RoPE_Transformer_Block_LM.ipynb) | Why attention needs positional information, the two properties of RoPE, multi-head causal self-attention with RoPE, building a pre-norm transformer block, assembling a transformer language model, the next-token objective, prefix invariance |
| 13 | Cross Entropy / AdamW / LR Schedule | [13_Cross_Entropy_AdamW_LR_Schedule.ipynb](notes/week06_Transformer_AdamW/13_Cross_Entropy_AdamW_LR_Schedule.ipynb) | Cross entropy from logits, why you must not `exp(logits)` first, verifying the gradient formula, AdamW's momentum / bias correction / decoupled weight decay, AdamW ≠ Adam + L2, warmup + cosine decay, gradient clipping on the global norm, perplexity |

### Week 7 — TinyStories Training 🚧

| # | Lesson | Notebook | Highlights |
| --- | --- | --- | --- |
| 14 | Data Loading / Training Loop / Checkpoint / Sampling | [14_Data_Training_Checkpoint_Sampling.ipynb](notes/week07_TinyStories_Training/14_Data_Training_Checkpoint_Sampling.ipynb) | LM data loading and its edge cases, a minimal decoder for training, a training loop that actually runs, keeping train and eval separate, checkpoints that hold more than weights, autoregressive generation one token at a time, sampling edge cases, moving toward TinyStories |
| 15 | Assignment 1 Integration & Resource Accounting | [15_Assignment1_Knowledge_Resource_Accounting.ipynb](notes/week07_TinyStories_Training/15_Assignment1_Knowledge_Resource_Accounting.ipynb) | The overall Assignment 1 data flow, module interfaces and typical shapes, deriving transformer parameter counts from the architecture, matmul FLOPs and the cost of context length, what training memory actually contains, activation memory and throughput, cross-module invariants from lesson 07 to 14, transitioning to the official Assignment 1 |

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
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (capability analysis in section 25 of the roadmap)
- **Env management**: mainly `mamba`; `uv` + official lockfiles when reproducing Stanford assignments
- **Editor**: VS Code + WSL Remote

The environment plan (dividing responsibilities between Windows and WSL2, keeping Linux project files on ext4 instead of `/mnt`) is in section 2 of [cs336_learning_roadmap.md](cs336_learning_roadmap.md); Git workflow and backup strategy are in sections 26–27.

Notes are Jupyter notebooks and can be run directly. The recommended workflow is: implement it yourself first, then align against the PyTorch reference.

---

## Data Preparation

Assignment 1 uses [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (the GPT-4 generated version) plus a subsample of OpenWebText. Download them into `data/` at the repository root:

```bash
mkdir -p data
cd data

# TinyStories V2 (GPT-4)
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText subsample
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

| File | Approx. size | Purpose |
| --- | ---: | --- |
| `TinyStoriesV2-GPT4-train.txt` | 2.1 GB | Training corpus |
| `TinyStoriesV2-GPT4-valid.txt` | 22 MB | Validation split |
| `owt_train.txt` / `owt_valid.txt` | a few GB / tens of MB | OWT subsample, for comparing data distributions |

Notes:

- Both the TinyStories and OWT corpora use `<|endoftext|>` as the document separator;
- If the network is flaky, use `wget -c` to resume, or replace `wget` with `curl -L -O`;
- `data/` is excluded via [.gitignore](.gitignore), so datasets never enter the repository; re-run the commands above on a new machine.

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
- [x] Week 3 — Tokenizer (Lessons 01–08: basics / toy BPE / pre-tokenization / special tokens / correct BPE / profiling and optimization / file-based chunking and multi-processing / inference)
- [ ] Week 4 — Neural Network Primitives (🚧 Lessons 09–10: tensor shapes / linear / embedding / RMSNorm / SwiGLU / residual)
- [ ] Week 5 — Attention (🚧 Lesson 11: scaled dot-product / causal attention / multi-head)
- [ ] Week 6 — Transformer + AdamW (🚧 Lessons 12–13: RoPE / transformer block / LM / cross entropy / AdamW / LR schedule)
- [ ] Week 7 — TinyStories Training (🚧 Lessons 14–15: data loading / training loop / checkpoints / sampling / resource accounting)
- [ ] Phases 2–6 (Systems / Scaling / Data / Alignment / Portfolio)
