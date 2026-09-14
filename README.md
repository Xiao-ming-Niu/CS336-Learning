# CS336 Learning

**中文** | [English](README-en.md)

> 自学 Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) 的完整学习仓库：
> 从 PyTorch 张量思维出发，一路走到 Tokenizer → Transformer → 训练 → GPU 系统 → Scaling Laws → Data → Alignment 的完整 LLM Training Stack。

**当前进度：Week 3 / 20（Phase 0 完成，Phase 1 进行中）**

---

## 目录

- [项目简介](#项目简介)
- [学习路线](#学习路线20-周)
- [仓库结构](#仓库结构)
- [笔记索引](#笔记索引)
- [学习方法论](#学习方法论)
- [学习环境](#学习环境)
- [参考资料](#参考资料)
- [Git 提交规范](#git-提交规范)

---

## 项目简介

这个仓库不是「把 CS336 视频看完」的记录，而是围绕一条能力链逐步构建的工程化笔记：

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

目标分三层：

| 层次 | 目标 |
| --- | --- |
| **原理层** | 能解释 BPE 为什么工作、Transformer 每个张量的 shape、Attention 的数学推导、RoPE / RMSNorm / SwiGLU 的作用、AdamW 与 Adam + L2 的差异、FlashAttention 为什么快、DDP / FSDP 在解决什么问题、Scaling law 如何指导 compute-optimal training |
| **实现层** | 从零手写 tokenizer、Linear / Embedding / Softmax / Cross Entropy、Attention、完整 Transformer、训练循环、Triton kernel、分布式训练与 alignment 流程 |
| **展示层** | 以可公开阅读的笔记 + 可复现实验 + GitHub / Jekyll 作品集沉淀学习轨迹 |

所有笔记以 **中文为主**，保留英文术语、公式与 API 名称，便于与官方资料、论文和面试表达对齐。

---

## 学习路线（20 周）

完整计划见 **[cs336_learning_roadmap.md](cs336_learning_roadmap.md)**（含环境搭建、每周任务、里程碑与面试复盘）。

| 阶段 | 周数 | 核心内容 | 主要产出 | 状态 |
| --- | ---: | --- | --- | :---: |
| Phase 0 | Week 1–2 | Linux + Python / PyTorch Bootcamp | PyTorch fundamentals notes | ✅ |
| Phase 1 | Week 3–7 | Assignment 1: Basics | Transformer + TinyStories LM | 🚧 |
| Phase 2 | Week 8–11 | Assignment 2: Systems | Triton / profiling / distributed | ⬜ |
| Phase 3 | Week 12–13 | Assignment 3: Scaling | Scaling-law experiment | ⬜ |
| Phase 4 | Week 14–16 | Assignment 4: Data | Data pipeline | ⬜ |
| Phase 5 | Week 17–19 | Assignment 5: Alignment | SFT / DPO / GRPO experiments | ⬜ |
| Phase 6 | Week 20 | Portfolio | GitHub + Jekyll 项目页 | ⬜ |

默认每周约 15 小时：

```text
理论 / Lecture     3–4 h
代码实现            6–8 h
测试 / Debug        2–3 h
笔记 / GitHub       1–2 h
面试复盘            1 h
```

---

## 仓库结构

```text
cs336-from-scratch/
├── README.md                    # 中文说明（本文件）
├── README-en.md                 # English version
├── cs336_learning_roadmap.md    # 20 周自学路线图（环境 / 周计划 / 里程碑 / 资料）
└── notes/
    ├── week01_Tensor_thinking/  # Lesson 1–5：张量思维与 Autograd
    ├── week02_Pytorch_nn/       # Lesson 6–12：PyTorch NN 基础 + Mini Project
    └── week03_Tokenizer/        # Tokenizer 基础 / Toy BPE / Pre-tokenization
```

后续 Assignment 会按 Phase 逐步扩展出 `src/`、`tests/`、`benchmarks/`、`configs/`、`experiments/` 等目录。

> 注：`checkpoints/`、`*.pt`、`datasets/`、`.venv/` 等训练产物与环境目录已在 [.gitignore](.gitignore) 中排除，Notebook 运行时会自行生成。

---

## 笔记索引

### Week 1 — Tensor Thinking

| # | Lesson | Notebook | 关键内容 |
| --- | --- | --- | --- |
| 1 | Tensor & Shape Thinking | [01_tensor_basics.ipynb](notes/week01_Tensor_thinking/01_tensor_basics.ipynb) | Shape 语义、indexing / slicing、逐维记账法、reshape / transpose / permute、Attention 中的 shape |
| 2 | PyTorch Broadcasting | [02_broadcasting.ipynb](notes/week01_Tensor_thinking/02_broadcasting.ipynb) | Broadcasting rule、维度 1 的扩展、reduction 与 `keepdim`、RMSNorm / Attention Mask 中的 broadcasting |
| 3 | Matrix Multiplication & Tensor Shapes | [03_matrix_multiplication.ipynb](notes/week01_Tensor_thinking/03_matrix_multiplication.ipynb) | 二维 matmul、高维 matmul、Linear projection、Q/K/V、完整 Self-Attention shape pipeline |
| 4 | `einsum` & Tensor Contractions | [04_einsum.ipynb](notes/week01_Tensor_thinking/04_einsum.ipynb) | Einstein summation、index 语义、contraction / reorder / combine、用 einsum 重写 Attention、`einsum` vs `matmul` |
| 5 | Autograd & Computational Graph | [05_autograd.ipynb](notes/week01_Tensor_thinking/05_autograd.ipynb) | Computational graph、chain rule、VJP、gradient accumulation、leaf tensor、`no_grad` / `detach`、最小训练循环 |

### Week 2 — PyTorch NN Fundamentals

| # | Lesson | Notebook | 关键内容 |
| --- | --- | --- | --- |
| 6 | `nn.Module` & `Parameter` | [06_nn_module_parameters.ipynb](notes/week02_Pytorch_nn/06_nn_module_parameters.ipynb) | Parameter 注册、Module tree、`named_parameters` / `state_dict`、Parameter vs Buffer、ModuleList、train / eval |
| 7 | Linear Layer From Scratch | [07_linear_layer.ipynb](notes/week02_Pytorch_nn/07_linear_layer.ipynb) | Batch linear、PyTorch weight shape、初始化（Xavier）、forward / backward reference test、参数量 |
| 8 | Embedding From Scratch | [08_embedding.ipynb](notes/week02_Pytorch_nn/08_embedding.ipynb) | Token ID → vector、row lookup、重复 token 的 gradient accumulation、one-hot 等价性、weight tying |
| 9 | Softmax & Cross Entropy | [09_softmax_cross_entropy.ipynb](notes/week02_Pytorch_nn/09_softmax_cross_entropy.ipynb) | 数值稳定 softmax、logsumexp、cross entropy 梯度推导、next-token shift、perplexity、`log V` baseline |
| 10 | SGD / Momentum / Adam / AdamW | [10_optimizers.ipynb](notes/week02_Pytorch_nn/10_optimizers.ipynb) | 手写 SGD、first / second moment、bias correction、weight decay、Adam 与 AdamW 的区别、optimizer state |
| 11 | Training Loop & DataLoader | [11_training_loop.ipynb](notes/week02_Pytorch_nn/11_training_loop.ipynb) | Dataset / DataLoader / mini-batch、epoch 与 step、validation loop、gradient norm & clipping、checkpoint |
| 12 | Mini Project: Tiny Network | [12_tiny_neural_netwrk.ipynb](notes/week02_Pytorch_nn/12_tiny_neural_netwrk.ipynb) | 端到端训练一个小网络：可复现性、overfit-one-batch、accuracy、training curve、保存与 reload checkpoint、inference pipeline |

### Week 3 — Tokenizer

| 内容 | 文件 | 关键内容 |
| --- | --- | --- |
| Tokenizer Basic | [01_Tokenizer_Basic.ipynb](notes/week03_Tokenizer/01_Tokenizer_Basic.ipynb) | tokenizer 的四个概念、`ord()` vs bytes、256 个 byte 的完整性、UTF-8 round trip、compression ratio |
| Toy BPE Trainer | [02_Toy_BPE_Trainer.ipynb](notes/week03_Tokenizer/02_Toy_BPE_Trainer.ipynb) | pair counting、tie-breaking、merge 实现、vocabulary 增长、toy BPE training loop、training 与 encoding 的区别 |
| Pre-tokenization | [03_Pre-Tokenization.ipynb](notes/week03_Tokenizer/03_Pre-Tokenization.ipynb) | GPT-2 style pre-tokenization regex、contractions / numbers / emoji 的处理、pre-token frequency 与 pair frequency、正式的 BPE 数据结构 |
| Special Tokens | [04_Special_Tokens.ipynb](notes/week03_Tokenizer/04_Special_Tokens.ipynb) | special token 的 atomic 性质、正确的处理顺序、special token pattern（长度降序 / capturing group）、隔离 ordinary span 与 special span |
| Tokenizer 理论笔记 | [Tokenizer.md](notes/week03_Tokenizer/Tokenizer.md) | Unicode / UTF-8 / bytes、BPE 的两个阶段、pair counting 与 tie-breaking、merge、encode / decode 流程、训练性能问题 |

---

## 学习方法论

每个知识单元（每一课）统一使用同一套模板，保证「原理 → 实现 → 验证 → 展示」闭环：

```text
1. Why                为什么需要这个技术
2. Math               公式与推导
3. Shapes             所有重要 tensor shape
4. Naive implementation   最容易理解的手写版本
5. Reference implementation  与 PyTorch 参考实现对比
6. Unit tests         正确性测试（shape / forward / backward）
7. Numerical issues   数值稳定性
8. Performance        时间 / 显存 / kernel 行为
9. CS336 mapping      对应 Lecture / Assignment
10. Interview questions    算法岗面试问题
11. GitHub notes      整理成适合公开展示的内容
```

一个 primitive 必须同时通过四类测试才算「实现正确」：

```text
Shape 正确  →  Forward 与 reference 对齐  →  Backward 与 reference 对齐  →  极端输入（大 logits / uniform logits）稳定
```

---

## 学习环境

- **开发环境**：Windows + WSL2 / Ubuntu 24.04（安装到非 C 盘，swap 亦不占 C 盘）
- **GPU**：NVIDIA GeForce RTX 3050 Ti Laptop GPU（定位与显存限制分析见 roadmap 第 36 节）
- **环境管理**：`mamba` 为主；复现 Stanford 官方 Assignment 时使用 `uv` + 官方 lockfile
- **编辑器**：VS Code + WSL Remote

环境搭建的完整步骤（WSL 安装位置、swap 配置、CUDA 验证、VS Code 工作流）见 [cs336_learning_roadmap.md](cs336_learning_roadmap.md) 第 2–12 节。

笔记为 Notebook 形式，可直接运行验证；建议先自己实现，再与 PyTorch reference 对齐。

---

## 参考资料

- Stanford CS336: <https://cs336.stanford.edu/>
- Stanford CS336 GitHub: <https://github.com/stanford-cs336>
- PyTorch: <https://pytorch.org/>
- Triton: <https://triton-lang.org/>
- WSL 文档: <https://learn.microsoft.com/windows/wsl/install>
- Mamba: <https://mamba.readthedocs.io/>

始终以最新官方资料为准。

---

## Git 提交规范

让 Git history 本身成为学习轨迹，避免 `update` / `fix` / `test` 这类无信息量的 message：

```bash
git status
git add
git commit -m "Implement byte-level BPE merge loop"
git push
```

推荐的 message 风格：

```text
Implement byte-level BPE merge loop
Add RMSNorm reference tests
Benchmark causal attention on RTX 3050 Ti
Fix RoPE broadcasting bug
```

---

## Roadmap 进度

- [x] Week 1 — Tensor Thinking（Lesson 1–5）
- [x] Week 2 — PyTorch NN Fundamentals（Lesson 6–12，含 Mini Project）
- [x] Week 3 — Tokenizer（基本概念 / Toy BPE / Pre-tokenization）
- [ ] Week 4 — Neural Network Primitives
- [ ] Week 5 — Attention
- [ ] Week 6 — Transformer + AdamW
- [ ] Week 7 — TinyStories Training
- [ ] Phase 2–6（Systems / Scaling / Data / Alignment / Portfolio）
