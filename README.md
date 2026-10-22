# CS336 Learning

**中文** | [English](README-en.md)

> 自学 Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) 的完整学习仓库：
> 从 PyTorch 张量思维出发，一路走到 Tokenizer → Transformer → 训练 → GPU 系统 → Scaling Laws → Data → Alignment 的完整 LLM Training Stack。

**当前进度：Week 7 / 20（Week 1–3 已完成；Week 4–7 笔记已铺开，进行中）**

---

## 目录

- [项目简介](#项目简介)
- [学习路线](#学习路线20-周)
- [仓库结构](#仓库结构)
- [实现代码](#实现代码)
- [笔记索引](#笔记索引)
- [学习方法论](#学习方法论)
- [学习环境](#学习环境)
- [数据准备](#数据准备)
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

完整计划见 **[cs336_learning_roadmap.md](cs336_learning_roadmap.md)**（含环境方案、每周任务、最终完成状态与参考资料）。

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
CS336-Learning/
├── README.md                    # 中文说明（本文件）
├── README-en.md                 # English version
├── cs336_learning_roadmap.md    # 20 周自学路线图（环境 / 周计划 / 完成状态 / 资料）
├── pyrightconfig.json           # Pyright 类型检查配置（指向 cs336 conda 环境）
├── data/                        # 数据集（已 gitignore，见「数据准备」）
├── Code/                        # Assignment 1 核心实现
│   ├── train_bpe.py             # byte-level BPE 训练（倒排索引 + 增量更新 + heap）
│   ├── tokenizer.py             # Tokenizer 类：encode / decode / 序列化
│   └── pretokenization.py       # GPT-2 regex 预分词、special token、可流式分块
└── notes/
    ├── week01_Tensor_thinking/  # Lesson 01–05：张量思维与 Autograd
    ├── week02_Pytorch_nn/       # Lesson 06–12：PyTorch NN 基础 + Mini Project
    ├── week03_Tokenizer/        # Lesson 01–08：Tokenizer 全流程（Basic → Inference）
    ├── week04_Neural_Network_Primitives/  # Lesson 09–10：Tensor Shapes / Linear / Embedding / RMSNorm / SwiGLU / Residual
    ├── week05_Attention/        # Lesson 11：Scaled Dot-Product / Causal / Multi-Head Attention
    ├── week06_Transformer_AdamW/  # Lesson 12–13：RoPE / Transformer Block / LM / Cross Entropy / AdamW / LR Schedule
    └── week07_TinyStories_Training/  # Lesson 14–15：训练循环 / Checkpoint / Sampling / Resource Accounting
```

后续 Assignment 会按 Phase 逐步扩展出 `src/`、`tests/`、`benchmarks/`、`configs/`、`experiments/` 等目录。

> 注：`data/`、`checkpoints/`、`*.pt`、`datasets/`、`.venv/` 等数据集与训练产物已在 [.gitignore](.gitignore) 中排除，需要时按「数据准备」一节重新下载。

---

## 实现代码

`Code/` 下是 Assignment 1 的模块化实现，按「训练阶段」与「使用阶段」拆分：

| 文件 | 职责 | 关键点 |
| --- | --- | --- |
| [train_bpe.py](Code/train_bpe.py) | byte-level BPE 训练 | 倒排索引 `pair → {word}`、增量更新 `pair_freq`、heap + lazy deletion 选 winner、与 Assignment 1 一致的 `vocab` / `merges` 契约 |
| [tokenizer.py](Code/tokenizer.py) | BPE 使用阶段 | 按 merge rank 贪心合并、special token 原子化、`encode` / `decode` / `encode_iterable` / `to_files` / `from_files` |
| [pretokenization.py](Code/pretokenization.py) | 预分词与计数 | GPT-2 regex、special token 隔离、UTF-8 bytes 化、可流式安全分块的 `Counter` 累加 |

训练（`corpus → vocab + merges`）与编码（`text + vocab + merges → ids`）严格分离；对应原理与逐行推导见 [Week 3 Tokenizer](notes/week03_Tokenizer/) 与 [Week 7](notes/week07_TinyStories_Training/) 笔记。

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
| Special Tokens | [04_Special_Tokens.ipynb](notes/week03_Tokenizer/04_Special_Tokens.ipynb) | special token 的 atomic 性质、正确的处理顺序、special token pattern（长度降序 / capturing group）、round-trip 不变量、overlapping special tokens、三层 encode 逻辑与统一的 decode 结构 |
| Correct BPE Trainer | [05_Correct_BPE_Trainer.ipynb](notes/week03_Tokenizer/05_Correct_BPE_Trainer.ipynb) | weighted pair counting、tie-breaking、non-overlapping merge 与其传播、vocab 初始化与增长、segmentation 的演化、determinism / special token contamination 测试、final validator |
| Profile & Optimize BPE Training | [06_profile_optimize_bpe_training.ipynb](notes/week03_Tokenizer/06_profile_optimize_bpe_training.ipynb) | 保留「正确的慢版本」作为 oracle、profiling 实测热点、merge 的局部性、`pair_counts` + `pair_to_sequences` 反向索引、增量更新公式、merge 后 sequence「撞车」的处理、与慢版本逐轮对答案 |
| File-based BPE Correctness | [07_File_Based_BPE_Correctness.ipynb](notes/week03_Tokenizer/07_File_Based_BPE_Correctness.ipynb) | 文件分块的安全边界（delimiter 恰好落在读取块边界）、$C(T)=\sum_i C(T_i)$ 何时成立、局部 Counter 与聚合、差分测试（全文件统计 == 分块统计）、多进程 pre-tokenization，以及「为什么 BPE merge 不能每块各训一次」 |
| Tokenizer Inference | [08_Tokenizer_Inference.ipynb](notes/week03_Tokenizer/08_Tokenizer_Inference.ipynb) | merge rank 与 encode 的贪心合并、从 256 个基础 bytes 建立 vocab、special token 检测先于普通 regex、decode 先连接 bytes 再按 UTF-8 解码、round-trip 与边界情况、chunked vs streaming encode、复杂度与真实工程 |
| Tokenizer 理论笔记 | [Tokenizer.md](notes/week03_Tokenizer/Tokenizer.md) | Unicode / UTF-8 / bytes、BPE 的两个阶段、pair counting 与 tie-breaking、merge、encode / decode 流程、训练性能问题 |

### Week 4 — Neural Network Primitives 🚧

| # | Lesson | Notebook | 关键内容 |
| --- | --- | --- | --- |
| 9 | Tensor Shapes / Linear / Embedding | [09_Tensor_Shapes_Linear_Embedding.ipynb](notes/week04_Neural_Network_Primitives/09_Tensor_Shapes_Linear_Embedding.ipynb) | shape / stride / dtype、broadcasting 与 batched matmul、Linear 公式与 PyTorch 权重存储惯例、forward 相同之外还要看 backward、Embedding 是查表而非把 token ID 当数值、重复 token 的梯度累加、参数量与显存的第一层估算、极端 shape 边界 |
| 10 | RMSNorm / SwiGLU / Residual | [10_RMSNorm_SwiGLU_Residual.ipynb](notes/week04_Neural_Network_Primitives/10_RMSNorm_SwiGLU_Residual.ipynb) | 为什么需要 normalization、RMSNorm 数值实验与 ε 的作用、SwiGLU 为什么使用三次 Linear、residual 的 identity path、Pre-Norm block 的顺序、关键正确性检查 |

> 说明：Week 4 的 Softmax 与 Cross Entropy 已在 Week 2 [Lesson 09](notes/week02_Pytorch_nn/09_softmax_cross_entropy.ipynb) 中完成。

### Week 5 — Attention 🚧

| # | Lesson | Notebook | 关键内容 |
| --- | --- | --- | --- |
| 11 | Scaled Dot-Product / Causal Attention | [11_Scaled_Dot_Product_Causal_Attention.ipynb](notes/week05_Attention/11_Scaled_Dot_Product_Causal_Attention.ipynb) | 从 `(B,T,D)` 得到 `(B,H,T,D_head)` 的 reshape / transpose、scaled dot-product、causal mask 与 `-inf`、与 `F.scaled_dot_product_attention` 数值对齐、完整 Multi-Head Attention 模块、修改未来 token 检验因果性、梯度与单 token shape 边界 |

### Week 6 — Transformer + AdamW 🚧

| # | Lesson | Notebook | 关键内容 |
| --- | --- | --- | --- |
| 12 | RoPE / Transformer Block / Transformer LM | [12_RoPE_Transformer_Block_LM.ipynb](notes/week06_Transformer_AdamW/12_RoPE_Transformer_Block_LM.ipynb) | 为什么 attention 需要位置信息、RoPE 的两条性质、带 RoPE 的 Multi-Head Causal Self-Attention、Pre-Norm Transformer Block、组装 Transformer Language Model、next-token objective、prefix invariance |
| 13 | Cross Entropy / AdamW / LR Schedule | [13_Cross_Entropy_AdamW_LR_Schedule.ipynb](notes/week06_Transformer_AdamW/13_Cross_Entropy_AdamW_LR_Schedule.ipynb) | 从 logits 计算 cross entropy、为什么不能先 `exp(logits)`、梯度公式验证、AdamW 的动量 / 偏差修正 / 解耦权重衰减、AdamW ≠ Adam + L2、warmup + cosine decay、gradient clipping 作用于全局梯度范数、perplexity |

### Week 7 — TinyStories Training 🚧

| # | Lesson | Notebook | 关键内容 |
| --- | --- | --- | --- |
| 14 | Data Loading / Training Loop / Checkpoint / Sampling | [14_Data_Training_Checkpoint_Sampling.ipynb](notes/week07_TinyStories_Training/14_Data_Training_Checkpoint_Sampling.ipynb) | LM 数据加载与边界案例、最小训练用 decoder、真正执行的 training loop、train / eval 分离、checkpoint 不只保存权重、autoregressive generation 每次只生成一个 token、sampling 边界情况、走向 TinyStories |
| 15 | Assignment 1 知识整合与 Resource Accounting | [15_Assignment1_Knowledge_Resource_Accounting.ipynb](notes/week07_TinyStories_Training/15_Assignment1_Knowledge_Resource_Accounting.ipynb) | Assignment 1 整体数据流、模块接口与典型 shape、从结构推导 Transformer 参数量、矩阵乘法 FLOPs 与 context length 成本、训练显存构成、激活显存与吞吐量、从 07 到 14 的跨模块不变量、过渡到官方 Assignment 1 |

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
- **GPU**：NVIDIA GeForce RTX 3050 Ti Laptop GPU（定位与显存限制分析见 roadmap 第 25 节）
- **环境管理**：`mamba` 为主；复现 Stanford 官方 Assignment 时使用 `uv` + 官方 lockfile
- **编辑器**：VS Code + WSL Remote

环境方案（Windows 与 WSL2 的职责划分、Linux 项目文件放在 ext4 而非 `/mnt` 下）见 [cs336_learning_roadmap.md](cs336_learning_roadmap.md) 第 2 节；Git 工作流与备份策略见第 26–27 节。

笔记为 Notebook 形式，可直接运行验证；建议先自己实现，再与 PyTorch reference 对齐。

---

## 数据准备

Assignment 1 使用 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)（GPT-4 生成版）与 OpenWebText 子采样。执行以下命令下载到仓库根目录的 `data/`：

```bash
mkdir -p data
cd data

# TinyStories V2 (GPT-4)
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText 子采样
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

| 文件 | 大小（约） | 用途 |
| --- | ---: | --- |
| `TinyStoriesV2-GPT4-train.txt` | 2.1 GB | 训练语料 |
| `TinyStoriesV2-GPT4-valid.txt` | 22 MB | 验证集 |
| `owt_train.txt` / `owt_valid.txt` | 数 GB / 数十 MB | OWT 子采样，用于对比不同数据分布 |

说明：

- TinyStories 与 OWT 语料均以 `<|endoftext|>` 作为文档分隔符；
- 网络不稳定时可用 `wget -c` 断点续传，或把 `wget` 换成 `curl -L -O`；
- `data/` 已在 [.gitignore](.gitignore) 中排除，数据集不会进入仓库，换机器时重新执行上面的命令即可。

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
- [x] Week 3 — Tokenizer（Lesson 01–08：基本概念 / Toy BPE / Pre-tokenization / Special Tokens / Correct BPE / Profiling 与优化 / 文件分块与多进程 / Inference）
- [ ] Week 4 — Neural Network Primitives（🚧 Lesson 09–10：Tensor Shapes / Linear / Embedding / RMSNorm / SwiGLU / Residual）
- [ ] Week 5 — Attention（🚧 Lesson 11：Scaled Dot-Product / Causal Attention / Multi-Head）
- [ ] Week 6 — Transformer + AdamW（🚧 Lesson 12–13：RoPE / Transformer Block / LM / Cross Entropy / AdamW / LR Schedule）
- [ ] Week 7 — TinyStories Training（🚧 Lesson 14–15：数据加载 / 训练循环 / Checkpoint / Sampling / Resource Accounting）
- [ ] Phase 2–6（Systems / Scaling / Data / Alignment / Portfolio）
