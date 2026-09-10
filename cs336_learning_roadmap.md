# Stanford CS336 自学路线：从 PyTorch 基础到完整 LLM Training Stack

> 适用目标：算法岗 / 面试 / 完成 Stanford CS336 全部 Assignment / 构建 GitHub 与 Jekyll 作品集  
> 学习语言：中文为主，保留英文术语、公式与 API 名称  
> 周投入：10–20 小时（默认按 15 小时规划）  
> 预计周期：约 20 周  
> 本地设备：Dell Inspiron 7510 / NVIDIA GeForce RTX 3050 Ti Laptop GPU / 16 GB RAM  
> 主系统：Windows  
> Linux 开发环境：WSL2 + Ubuntu 24.04，安装到非 C 盘  
> Python 环境管理：mamba 为主；Stanford 官方作业必要时使用 uv 复现官方 lockfile  
> Portfolio：GitHub + Jekyll GitHub Pages

---

## 0. 最终目标

这份计划不是“把 CS336 视频看完”，而是完成下面这条能力链：

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
        ↓
GitHub Portfolio + Jekyll Project Page
```

最终希望达到三个层面的能力：

### 0.1 原理层

能够解释：

- BPE 为什么工作；
- Transformer 每个张量的 shape；
- Attention 的数学推导；
- RoPE、RMSNorm、SwiGLU 的作用；
- AdamW 为什么与 Adam + L2 不完全等价；
- FlashAttention 为什么快；
- GPU kernel 为什么常常受 memory bandwidth 限制；
- DDP、FSDP / sharding 在解决什么问题；
- Scaling law 如何指导 compute-optimal training；
- pretraining data 的 filtering / deduplication 为什么重要；
- SFT、DPO、PPO、GRPO 的目标函数和区别。

### 0.2 实现层

能够独立实现并测试：

- Byte-level BPE tokenizer
- Linear / Embedding / RMSNorm / SwiGLU
- Multi-Head Causal Self-Attention
- RoPE
- Transformer block
- Transformer LM
- Cross Entropy
- AdamW
- LR scheduler
- Training loop
- Sampling
- Basic GPU benchmarks
- Triton kernels
- Distributed training primitives
- Scaling-law experiments
- Data filtering / deduplication pipeline
- SFT / preference optimization / GRPO 核心逻辑

### 0.3 求职展示层

最终 GitHub 不只是“5 个作业仓库”，而是一个完整的技术叙事：

```text
I implemented a language-model training stack from scratch,
then studied how to make it fast, scalable, data-efficient,
and alignable.
```

---

# 1. 总体时间规划

| 阶段    |       周数 | 核心内容                        | 主要产出                         |
| ------- | ---------: | ------------------------------- | -------------------------------- |
| Phase 0 |   Week 1–2 | Linux + Python/PyTorch Bootcamp | PyTorch fundamentals notes       |
| Phase 1 |   Week 3–7 | CS336 Assignment 1: Basics      | Transformer + TinyStories LM     |
| Phase 2 |  Week 8–11 | Assignment 2: Systems           | Triton / profiling / distributed |
| Phase 3 | Week 12–13 | Assignment 3: Scaling           | Scaling-law experiment           |
| Phase 4 | Week 14–16 | Assignment 4: Data              | Data pipeline                    |
| Phase 5 | Week 17–19 | Assignment 5: Alignment         | SFT / DPO / GRPO experiments     |
| Phase 6 |    Week 20 | Portfolio                       | GitHub + Jekyll 项目页           |

默认每周按照约 15 小时规划：

```text
理论 / Lecture       3–4 h
代码实现              6–8 h
测试 / Debug          2–3 h
笔记 / GitHub         1–2 h
面试复盘              1 h
```

如果某周只能投入 10 小时，优先级：

```text
实现 > 测试 > Lecture > 笔记美化
```

如果能投入 20 小时：

```text
增加 benchmark、ablation、论文阅读和面试题整理
```

---

# 2. 环境总方案

## 2.1 Windows 与 Linux 的职责

```text
Dell Inspiron 7510
│
├── Windows
│   ├── 浏览器 / Office / 日常软件
│   ├── Windows Terminal
│   ├── VS Code GUI
│   ├── NVIDIA Windows Driver
│   └── Jekyll（如果当前工作流已经稳定）
│
└── WSL2
    └── Ubuntu 24.04
        ├── Git
        ├── Python
        ├── Miniforge / mamba
        ├── PyTorch
        ├── CUDA runtime access
        ├── Triton
        ├── Stanford CS336
        └── 所有 CS336 Linux 项目数据
```

原则：

> 以后 CS336 / PyTorch / Triton 的 Linux 项目文件放在 Ubuntu 自己的 ext4 Linux 文件系统中，而不是 `/mnt/c/...` 或 `/mnt/d/...` 中。

虽然 Ubuntu 的虚拟磁盘文件物理上存储在 D:/E:/F:，但 Linux 内部仍然使用：

```text
/home/<username>/projects
```

这种 Linux 原生目录。

---

# 3. 将 WSL2 / Ubuntu 安装到非 C 盘

下面以 `D:` 为例。

如果你实际希望使用 `E:` 或 `F:`，把所有 `D:` 替换即可。

推荐目录：

```text
D:\
└── WSL\
    ├── Ubuntu-24.04\
    ├── swap\
    └── backups\
```

## 3.1 不建议的结构

不要把 Linux repo 直接放成：

```text
D:\projects\cs336
```

然后在 WSL 中通过：

```text
/mnt/d/projects/cs336
```

开发。

原因：这仍然属于 Windows NTFS 文件系统，Linux 工具大量访问小文件时通常不如 WSL 内部 ext4 文件系统合适。

正确思路是：

```text
D:\WSL\Ubuntu-24.04\...
            ↓
         ext4.vhdx
            ↓
Ubuntu 看见：
/home/<username>/projects/cs336
```

也就是说：

> 物理空间占 D 盘，但开发时仍然按照正常 Linux 文件系统使用。

---

## 3.2 安装 WSL 本体，但先不安装 Linux distribution

以管理员身份打开 PowerShell。

先检查：

```powershell
wsl --version
wsl --status
```

如果 WSL 尚未安装：

```powershell
wsl --install --no-distribution
```

如果系统提示重启：

```text
Restart Windows
```

重启后继续。

更新 WSL：

```powershell
wsl --update
```

检查：

```powershell
wsl --version
```

---

## 3.3 查看 Ubuntu 名称

```powershell
wsl --list --online
```

目标通常是：

```text
Ubuntu-24.04
```

---

## 3.4 创建非 C 盘目录

例如：

```powershell
New-Item -ItemType Directory -Force D:\WSL\Ubuntu-24.04
New-Item -ItemType Directory -Force D:\WSL\swap
New-Item -ItemType Directory -Force D:\WSL\backups
```

---

## 3.5 直接把 Ubuntu 安装到 D 盘

新版 WSL 支持 `--location`。

运行：

```powershell
wsl --install -d Ubuntu-24.04 --location D:\WSL\Ubuntu-24.04
```

安装后检查：

```powershell
wsl --list --verbose
```

应该看到类似：

```text
NAME             STATE     VERSION
Ubuntu-24.04     Stopped   2
```

确认 `VERSION = 2`。

如果不是：

```powershell
wsl --set-version Ubuntu-24.04 2
```

---

# 4. 让 WSL swap 也不占 C 盘

WSL2 默认可能在 Windows 临时目录创建 swap VHD。

如果希望 Linux 的大体积虚拟磁盘数据尽量都落在其他盘，可以配置：

```text
%UserProfile%\.wslconfig
```

注意：

- `.wslconfig` 这个文本文件本身仍会位于 `C:\Users\<username>`；
- 它只有几行文本，体积可忽略；
- 真正可能占 GB 级空间的 swap VHD 可以放到 D 盘。

推荐在 Windows 的 WSL Settings 中配置；也可以手工创建：

```ini
[wsl2]
memory=10GB
swap=6GB
swapFile=D:\\WSL\\swap\\wsl-swap.vhdx
```

说明：

### memory=10GB

电脑总内存 16 GB。

先给 WSL 10 GB 上限，给 Windows 保留约 6 GB。

以后如果发现：

```text
Windows 很卡
```

可以改成：

```ini
memory=8GB
```

如果 Windows 几乎只运行 VS Code / 浏览器，并且 WSL 工作负载确实需要更多 RAM，再根据实际使用调整。

### swap=6GB

swap 不是 GPU VRAM。

它只是在物理 RAM 不够时提供磁盘级缓冲。

机器学习训练如果大量进入 swap 会非常慢，因此：

> swap 是防止某些任务直接 OOM 的安全垫，不是正常训练内存。

修改后：

```powershell
wsl --shutdown
```

重新进入 Ubuntu。

---

# 5. Linux 文件到底保存在哪里

假设 Ubuntu 安装目录是：

```text
D:\WSL\Ubuntu-24.04
```

Linux 内部：

```bash
cd ~
pwd
```

会看到：

```text
/home/<username>
```

以后创建：

```bash
mkdir -p ~/projects
mkdir -p ~/datasets
mkdir -p ~/checkpoints
```

那么这些 Linux 文件：

```text
/home/<username>/projects
/home/<username>/datasets
/home/<username>/checkpoints
```

最终都会存储到 D 盘 Ubuntu 的虚拟磁盘中。

因此你的目标可以实现：

```text
CS336 source code      → D:
mamba environments     → D:
pip/conda packages     → D:
datasets               → D:
model checkpoints      → D:
Linux home directory   → D:
Linux system packages  → D:
WSL swap               → D:
```

仍然可能存在少量 WSL / Windows 组件、配置和系统级元数据位于 C 盘，这是 Windows/WSL 本身的一部分；但 Linux distribution 的主要数据和可增长磁盘内容可以放到其他盘。

---

# 6. WSL 数据目录建议

Ubuntu 内部推荐：

```text
/home/<username>/
├── projects/
│   ├── cs336-from-scratch/
│   ├── cs336-assignment1/
│   ├── cs336-assignment2/
│   ├── cs336-assignment3/
│   ├── cs336-assignment4/
│   └── cs336-assignment5/
│
├── datasets/
│   ├── tinystories/
│   ├── openwebtext/
│   └── common-crawl/
│
└── checkpoints/
    ├── a1/
    ├── scaling/
    └── alignment/
```

但注意：

> 特别大的原始 dataset / checkpoint 后期可以单独设计存储策略，不一定全部塞进 Ubuntu 主 VHD。

原因是 WSL2 的 ext4 VHD 会动态增长。

刚开始不需要复杂化。

---

# 7. 第一次进入 Ubuntu

第一次启动：

```powershell
wsl -d Ubuntu-24.04
```

创建：

```text
UNIX username
password
```

Linux 输入密码时不会显示 `***`，这是正常行为。

然后：

```bash
pwd
whoami
uname -a
```

学习最常用的 Linux 命令：

```bash
pwd
ls
cd
mkdir
cp
mv
rm
cat
less
grep
find
```

第一周只需要真正熟练：

```text
pwd
ls
cd
mkdir
cp
mv
rm
```

---

# 8. Ubuntu 初始化

更新系统：

```bash
sudo apt update
sudo apt upgrade -y
```

安装基础工具：

```bash
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    zip \
    ca-certificates
```

检查：

```bash
git --version
gcc --version
```

---

# 9. GPU 验证

不要在 Ubuntu 里安装 Linux NVIDIA display driver。

WSL2 使用 Windows 侧 NVIDIA driver 提供 GPU virtualization。

Ubuntu 内运行：

```bash
nvidia-smi
```

目标是识别：

```text
NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

之后 PyTorch 再验证：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

预期：

```text
True
NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

---

# 10. mamba 策略

继续使用 mamba。

Windows mamba 和 WSL Linux mamba 是两套环境。

推荐：

```text
Windows
└── 原来的 mamba

WSL Ubuntu
└── Miniforge
    └── mamba
```

Linux 中 Miniforge 安装在：

```text
/home/<username>/miniforge3
```

因为整个 Ubuntu distribution 位于非 C 盘，所以：

```text
mamba env
package cache
Python interpreter
PyTorch
```

也都实际占用非 C 盘空间。

---

# 11. mamba 与 uv 的分工

不需要放弃 mamba。

推荐策略：

## 自己的学习 / 实验

```text
mamba
```

例如：

```bash
mamba create -n cs336 python=3.13
mamba activate cs336
```

## Stanford 官方 Assignment

如果官方仓库包含：

```text
pyproject.toml
uv.lock
```

并且测试环境明确使用：

```bash
uv run pytest
```

则：

> 为了 reproducibility，作业目录优先遵守官方 lockfile。

这并不意味着日常环境管理要全面切换到 uv。

原则：

```text
个人环境       mamba
课程复现       official lockfile
```

---

# 12. VS Code 开发工作流

Windows 安装：

```text
VS Code
WSL Extension
```

不要在 Ubuntu 里另外安装完整 VS Code GUI。

进入项目：

```bash
cd ~/projects
code .
```

VS Code 左下角确认：

```text
WSL: Ubuntu-24.04
```

此时：

```text
VS Code UI         Windows
Terminal           Ubuntu
Python             Ubuntu
Git                Ubuntu
Project filesystem Ubuntu ext4
GPU                Windows Driver → WSL
```

---

# 13. Phase 0：Week 1–2 PyTorch Bootcamp

你已经了解 Python 简单语法，但不能独立实现 Transformer。

由于数学背景是计算数学研究生，不需要重复花大量时间学习线性代数和微积分。

重点补：

> Python engineering + PyTorch tensor programming。

---

## Week 1：Tensor Thinking

### Lesson 0

环境：

- WSL2
- Linux basics
- Git
- mamba
- PyTorch
- CUDA
- VS Code

### Lesson 1：Tensor fundamentals

掌握：

```python
shape
dtype
device
stride
contiguous
reshape
view
transpose
permute
```

要求：

看到 tensor 操作先写 shape，再写代码。

---

### Lesson 2：Broadcasting

例如：

```python
x.shape == (B, T, D)
bias.shape == (D,)
```

理解：

```python
x + bias
```

为什么合法。

---

### Lesson 3：Matrix Multiplication

要求看到：

```python
x @ W
```

立刻判断：

```text
(B, T, D) @ (D, H)
→
(B, T, H)
```

---

### Lesson 4：einsum

目标不是炫技，而是形成 index notation ↔ tensor code 的映射。

例如：

$$
y_{btd} = \sum_h x_{bth}W_{hd}
$$

对应：

```python
torch.einsum("bth,hd->btd", x, W)
```

---

### Lesson 5：Autograd

理解：

```text
forward graph
↓
loss
↓
backward
↓
.grad
```

至少自己验证一个简单函数的 analytic gradient 与 autograd。

---

## Week 2：PyTorch NN Fundamentals

### Lesson 6：nn.Module

理解：

```python
nn.Module
nn.Parameter
state_dict
train()
eval()
```

---

### Lesson 7：Linear from scratch

自己写：

```python
class Linear(nn.Module):
    ...
```

与：

```python
nn.Linear
```

做 numerical comparison。

---

### Lesson 8：Embedding

理解：

```text
token id
↓
row lookup
↓
embedding vector
```

以及：

```text
Embedding != Linear(one_hot)
```

虽然数学上可等价，但实现和性能不同。

---

### Lesson 9：Softmax + Cross Entropy

重点：

- log-sum-exp trick
- numerical stability
- vocabulary dimension
- next-token prediction

---

### Lesson 10：SGD / Adam

自己理解：

$$
m_t
$$

$$
v_t
$$

$$
\theta_{t+1}
$$

再进入 AdamW。

---

### Lesson 11：Training Loop

完整流程：

```text
batch
↓
forward
↓
loss
↓
zero_grad
↓
backward
↓
optimizer.step
↓
metrics
```

---

### Lesson 12：Mini Project

训练一个很小的 neural network。

目的不是模型效果，而是熟悉：

```text
dataset
model
loss
optimizer
training loop
checkpoint
test
```

---

# 14. Phase 1：Weeks 3–7 — Assignment 1 Basics

这是整个 CS336 最关键的阶段。

目标：

> 从零实现并训练一个小型 Transformer Language Model。

---

## Week 3：Tokenizer

学习链：

```text
Unicode
↓
UTF-8
↓
bytes
↓
BPE
↓
token ids
```

实现：

```python
train_bpe(...)
encode(...)
decode(...)
```

测试核心：

```python
assert tokenizer.decode(tokenizer.encode(text)) == text
```

重点问题：

- 为什么使用 byte-level representation？
- Unicode character 与 byte 有什么区别？
- vocab size 如何影响 sequence length？
- tokenizer 如何改变训练计算量？
- merge algorithm 的主要复杂度在哪里？

GitHub 笔记：

```text
notes/01-tokenization.md
```

---

# 15. Week 4：Neural Network Primitives

实现：

```text
Linear
Embedding
RMSNorm
SwiGLU
Softmax
Cross Entropy
```

每个模块遵循：

```text
数学公式
↓
shape
↓
naive implementation
↓
PyTorch reference
↓
unit test
```

例如 RMSNorm：

$$
\operatorname{RMSNorm}(x)
=
\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}
\odot g
$$

要求：

- 理解 normalization axis；
- 理解 epsilon；
- 理解 fp16/bf16 下的 numerical stability；
- 对比 LayerNorm。

---

# 16. Week 5：Attention

核心公式：

$$
Q=XW_Q,\quad
K=XW_K,\quad
V=XW_V
$$

$$
A=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
+
M
\right)
$$

$$
O=AV
$$

实现顺序：

```text
single-head
↓
multi-head
↓
causal mask
↓
RoPE
↓
attention module
```

核心 shape：

```text
X
(B,T,D)

Q,K,V
(B,T,D)

reshape
(B,T,H,Dh)

transpose
(B,H,T,Dh)

QK^T
(B,H,T,T)

softmax
(B,H,T,T)

attention @ V
(B,H,T,Dh)
```

面试目标：

> 不运行代码，能够解释每一步 shape。

---

# 17. Week 6：Transformer + AdamW

组合：

```text
Embedding
↓
Transformer Block × L
↓
RMSNorm
↓
LM Head
```

Block：

```text
x
│
├─ RMSNorm
│    ↓
│  Attention
│    ↓
└─ residual
     │
     ├─ RMSNorm
     │    ↓
     │  SwiGLU
     │    ↓
     └─ residual
```

实现：

```python
class TransformerLM(nn.Module):
    ...
```

优化器：

```python
class AdamW:
    ...
```

学习：

- bias correction
- exponential moving averages
- weight decay
- AdamW decoupled weight decay

---

# 18. Week 7：TinyStories Training

建立完整训练流程：

```text
tokenized dataset
↓
batch sampling
↓
forward
↓
cross entropy
↓
backward
↓
AdamW
↓
LR schedule
↓
checkpoint
↓
sampling
```

记录：

- train loss
- validation loss
- tokens/sec
- GPU memory
- generated samples

GitHub 第一阶段成果：

```text
CS336 A1 — Language Model From Scratch

✓ BPE tokenizer
✓ Transformer
✓ RoPE
✓ RMSNorm
✓ SwiGLU
✓ AdamW
✓ LR scheduler
✓ training loop
✓ sampling
✓ unit tests
```

---

# 19. Phase 2：Weeks 8–11 — Assignment 2 Systems

目标：

> 不只知道模型“能跑”，而是知道为什么快 / 为什么慢。

---

## Week 8：GPU Performance Fundamentals

理解：

```text
FLOPs
memory traffic
memory bandwidth
arithmetic intensity
roofline model
kernel launch overhead
HBM
SRAM / shared memory
register
```

每个 operation 问：

```text
多少 FLOPs？
读取多少 bytes？
写多少 bytes？
compute-bound？
memory-bound？
```

学习 benchmark 正确姿势：

```text
warmup
synchronize
multiple iterations
median / quantile
fixed shape
fixed dtype
```

---

# 20. Week 9：Profiling + torch.compile

学习：

```text
PyTorch profiler
CUDA events
kernel timeline
memory allocation
operator breakdown
```

比较：

```text
PyTorch eager
torch.compile
optimized primitive
```

原则：

> 没有 benchmark 的“优化”不算优化。

---

# 21. Week 10：Triton

推荐顺序：

```text
Vector Add
↓
Fused Softmax
↓
Matrix Multiplication
↓
RMSNorm
↓
FlashAttention
```

理解：

```text
program id
block
mask
load
store
tiling
fusion
```

Benchmark：

```text
PyTorch eager
PyTorch compiled
Triton implementation
PyTorch optimized primitive
```

指标：

```text
latency
bandwidth
TFLOP/s
shape scaling
```

RTX 3050 Ti 主要负责：

```text
correctness
small kernels
small benchmark
learning
```

不要追求 B200 leaderboard 数字。

---

# 22. Week 11：Distributed Training

学习顺序：

```text
single GPU
↓
data parallel
↓
gradient all-reduce
↓
DDP
↓
communication bucket
↓
communication overlap
↓
optimizer state sharding
↓
FSDP idea
```

必须能够估算：

```text
parameters
gradients
optimizer states
activations
```

分别占多少显存。

本地没有多 GPU 时：

- 完成算法与 API 理解；
- 能进行的测试本地完成；
- 必须多 GPU 的实验使用云 GPU；
- 控制实验规模。

---

# 23. Phase 3：Weeks 12–13 — Scaling Laws

这部分非常适合计算数学背景。

核心：

$$
C \approx 6ND
$$

其中：

```text
C = training compute
N = number of parameters
D = number of training tokens
```

---

## Week 12

设计小规模 experiment grid：

```text
model sizes:
N1, N2, N3, ...

token budgets:
D1, D2, D3, ...

↓
validation loss
```

学习：

- power law
- log-log plot
- curve fitting
- IsoFLOP
- compute frontier

---

## Week 13

拟合：

```text
compute
↓
optimal model size
optimal token count
↓
prediction
```

Portfolio 中展示：

- experiment table
- fitted curve
- residual
- prediction
- failure cases

这可以很好地连接：

```text
计算数学
+
LLM training
```

---

# 24. Phase 4：Weeks 14–16 — Data

目标：

> 理解 LLM 的 performance 不只是 architecture 问题，也是 data engineering 问题。

---

## Week 14：Extraction + Filtering

流程：

```text
Common Crawl
↓
text extraction
↓
language filtering
↓
quality filtering
```

分析：

- text quality
- language probability
- document length
- weird symbol ratio
- repetition

---

# 25. Week 15：Deduplication

学习：

```text
exact dedup
near dedup
MinHash
LSH
```

理解：

- Jaccard similarity
- hash collision
- precision / recall tradeoff
- contamination

---

# 26. Week 16：Train on Different Data Mixtures

比较：

```text
raw data
vs
filtered data
vs
filtered + deduplicated
```

尽量保持：

```text
same model
same token budget
same optimizer
```

然后比较：

```text
validation loss
perplexity
training stability
sample quality
```

核心不是训练最大模型，而是完成可信的 controlled experiment。

---

# 27. Phase 5：Weeks 17–19 — Alignment

学习顺序：

```text
Base LM
↓
SFT
↓
Preference Optimization
↓
DPO
↓
RL
↓
GRPO
```

---

## Week 17：SFT

理解：

$$
\mathcal{L}_{SFT}
=
-\sum_t \log p_\theta(y_t|x,y_{<t})
$$

重点：

- instruction dataset
- prompt / completion masking
- teacher forcing
- response loss

---

# 28. Week 18：DPO

理解 preference pair：

```text
prompt
├── chosen
└── rejected
```

学习：

- reference model
- preference ratio
- KL interpretation
- 为什么 DPO 不显式训练 reward model

面试问题：

```text
DPO 与 RLHF/PPO 最大区别是什么？
reference model 有什么作用？
beta 控制什么？
```

---

# 29. Week 19：GRPO / Reasoning

做一个小型 reasoning experiment。

例如：

```text
GSM8K subset
```

比较：

```text
base model
vs
SFT
vs
GRPO
```

记录：

```text
accuracy
reward
response length
KL
training stability
```

理解：

```text
on-policy
advantage
group-relative normalization
reward hacking
KL regularization
```

---

# 30. Phase 6：Week 20 — Portfolio

最终建议至少维护：

```text
cs336-from-scratch
```

以及：

```text
<username>.github.io
```

---

# 31. 主项目结构

最终整理成：

```text
cs336-from-scratch/
├── src/
│   └── mini_llm/
│       ├── tokenizer/
│       ├── nn/
│       ├── model/
│       ├── optim/
│       ├── training/
│       ├── kernels/
│       ├── distributed/
│       ├── data/
│       ├── scaling/
│       ├── eval/
│       └── alignment/
│
├── assignments/
├── tests/
├── benchmarks/
├── configs/
├── experiments/
├── scripts/
├── notes/
├── assets/
├── README.md
└── pyproject.toml
```

不一定第一天就创建完整目录。

目录应该随着课程逐步生长。

---

# 32. Jekyll GitHub Pages

Jekyll 首页不需要复制所有代码。

Portfolio 更适合展示：

```text
Stanford CS336
Language Modeling From Scratch

01 Tokenization
02 Transformer
03 Training
04 GPU Systems
05 Scaling Laws
06 Data
07 Alignment
```

每个 section：

```text
What I implemented
What I learned
Key equations
Experiment results
Benchmark
Interesting failure
Code →
Detailed Notes →
```

等 A1 有真正成果以后再增加项目页面。

避免首页长期出现：

```text
Coming soon
TODO
Work in progress
```

---

# 33. 每个知识单元统一学习模板

以后课程笔记固定采用：

## 1. Why

为什么需要这个技术？

## 2. Math

公式和推导。

## 3. Shapes

所有重要 tensor shape。

## 4. Naive implementation

最容易理解的手写版本。

## 5. Reference implementation

与 PyTorch reference / optimized primitive 对比。

## 6. Unit tests

正确性测试。

## 7. Numerical issues

数值稳定性。

## 8. Performance

时间 / 显存 / kernel 行为。

## 9. CS336 mapping

对应 Lecture / Assignment。

## 10. Interview questions

算法岗面试问题。

## 11. GitHub notes

整理成适合公开展示的内容。

---

# 34. 代码规范

项目代码统一遵循：

```text
modern Python type hints
small composable modules
pytest
Ruff
explicit tensor shapes
device agnostic
dtype aware
no hidden global state
deterministic unit tests when possible
reference implementation first
benchmark before optimization
```

Tensor-heavy API 可以使用：

```python
jaxtyping
```

帮助表达：

```python
Float[Tensor, "batch seq d_model"]
```

---

# 35. “自己实现”与“现代工程实现”分开

学习阶段：

```text
自己实现
softmax
attention
RMSNorm
AdamW
```

目的是理解。

现代工程阶段则优先研究：

```text
PyTorch SDPA
torch.compile
FlexAttention
Triton
optimized distributed primitives
```

原则：

```text
手写 primitive
    ↓
unit test
    ↓
与 reference implementation 对齐
    ↓
profile
    ↓
optimized implementation
```

不要一开始就调用最高层 API。

否则很容易变成：

```text
会用模型
≠
理解模型
```

---

# 36. RTX 3050 Ti 的定位

本地 GPU 主要用于：

```text
PyTorch fundamentals
Transformer correctness
TinyStories small runs
small-model training
Triton learning
profiling
unit tests
debug
```

不适合强行承担：

```text
large-scale pretraining
large Common Crawl experiments
multi-GPU distributed experiments
Stanford B200 leaderboard
```

正常策略：

```text
local development
↓
unit test
↓
small experiment
↓
Git commit
↓
cloud GPU when necessary
↓
download metrics / plots / checkpoint
↓
local analysis
```

---

# 37. 面试路线同步进行

从 Week 3 开始，每周增加一组面试问题。

## Transformer

- Self-Attention 时间复杂度？
- 为什么除以 \(\sqrt{d_k}\)？
- causal mask 在哪里加？
- MHA 与 GQA 区别？
- RoPE 如何编码 relative position？
- RMSNorm 与 LayerNorm 区别？
- SwiGLU 为什么常见？

## Training

- Adam 与 AdamW 区别？
- gradient clipping 为什么需要？
- warmup 为什么有效？
- training loss 与 validation loss 分别说明什么？

## Systems

- FlashAttention 为什么减少 HBM traffic？
- arithmetic intensity 是什么？
- compute-bound / memory-bound 如何判断？
- DDP 的 communication cost？
- ZeRO/FSDP shard 什么？

## Scaling

- Chinchilla-style compute optimality 是什么？
- 为什么 parameter count 不是唯一 scaling variable？

## Alignment

- SFT / RLHF / DPO / GRPO 区别？
- on-policy / off-policy？
- KL penalty？
- reward hacking？

---

# 38. 每周完成标准

不要用：

```text
“视频看完了”
```

作为完成标准。

使用：

```text
[ ] 核心公式能解释
[ ] tensor shapes 能手算
[ ] 核心代码自己写过
[ ] pytest 通过
[ ] 至少一个 experiment
[ ] 记录失败案例
[ ] 整理 GitHub note
[ ] 能回答 3–5 个面试问题
```

---

# 39. Git 工作流

建议从一开始形成：

```bash
git status
git add
git commit
git push
```

commit 不要写：

```text
update
fix
test
```

尽量写：

```text
Implement byte-level BPE merge loop
Add RMSNorm reference tests
Benchmark causal attention on RTX 3050 Ti
Fix RoPE broadcasting bug
```

这样半年后 Git history 本身就是学习轨迹。

---

# 40. Backup

由于整个 Linux distribution 在一个 VHD 中，建议阶段性备份。

例如完成 A1 后：

```powershell
wsl --shutdown
wsl --export Ubuntu-24.04 D:\WSL\backups\ubuntu-cs336-a1.tar
```

完成 A2 后再做一次。

Git repo 仍然要 push 到 GitHub。

原则：

```text
Git
=
source / notes / configs

WSL backup
=
environment / Linux state

重要 checkpoint
=
单独备份
```

不要只依赖其中一种。

---

# 41. 前两周立即执行计划

## Week 1

### Day 1

```text
WSL2 安装到非 C 盘
Ubuntu 初始化
Linux 基础命令
GPU nvidia-smi
```

### Day 2

```text
Miniforge / mamba
VS Code WSL
Git
PyTorch CUDA
```

### Day 3

```text
Tensor shape
dtype
device
reshape
transpose
```

### Day 4

```text
broadcasting
```

### Day 5

```text
matrix multiplication
einsum
```

### Weekend

```text
autograd
小练习
GitHub note
```

---

## Week 2

```text
nn.Module
Linear
Embedding
Softmax
Cross Entropy
SGD
Adam
Training Loop
Mini Project
```

完成后正式进入：

```text
CS336 Lecture 1
+
Assignment 1
+
BPE Tokenizer
```

---

# 42. 第一阶段里程碑

当以下内容完成时，说明你已经可以正式进入 CS336：

```text
[ ] WSL2 Ubuntu 正常
[ ] Ubuntu 数据位于非 C 盘
[ ] nvidia-smi 能识别 RTX 3050 Ti
[ ] mamba 正常
[ ] PyTorch CUDA available
[ ] VS Code WSL 正常
[ ] Git 正常
[ ] 能熟练 cd / ls / pwd / mkdir / cp / mv / rm
[ ] 理解 tensor shape
[ ] 理解 broadcasting
[ ] 能使用 matmul / einsum
[ ] 理解 autograd
[ ] 能写 nn.Module
[ ] 能写最简单的 training loop
```

---

# 43. 推荐资料优先级

以后资料搜索顺序：

```text
1. Stanford CS336 2026 官方课程
2. Stanford CS336 官方 lecture repository
3. Stanford CS336 官方 assignment repository
4. PyTorch 官方文档
5. Triton 官方文档
6. 原论文
7. 高质量第三方笔记
8. 其他教程
```

不要优先搜索 assignment solution。

使用第三方 solution 的原则：

```text
先独立实现
↓
测试失败
↓
定位问题
↓
仍然无法解决
↓
再参考思路
```

而不是直接复制。

---

# 44. 推荐的长期学习原则

### Principle 1

```text
Correctness before performance
```

### Principle 2

```text
Shape before code
```

### Principle 3

```text
Formula before API
```

### Principle 4

```text
Benchmark before optimization
```

### Principle 5

```text
Experiment before conclusion
```

### Principle 6

```text
Explain before memorizing
```

### Principle 7

```text
Small reproducible experiment
>
large uncontrolled experiment
```

---

# 45. 最终完成状态

20 周后希望你的 GitHub 可以明确证明：

```text
I can implement a Transformer.

I understand the tensor shapes and mathematics.

I can train it.

I can profile it.

I can write GPU kernels.

I understand distributed training.

I can run scaling experiments.

I can build a pretraining data pipeline.

I understand modern post-training.

I can explain all of these in an interview.
```

这才是完成 CS336 的真正标准。

---

# 46. 官方参考资料

建议始终以最新官方资料为准：

- Stanford CS336: <https://cs336.stanford.edu/>
- Stanford CS336 GitHub: <https://github.com/stanford-cs336>
- Microsoft WSL installation: <https://learn.microsoft.com/windows/wsl/install>
- Microsoft WSL basic commands: <https://learn.microsoft.com/windows/wsl/basic-commands>
- Microsoft WSL configuration: <https://learn.microsoft.com/windows/wsl/wsl-config>
- Microsoft WSL disk space: <https://learn.microsoft.com/windows/wsl/disk-space>
- NVIDIA CUDA on WSL: <https://docs.nvidia.com/cuda/wsl-user-guide/>
- PyTorch: <https://pytorch.org/>
- Triton: <https://triton-lang.org/>
- Mamba: <https://mamba.readthedocs.io/>

---

## 当前下一步

先不要急着安装 mamba 或 PyTorch。

第一步只完成：

```powershell
wsl --version
wsl --status
wsl --list --online
```

然后把 Ubuntu 安装到目标非 C 盘：

```powershell
wsl --install -d Ubuntu-24.04 --location D:\WSL\Ubuntu-24.04
```

实际盘符不是 D: 时替换成你的目标盘符。

安装完成后：

```bash
pwd
uname -a
nvidia-smi
```

之后再继续配置：

```text
Miniforge / mamba
→ PyTorch CUDA
→ VS Code WSL
→ Git
→ PyTorch Bootcamp
```
