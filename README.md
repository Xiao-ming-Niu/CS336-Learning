# CS336 Learning

自学 Stanford CS336（Language Modeling from Scratch）的学习笔记与代码仓库。

## 目录结构

```
cs336_learning_roadmap.md   # 完整自学路线图（约 20 周计划、能力目标、每周任务）
notes/
  week01_Tensor_thinking/   # 张量基础、广播、矩阵乘法、einsum、autograd
  week02_Pytorch_nn/        # nn.Module、Linear、Embedding、Softmax/CrossEntropy、优化器、训练循环
  week03_Tokenizer/         # Tokenizer 基础、Toy BPE Trainer、Pre-Tokenization
```

## 学习主线

```
PyTorch Fundamentals → Tensor Shape Reasoning → BPE Tokenizer
→ Transformer From Scratch → LM Training → GPU / Profiling / Triton
→ Distributed Training → Scaling Laws → Data Pipeline
→ SFT / DPO / GRPO → Mini LLM Training Stack
```

## 环境

- Python + PyTorch（笔记以 Notebook 形式记录，可直接运行验证）
- 建议使用 `mamba` / `uv` 管理环境

## 说明

本仓库为个人学习沉淀，笔记以中文为主，保留英文术语与 API 名称。
