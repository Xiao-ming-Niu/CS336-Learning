# AGENTS.md — CS336-Learning 仓库约定

## 推送流程（必须遵守）

每次执行 `git push` 到远程仓库前，**必须同步更新 `README.md` 和 `README-en.md`**：

1. 先确认本次改动新增 / 删除 / 重命名了哪些 notebook、目录或配置文件（`git status`）；
2. 在 `README.md` 与 `README-en.md` 中同步更新：
   - 顶部「当前进度」/「Current progress」；
   - 「仓库结构」/「Repository Layout」；
   - 「笔记索引」/「Notes Index」中对应的 Lesson 表格（新增行、修正描述、去掉已完成的 🚧）；
   - 末尾「Roadmap 进度」/「Roadmap Progress」的勾选状态；
3. 两个 README 必须保持**结构一致、内容对应**：同样的段落、同样的顺序、同样的链接；
4. README 的更新与代码 / 笔记改动**放进同一个 commit**，一起 push。

例外：若本次改动确实无需更新 README（例如仅修 typo、仅调整 notebook 的 `kernelspec` 元数据），必须在回复中说明理由。

## 仓库信息

- 远程：`origin` = `git@github.com:Xiao-ming-Niu/CS336-Learning.git`（SSH，保持 SSH 不要改回 HTTPS）
- 主分支：`main`
- 笔记目录：`notes/weekNN_*/`，文件名形如 `NN_topic.ipynb`；Lesson 编号是贯穿全仓库的连续序列
- 笔记正文以中文为主，保留英文术语、公式与 API 名称
- `cs336_learning_roadmap.md` 是 20 周计划的事实来源：判断某个 notebook 属于第几周时以它为准（如 Week 4 = Linear / Embedding / RMSNorm / SwiGLU / Softmax / Cross Entropy）

## Commit message

英文、祈使句、说明「做了什么」：

```text
Implement byte-level BPE merge loop
Add RMSNorm reference tests
Expand BPE correctness notebook and add tokenizer inference notes
```

避免 `update` / `fix` / `test` 这类无信息量的 message。

## 提交前的安全检查

- `.gitignore` 已排除 `*.pt`、`*.pth`、`*.ckpt`、`checkpoints/`、`datasets/`、`.venv/`、`__pycache__/`、`.ipynb_checkpoints/`，不要把训练产物提交进来。
- 若某个 notebook 的 cell 数或源码字符数**骤减**（例如 25 cells → 1 cell），视为疑似误覆盖，先向用户确认再提交。
