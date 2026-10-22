#!/usr/bin/env python
# @FileName  :pretokenization.py
# @Time      :2026/10/19
# @Author    :Xiaoming
"""GPT-2 风格的预分词（pre-tokenization）、special token 隔离，以及可流式统计的安全分块。

整体数据流：

    raw text
      -> 先按 special token 切开（special token 必须保持 atomic）
      -> 每个 ordinary span 用 GPT-2 regex 做 pre-tokenization
      -> 每个 pre-token 转成 UTF-8 bytes，再把每个 byte 当作一个 BPE symbol
      -> Counter[Word] 统计频次

本模块维护两个核心不变量：

1. 对 ordinary 文本满足 ``"".join(pretokens) == 原文``（不丢字、不重复）；
2. 对文件分块统计时，各 chunk 的 Counter 累加 == 整篇文本一次性统计的结果。

注意：这里统计的是 **pre-token frequency**，还不是 BPE 的 pair frequency。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path

import regex  # 使用 regex 而不是标准库 re：需要 \p{L} / \p{N} 这类 Unicode property

GPT2_PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)"""  # 1) 英语缩写：'s / 'd / 'm / 't / 'll / 've / 're
    r"""| ?\p{L}+"""  # 2) 可选前导空格 + 连续字母（Unicode-aware，支持中文等）
    r"""| ?\p{N}+"""  # 3) 可选前导空格 + 连续数字
    r"""| ?[^\s\p{L}\p{N}]+"""  # 4) 可选前导空格 + 标点/符号/emoji 等
    r"""|\s+(?!\S)"""  # 5) 一段空白，且其后不是非空白字符
    r"""|\s+"""  # 6) 兜底：匹配剩余的任意空白
)

GPT2_PATTERN = regex.compile(GPT2_PRETOKEN_PATTERN)

Word = tuple[bytes, ...]


def _special_pattern(special_tokens: Sequence[str]) -> regex.Pattern | None:
    """把 special token 列表编译成“最长匹配优先”的正则。

    Args:
        special_tokens: 需要保持 atomic 的 special token 字符串列表。

    Returns:
        编译好的正则；若列表为空则返回 ``None``（调用方可以跳过 split）。

    Raises:
        ValueError: 列表中存在空字符串。
    """
    # 先 set 去重，再按“长度降序、字典序升序”排序：
    #   - 长度降序：保证 overlapping 时优先匹配更长的 token。
    #   - 字典序升序：长度相同时保证结果 deterministic，不依赖 set 的迭代顺序。
    tokens = sorted(set(special_tokens), key=lambda s: (-len(s), s))

    if not tokens:
        return None

    # 空字符串会变成零宽匹配，导致 split() 在任意位置切开，必须拒绝。
    if any(not s for s in tokens):
        raise ValueError("A special token cannot be empty")

    # regex.escape 让 token 里的 | . < > 等字符按字面量匹配，而不是被当作正则语法。
    return regex.compile("|".join(regex.escape(s) for s in tokens))


def iter_pretokens(text: str, special_tokens: Sequence[str] = ()) -> Iterator[str]:
    """逐个产出 ordinary pre-token；special token 只作为边界，不会出现在结果里。

    处理顺序非常重要（不能反过来）：

        先隔离 special token  ->  再对普通片段跑 GPT-2 regex

    如果先跑 GPT-2 regex，special token 可能被拆成 "<|"、"endoftext"、"|>" 等碎片，
    后续 BPE 训练就会把 special token 的内部结构学进 merges，破坏它的 atomic 性质。

    Args:
        text: 输入文本。
        special_tokens: 需要先隔离的 special token 列表。

    Yields:
        ordinary pre-token 字符串，顺序与原文一致。
    """
    splitter = _special_pattern(special_tokens)

    # 没有 special token 时，整段文本就是一个 ordinary span；
    # 有 special token 时，split 会把它们当作分隔符移除，只保留普通片段。
    pieces = (text,) if splitter is None else splitter.split(text)

    for piece in pieces:
        # GPT-2 regex 的匹配结果互不重叠，拼起来恰好等于 piece，
        # 这就是 pre-tokenization 这一层的 round-trip 不变量。
        for match in GPT2_PATTERN.finditer(piece):
            yield match.group()


def iter_stable_chunks(
    path: str | Path,
    special_tokens: Sequence[str] = (),
    chunk_chars: int = 1 << 20,
) -> Iterator[str]:
    """按“完整的 pre-token 边界”流式切分文件，产出安全的文本 chunk。

    目标：让各 chunk 上的统计结果可以直接累加，即

        count(whole) == sum(count(chunk_i))

    为什么不能简单地按固定字符数切？
        - 可能切断一个 pre-token，例如把 " wonderful" 切成 " won" + "derful"；
        - 可能切断 special token，破坏它的 atomic 性质；
        - GPT-2 regex 含有 lookahead（``\\s+(?!\\S)``），cut 附近的匹配
          会依赖右侧上下文，同一段文本在不同切分下可能得到不同结果；
        - 若按字节切分，还可能切断一个多字节 UTF-8 字符。

    Args:
        path: 输入文件路径，按 UTF-8 文本读取。
        special_tokens: 需要保持 atomic 的 special token 列表。
        chunk_chars: 单次 read 的字符数（不是字节数），同时近似作为保留的右上下文大小。

    Yields:
        若干连续文本片段；把它们按顺序拼接起来等于原文件的文本内容。

    Raises:
        ValueError: chunk_chars 太小。
    """
    if chunk_chars < 256:
        raise ValueError("chunk_chars must be >= 256")

    splitter = _special_pattern(special_tokens)
    longest_special = max((len(s) for s in special_tokens), default=0)

    with open(path, "r", encoding="utf-8", newline="") as handle:
        buffer = ""

        while True:
            piece = handle.read(chunk_chars)
            if not piece:
                if buffer:
                    yield buffer
                break

            buffer += piece

            # buffer 还不够长时继续读，保证 cut 右侧始终有至少 chunk_chars 的上下文。
            if len(buffer) < 2 * chunk_chars:
                continue

            # 期望切在 len(buffer) - chunk_chars 附近，
            # 同时保留 chunk_chars 个字符作为右上下文。
            target = len(buffer) - chunk_chars

            # 在 buffer 里找“结束位置 <= target”的最后一个完整 GPT-2 pre-token。
            #
            # finditer 按出现顺序返回互不重叠的匹配，因此直接迭代到第一个
            # end() > target 的匹配为止，此时 safe_end 就是最后一个合法切点。
            safe_end = 0
            for match in GPT2_PATTERN.finditer(buffer):
                if match.end() > target:
                    break
                safe_end = match.end()

            # target 之前一个完整 pre-token 都没有（例如遇到超长 token）：
            # 继续读更多数据，直到能找到安全切点。
            if not safe_end:
                continue

            if splitter is not None:
                # 安全检查：如果 safe_end 落在某个 special token 内部，
                # 就把切点往前移动到该 special token 的起点，避免把它切成两半。
                #
                # 只在 safe_end ± longest_special 的窗口里搜索即可：
                # 任何覆盖 safe_end 的 special token 起点都不会早于 lo，
                # 终点也不会晚于 hi。
                lo = max(0, safe_end - longest_special)
                hi = min(len(buffer), safe_end + longest_special)

                for match in splitter.finditer(buffer[lo:hi]):
                    # match 的偏移是相对于 buffer[lo:hi] 的，需要加回 lo。
                    a, b = lo + match.start(), lo + match.end()
                    if a < safe_end < b:
                        # safe_end 严格落在 [a, b) 内部 -> 退到 token 起点 a。
                        safe_end = a
                        break

            if safe_end > 0:
                # 产出 [0, safe_end)，剩余部分留在 buffer 里继续参与下一次匹配。
                yield buffer[:safe_end]
                buffer = buffer[safe_end:]


def count_pretokens(
    path: str | Path,
    special_tokens: Sequence[str] = (),
    chunk_chars: int = 1 << 20,
) -> Counter[Word]:
    """统计整个文件的 pre-token 频次，返回 ``Counter[Word]``。

    返回的 ``Word = tuple[bytes, ...]``，其中每个元素是单个字节。例如：

        " cat" -> (b" ", b"c", b"a", b"t")

    这样后续 BPE 就能直接在最相邻的两个 symbol 上做 merge。

    Args:
        path: 输入文件路径。
        special_tokens: 需要隔离的 special token 列表。
        chunk_chars: 传给 ``iter_stable_chunks`` 的分块大小。

    Returns:
        pre-token（byte symbol tuple）到频次的映射。
    """
    counts: Counter[Word] = Counter()

    # 逐块读取并统计，最后把各块结果累加到同一个 Counter。
    # 由于 iter_stable_chunks 保证块边界对齐到完整 pre-token，
    # 这种分块统计与“一次性读入整个文件再统计”的结果完全一致。
    for chunk in iter_stable_chunks(path, special_tokens, chunk_chars):
        for pretoken in iter_pretokens(chunk, special_tokens):
            # 一个 pre-token 的 UTF-8 编码可能包含多个字节
            # （中文通常 3 个，emoji 通常 4 个）。
            # 这里把每个 byte 单独包成 bytes，作为 BPE 的初始 symbol。
            counts[tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))] += 1

    return counts
