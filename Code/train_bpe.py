#!/usr/bin/env python
# @FileName  :train_bpe.py
# @Time      :2026/10/19
# @Author    :Xiaoming
"""Byte-level BPE 训练：带倒排索引的、增量式 pair frequency 更新。

与最朴素的 reference trainer 相比，这里的核心优化有两点：

1. **倒排索引（inverted index）**
   维护 ``pair -> {包含该 pair 的 word}``，每次 merge 只访问真正受影响的 word，
   而不是重新扫描整个 corpus。

2. **增量更新（incremental update）**
   维护全局 ``pair_freq``。一个 word 被 merge 时，只减去它旧的 pair 贡献、
   加上它新的 pair 贡献，而不是每轮重算所有 pair 的频次。

另外用 **heap** 做 winner selection：每轮取频次最高的 pair 从 O(P) 扫描
降到 O(log P)（P 为当前不同 pair 的数量）；堆里采用 lazy deletion，
过期条目在 pop 时跳过即可，不需要实现 decrease-key。

对外契约（与 CS336 Assignment 1 一致）：
    vocab:  dict[int, bytes]
            0..255 是单字节 token，然后是 special tokens，最后是 merge 产生的 token
    merges: list[tuple[bytes, bytes]]
            按学习顺序排列，顺序即 encode 时使用的 merge rank

注意：special token 计入 ``vocab_size``，但它们不会参与 BPE merge。
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .pretokenization import Word, count_pretokens

# 一个 BPE pair：两个 token（每个 token 都是一段 bytes）组成的相邻对。
# 例如 (b"t", b"h") 合并后得到新 token b"th"。
Pair = tuple[bytes, bytes]


@dataclass(frozen=True)
class _ReverseLex:
    """堆里的 tie-breaker：让字典序**更大**的 pair 先被弹出。

    Python 的 ``heapq`` 是 min-heap，而 CS336 要求：
        频次相同时，选择 lexicographically greater 的 pair。

    因此这里把比较方向反过来：``self.pair > other.pair`` 时认为 self "更小"，
    于是字典序更大的 pair 会排在堆顶。

    堆条目整体是 ``(-count, _ReverseLex(pair))``：
        - ``-count`` 越小  <=>  count 越大  -> 高频 pair 优先；
        - count 相同时比较 ``_ReverseLex`` -> 字典序更大的 pair 优先。
    """

    pair: Pair

    def __lt__(self, other: _ReverseLex) -> bool:
        # 反向比较：字典序更大的 pair 被认为是 "更小"，从而先出堆。
        return self.pair > other.pair


def _word_pairs(word: Word) -> Counter[Pair]:
    """统计一个 word 内部所有相邻 pair 的**局部出现次数**（含重叠）。

    例如：
        (a, a, a)  ->  {(a, a): 2}     # 两个重叠的 (a, a)

    返回的是 multiplicity（在单个 word 内出现几次），
    乘上该 word 的 corpus frequency 才是它对全局 pair frequency 的贡献。
    """
    return Counter(zip(word, word[1:]))


def _merge_word(word: Word, pair: Pair) -> Word:
    """在单个 word 内应用一次 merge，采用**从左到右、不重叠**的规则。

    例如：
        word = (a, a, a, a), pair = (a, a)
        -> (b"aa", b"aa")      # 不是 (aa, a, a) 或 (a, aa, a)

    这与 reference 实现保持一致：
        - 匹配到 pair 时，输出两个 token 的 bytes 拼接，并前进 2 步；
        - 否则原样复制当前 token，前进 1 步。
    """
    result: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            # 命中 pair：新 token = 两个 bytes 直接拼接。
            result.append(word[i] + word[i + 1])
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs: object,
) -> tuple[dict[int, bytes], list[Pair]]:
    """在给定文件上训练 byte-level BPE。

    Args:
        input_path: 训练语料路径。
        vocab_size: **最终**词表大小，包含 256 个 byte、special tokens 和 merge token。
        special_tokens: 需要保持 atomic 的 special token 列表。
            它们直接进入 vocab，不参与 merge。
        **kwargs: 目前支持 ``chunk_chars``，透传给 ``count_pretokens`` 控制分块大小。

    Returns:
        ``(vocab, merges)``：
            - ``vocab[id] = bytes``
            - ``merges`` 按学习顺序排列，顺序即 encode 时的 merge rank。

    Raises:
        ValueError: ``vocab_size`` 小于 256 + special token 数量。
    """
    # 去重，同时保留用户传入的顺序（dict 从 Python 3.7 起保持插入顺序）。
    # 顺序会影响 special token 分到的 token id，因此这里不能简单用 set。
    specials = list(dict.fromkeys(special_tokens))

    # 最小合法词表：256 个单字节 token + 所有 special token。
    minimum = 256 + len(specials)
    if vocab_size < minimum:
        raise ValueError(f"vocab_size must be >= {minimum}")

    # ------------------------------------------------------------------
    # 1. 初始化 vocabulary
    # ------------------------------------------------------------------
    # 前 256 个 id 一一对应 0x00..0xFF 这 256 个单字节。
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # special token 紧接着 byte token 放入，并计入 vocab_size。
    # 它们不会进入后面的 pre-token Counter / pair 统计，因此永远不会被 merge。
    for token in specials:
        vocab[len(vocab)] = token.encode("utf-8")

    # ------------------------------------------------------------------
    # 2. 统计 pre-token frequency
    # ------------------------------------------------------------------
    # words: Word(tuple[bytes, ...]) -> corpus frequency
    # Word 的每个元素都是单字节 symbol；special token 已被当作边界隔离掉。
    words: Counter[Word] = count_pretokens(
        input_path,
        specials,
        chunk_chars=int(kwargs.get("chunk_chars", 1 << 20)),
    )

    # ------------------------------------------------------------------
    # 3. 构建初始的 pair_freq 和倒排索引 pair_words
    # ------------------------------------------------------------------
    # pair_freq: pair -> 全局加权频次
    #            = sum over words (word_freq * 该 pair 在 word 内的局部次数)
    # pair_words: pair -> 包含该 pair 的**不同 word** 的集合（倒排索引）
    pair_freq: Counter[Pair] = Counter()
    pair_words: dict[Pair, set[Word]] = defaultdict(set)

    for word, frequency in words.items():
        for pair, multiplicity in _word_pairs(word).items():
            # frequency 是该 word 在 corpus 中出现的次数；
            # multiplicity 是 pair 在这个 word 内出现了几次（含重叠）。
            pair_freq[pair] += frequency * multiplicity
            pair_words[pair].add(word)

    # ------------------------------------------------------------------
    # 4. 初始化 winner 选择用的 heap
    # ------------------------------------------------------------------
    # 堆条目为 (-count, _ReverseLex(pair))，这样：
    #   - count 越大，-count 越小，越先出堆；
    #   - count 相同时，字典序更大的 pair 越先出堆。
    #
    # 采用 lazy deletion：不维护 decrease-key，pair_freq 变化时直接 push 新条目；
    # 旧的过期条目留在堆里，pop 时通过“count 是否仍然一致”来跳过。
    heap: list[tuple[int, _ReverseLex]] = [
        (-count, _ReverseLex(pair)) for pair, count in pair_freq.items() if count > 0
    ]
    heapq.heapify(heap)

    merges: list[Pair] = []

    # ------------------------------------------------------------------
    # 5. 主循环：每次 merge 一个 pair，直到词表达到目标大小
    # ------------------------------------------------------------------
    while len(vocab) < vocab_size and heap:
        # ---- 5.1 从堆里选出一个仍然有效的最高频 pair ----
        selected: Pair | None = None
        while heap:
            negative_count, key = heapq.heappop(heap)
            pair = key.pair

            # 有效性判断：
            #   - 该条目的 count 必须等于 pair_freq 中的当前值（否则是过期条目）；
            #   - 当前频次必须 > 0（频次降到 0 的 pair 不应再被选中）。
            if -negative_count == pair_freq.get(pair, 0) and -negative_count > 0:
                selected = pair
                break

        if selected is None:
            # 堆已空：没有可 merge 的 pair 了（例如所有 word 都只剩单 token）。
            break

        # 取出所有包含该 pair 的 word。
        # 用 tuple 做快照：下面的循环会修改 pair_words，不能直接迭代 set。
        candidate_words = tuple(pair_words.get(selected, ()))
        if not candidate_words:
            # 理论上不该发生；做一次清理，避免死循环。
            pair_freq.pop(selected, None)
            continue

        # updates: merge 后的新 word -> 需要累加的 corpus frequency
        # changed_pairs: 频次发生变化的 pair，循环结束后统一刷新堆
        updates: Counter[Word] = Counter()
        changed_pairs: set[Pair] = set()

        # ---- 5.2 第一阶段：移除所有受影响的旧 word，并计算 merge 结果 ----
        #
        # 为什么先全部移除、再统一插入？
        #   因为 merge 可能让不同的 word “撞车”：
        #       (a, b, c) --merge(b,c)--> (a, bc)
        #       (a, bc)   --本来就已经存在--> 同一个 key
        #   如果边删边加，可能把已有频次覆盖掉或重复计数。
        #   先删后加可以保证：最终 words[merged] 是所有来源频次的正确总和。
        for word in candidate_words:
            count = words.pop(word, 0)
            if not count:
                continue

            # 减去这个旧 word 对全局 pair_freq 的贡献。
            for pair, multiplicity in _word_pairs(word).items():
                pair_freq[pair] -= count * multiplicity
                changed_pairs.add(pair)
                # 该 word 不再包含这个 pair，从倒排索引里移除。
                pair_words[pair].discard(word)

            # 记录 merge 后的新 word 及其频次（可能与其他来源合并）。
            updates[_merge_word(word, selected)] += count

        # ---- 5.3 第二阶段：把 merge 后的新 word 加回去 ----
        for word, count in updates.items():
            # 用 += 而不是 =：新 word 可能已经存在（撞车），需要累加。
            words[word] += count

            # 加上新 word 对全局 pair_freq 的贡献，并更新倒排索引。
            for pair, multiplicity in _word_pairs(word).items():
                pair_freq[pair] += count * multiplicity
                changed_pairs.add(pair)
                pair_words[pair].add(word)

        # ---- 5.4 记录 merge 结果 ----
        # merges 记录的是 merge 发生那一刻的两个 byte token；
        # 新 token 的 bytes 就是两者的拼接。
        merges.append(selected)
        vocab[len(vocab)] = selected[0] + selected[1]

        # ---- 5.5 把频次变化的 pair 重新 push 进堆 ----
        #
        # 这里不删除堆里的旧条目，而是直接 push 新条目：
        #   - 旧条目会因 count 不匹配而在未来 pop 时被跳过（lazy deletion）；
        #   - 频次降到 0 的 pair 从 pair_freq / pair_words 中删除，防止内存泄漏。
        for pair in changed_pairs:
            count = pair_freq.get(pair, 0)
            if count > 0:
                heapq.heappush(heap, (-count, _ReverseLex(pair)))
            else:
                pair_freq.pop(pair, None)
                pair_words.pop(pair, None)

        # ---- 5.6 堆压缩 ----
        #
        # lazy deletion 会让堆里积累大量过期条目。
        # 当堆大小超过 max(4096, 8 * 活跃 pair 数) 时，直接用当前 pair_freq
        # 重建堆，把内存和 pop 成本控制在合理范围。
        # max(4096, ...) 是为了避免在 pair 很少时过于频繁地重建。
        if len(heap) > max(4096, 8 * len(pair_freq)):
            heap = [(-n, _ReverseLex(p)) for p, n in pair_freq.items() if n > 0]
            heapq.heapify(heap)

    # 返回最终词表与按学习顺序排列的 merges。
    # 若 heap 提前耗尽（可 merge 的 pair 用完了），词表会小于 vocab_size。
    return vocab, merges
