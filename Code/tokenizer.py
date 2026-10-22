#!/usr/bin/env python
# @FileName  :tokenizer.py
# @Time      :2026/10/20
# @Author    :Xiaoming
"""Byte-level BPE tokenizer：按 merge rank 编码，并支持 special token。

这个类负责 BPE 的**使用阶段**（inference / encoding），与 ``train_bpe.py``
的训练阶段严格分开：

    training:  corpus -> vocab + merges
    encoding:  new text + vocab + merges -> token ids

编码时的核心规则：

1. 先按 GPT-2 regex 做 pre-tokenization；
2. 对每个 pre-token，反复选择“当前相邻 pair 中 merge rank 最小”的那个进行合并，
   直到没有可合并的 pair 为止（rank 越小 = 越早学到 = 优先级越高）；
3. special token 在进入 BPE 之前就被识别出来，直接映射为单个 id，
   不参与 pre-tokenization，也不参与 merge（保持 atomic）。

对外接口：
    encode / decode / encode_iterable / to_files / from_files
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

import regex

from .pretokenization import GPT2_PATTERN


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """用训练得到的 ``vocab`` / ``merges`` 构造一个可用的 tokenizer。

        Args:
            vocab: ``id -> bytes``。通常前 256 个是单字节 token，
                之后是 special tokens 与 merge 产生的 token。
            merges: 按学习顺序排列的 pair 列表，顺序即 merge rank。
            special_tokens: 需要保持 atomic 的 special token。
                若某个 special token 还不在 vocab 里，会自动追加到末尾。

        Notes:
            构造完成后应当把当前对象视为**不可变**的：
            ``_bpe_word`` 使用了 ``lru_cache``，之后再修改 vocab/merges
            不会让缓存失效。
        """
        # 做一份防御性拷贝，避免外部之后修改传入的容器影响 tokenizer 状态。
        self.vocab = dict(vocab)
        self.merges = list(merges)

        # 去重，同时保留用户传入的顺序（会影响自动追加 special token 时的 id）。
        self.special_tokens = list(dict.fromkeys(special_tokens or []))

        # bytes -> id 的反向索引。
        # 编码时用它把 BPE 合并后的 bytes 查回 token id。
        # 注意：这里假定 vocab 的 value（bytes）是唯一的；若有重复，
        # 后面的 id 会覆盖前面的。
        self.bytes_to_id = {value: key for key, value in self.vocab.items()}

        # merge rank：pair 在 merges 列表中的下标。
        # rank 越小，表示该 merge 越早被学到，编码时优先级越高。
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        # ---- special token -> id ----
        # special token 必须能直接映射到一个 id：
        #   - 如果在训练得到的 vocab 里已经存在，直接复用；
        #   - 否则追加到 vocab 末尾（id = 当前最大 id + 1）。
        self.special_to_id: dict[str, int] = {}
        for special in self.special_tokens:
            if not special:
                raise ValueError("Empty special token")

            encoded = special.encode("utf-8")
            if encoded not in self.bytes_to_id:
                next_id = max(self.vocab, default=-1) + 1
                self.vocab[next_id] = encoded
                self.bytes_to_id[encoded] = next_id

            self.special_to_id[special] = self.bytes_to_id[encoded]

        # ---- special token 的匹配正则 ----
        # 与 pretokenization.py 中一致：
        #   - 长度降序：保证 overlapping special token 采用 longest-match-first；
        #   - 字典序升序：长度相同时保证 deterministic。
        # 没有 special token 时保持 None，encode 可以走更快的纯 BPE 路径。
        tokens = sorted(self.special_tokens, key=lambda s: (-len(s), s))
        self._special_pattern = (
            regex.compile("|".join(regex.escape(t) for t in tokens)) if tokens else None
        )

    @lru_cache(maxsize=256)
    def _bpe_word(self, data: bytes) -> tuple[int, ...]:
        """对单个 pre-token（已经转成 UTF-8 bytes）执行 BPE merge。

        编码算法：
            在当前所有相邻 pair 中，选择 ``merge_ranks`` 最小的那个进行合并；
            重复直到没有任何相邻 pair 出现在 merges 中。

        为什么要选 rank 最小，而不是任意可合并的 pair？
            因为 merges 的顺序记录了训练时的学习顺序。
            越早学到的 merge 优先级越高；按 rank 从小到大应用，
            才与训练时的 segmentation 保持一致。

        例：
            merges = [(t,h), (th,e), (i,n)]
            "the" -> t h e -> th e -> the

        Args:
            data: 单个 pre-token 的 UTF-8 bytes。

        Returns:
            该 pre-token 编码后的 token id 元组。

        Notes:
            - ``lru_cache`` 的 key 包含 ``self``，因此不同 Tokenizer 实例不会
              串用结果；但缓存挂在函数上，会持有实例引用。
            - 这里假定 vocab 与 merges 一致：任何 merge 产生的 bytes 都能在
              ``bytes_to_id`` 中找到，否则会抛 ``KeyError``。
        """
        # 初始状态：每个字节是一个独立 symbol。
        pieces = [bytes((b,)) for b in data]

        # 每一轮合并一个 pair，直到没有可用的 merge。
        while len(pieces) > 1:
            # 收集当前相邻 pair 中“存在 merge rank”的候选，
            # 连同它们的 rank 一起记录。
            candidates = [
                (self.merge_ranks[pair], pair)
                for pair in zip(pieces, pieces[1:])
                if pair in self.merge_ranks
            ]
            if not candidates:
                break

            # 选择 rank 最小的 pair（rank 唯一，因此不会有并列）。
            _, selected = min(candidates, key=lambda item: item[0])

            # 在本次迭代中把选中的 pair 从左到右、非重叠地合并。
            # 这与训练阶段的 _merge_word 语义完全一致。
            changed: list[bytes] = []
            i = 0
            while i < len(pieces):
                if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == selected:
                    changed.append(pieces[i] + pieces[i + 1])
                    i += 2
                else:
                    changed.append(pieces[i])
                    i += 1
            pieces = changed

        # 把最终的 bytes symbol 映射回 token id。
        return tuple(self.bytes_to_id[part] for part in pieces)

    def _encode_plain(self, text: str) -> Iterator[int]:
        """对不含 special token 的普通文本做 pre-tokenization + BPE。

        先按 GPT-2 regex 切成 pre-token，再对每个 pre-token 单独执行 BPE。
        不同 pre-token 之间绝不会发生 merge。
        """
        for match in GPT2_PATTERN.finditer(text):
            yield from self._bpe_word(match.group().encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        """把任意文本编码成 token id 列表。

        处理顺序：
            1. 先扫描 special token；
            2. 两段 special token 之间的 ordinary span 走普通 BPE；
            3. special token 本身直接输出其 id（atomic）。

        注意：special token 不会进入 GPT-2 regex，也不会被拆成更小的 token。
        """
        # 没有 special token：整段文本都是 ordinary text，直接走快路径。
        if self._special_pattern is None:
            return list(self._encode_plain(text))

        result: list[int] = []
        cursor = 0

        for match in self._special_pattern.finditer(text):
            # match 之前的普通片段：做 pre-tokenization + BPE。
            result.extend(self._encode_plain(text[cursor : match.start()]))

            # special token 本身：直接查表得到单个 id，绕过 BPE。
            result.append(self.special_to_id[match.group()])

            cursor = match.end()

        # 处理最后一个 special token 之后的尾部普通文本。
        result.extend(self._encode_plain(text[cursor:]))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码：对每个 chunk（通常是文件的一行）调用 ``encode``。

        适用场景是 line-oriented corpus：每行本身就是一个完整的、可独立编码的单元，
        这样可以避免把整个文件读进内存。

        重要限制：
            chunk 的边界不能切断 special token，也不能切断一个普通 pre-token。
            例如把 " wonderful" 拆成 " won" + "derful" 会得到不同的结果。
            如果输入不满足这个条件，应改用带缓冲的流式编码（保留跨边界上下文），
            或者先拼接再编码。

        Args:
            iterable: 文本 chunk 的可迭代对象。

        Yields:
            依次产生的 token id。
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        """把 token id 序列解码回文本。

        关键点：必须先把所有 token 的 bytes **拼接成一个完整的 byte stream**，
        再统一做 UTF-8 解码。

        不能对每个 token 单独 decode：因为一个 token 的 bytes 可能只是某个
        多字节 UTF-8 字符的一部分（例如 "你" 的 3 个字节可能被拆在不同 token 里）。

        Args:
            ids: token id 序列。

        Returns:
            解码后的字符串。

        Notes:
            ``errors="replace"`` 保证遇到非法 UTF-8 时不会抛异常，
            而是用替换字符（�）代替；这会静默掩盖数据错误，
            调试阶段可以改成 ``errors="strict"`` 以便尽早发现问题。
        """
        # 先恢复完整 byte stream，再统一 decode。
        raw = b"".join(self.vocab[int(token_id)] for token_id in ids)
        return raw.decode("utf-8", errors="replace")

    def to_files(self, vocab_path: str | Path, merges_path: str | Path) -> None:
        """把 vocab 和 merges 序列化到磁盘（JSON 格式）。

        格式设计：
            - vocab: {"0": "00", "1": "01", ..., "256": "3c7c..."}，
              key 是 id 的字符串形式（JSON 的 object key 必须是字符串），
              value 是 bytes 的 hex（可移植、可读，避免编码问题）；
            - merges: [["74", "68"], ["7468", "65"], ...]，每一项是
              两个 bytes 的 hex 字符串。

        Args:
            vocab_path: vocab 输出路径。
            merges_path: merges 输出路径。
        """
        with open(vocab_path, "w", encoding="utf-8") as handle:
            json.dump({str(i): data.hex() for i, data in self.vocab.items()}, handle)

        with open(merges_path, "w", encoding="utf-8") as handle:
            json.dump([[a.hex(), b.hex()] for a, b in self.merges], handle)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """从 ``to_files`` 生成的 JSON 文件恢复 Tokenizer。

        Args:
            vocab_filepath: vocab JSON 路径。
            merges_filepath: merges JSON 路径。
            special_tokens: 需要保持 atomic 的 special token。
                若训练时已把它们写入 vocab，这里会直接复用对应 id；
                否则 ``__init__`` 会把它们追加到 vocab 末尾。

        Returns:
            恢复出来的 ``Tokenizer`` 实例。

        Notes:
            使用 ``cls(...)`` 而不是硬编码 ``Tokenizer(...)``，
            这样 subclasses 也能正确构造。
        """
        with open(vocab_filepath, encoding="utf-8") as handle:
            # JSON 的 key 一定是字符串，这里转回 int；
            # value 是 hex 字符串，用 bytes.fromhex 还原成 bytes。
            vocab = {
                int(i): bytes.fromhex(data) for i, data in json.load(handle).items()
            }

        with open(merges_filepath, encoding="utf-8") as handle:
            merges = [
                (bytes.fromhex(a), bytes.fromhex(b)) for a, b in json.load(handle)
            ]

        return cls(vocab, merges, special_tokens)
