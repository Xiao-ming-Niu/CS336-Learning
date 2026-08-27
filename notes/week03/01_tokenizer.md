# Byte-level BPE Tokenizer

## 1. Tokenizer 到底解决什么问题？

语言模型不能直接处理字符串。

Transformer 接收的是整数：

$$
x_1,x_2,\ldots,x_T
$$

其中

$$
x_i\in\{0,1,\ldots,V-1\}
$$

这里：

- $V$：vocabulary size
- $T$：sequence length
- $x_i$：第 $i$ 个 token ID

所以 Tokenizer 本质上是在实现映射：

$$
\text{text}
\longrightarrow
\text{token IDs}
$$

即：

$$
f:\Sigma^*
\rightarrow
\{0,\ldots,V-1\}^*
$$

例如：

```text
"hello world"

        ↓ Tokenizer

[314, 812, 42]
```

然后 Transformer 才通过 Embedding：

$$
[B,T]
\rightarrow
[B,T,d_{\text{model}}]
$$

进入神经网络。

---

# 2. 为什么不能直接按 character tokenize？

一个自然想法是：

```text
hello

→ h e l l o
```

但 Unicode character 的集合非常大。

例如：

```text
a
中
🚀
é
한
```

而且 Unicode 本身还有 normalization、组合字符等问题。

更重要的是：

> Unicode character 并不是计算机真正存储文本的基本单位。

文本最终会通过某种 encoding 变成 bytes。

现代 NLP 最常见的选择是：

```text
Unicode
    ↓
UTF-8
    ↓
bytes
```

---

# 3. UTF-8 与 Byte-level Tokenization

UTF-8 把 Unicode 字符编码成一个或多个 byte。

一个 byte 的取值范围为：

$$
0,\ldots,255
$$

因此只需要 **256 个基础 token** 就可以表示任意 UTF-8 文本。

例如 ASCII 字符：

```python
"a".encode("utf-8")
```

得到：

```text
b'a'
```

对应：

$$
97
$$

但中文：

```python
"你".encode("utf-8")
```

实际上是三个 bytes。

因此：

```text
Unicode character
≠
byte
```

Byte-level tokenizer 的第一层 vocabulary 就可以定义为：

$$
V_0=\{0,1,\ldots,255\}
$$

其中：

$$
v_i=\operatorname{bytes}([i])
$$

代码：

```python
vocab = {
    token_id: bytes([token_id])
    for token_id in range(256)
}
```

这带来一个非常重要的性质：

> **理论上不存在无法表示的 Unicode 字符。**

因此 byte-level tokenizer 通常不需要传统意义上的 `<unk>` token。

---

# 4. 但是为什么不能直接一个 byte 一个 token？

当然可以。

例如：

```text
hello
```

UTF-8 bytes：

```text
h e l l o
```

最终变成 5 个 token。

但语言模型的计算成本与 sequence length $T$ 强相关。

特别是 Self-Attention：

$$
QK^\top
$$

产生一个：

$$
T\times T
$$

矩阵。

Attention 的主要计算复杂度近似：

$$
O(T^2d)
$$

如果永远一个 byte 一个 token，sequence 会非常长。

所以我们希望：

```text
常见 byte sequence
        ↓
合并
        ↓
一个 token
```

例如：

```text
t h e
```

最终可能成为：

```text
the
```

一个 token。

这就是 BPE 的核心思想。

---

# 5. BPE 的数学目标

BPE：

> Byte Pair Encoding

核心操作非常简单：

> **不断找到语料中出现频率最高的相邻 token pair，然后把它合并成一个新的 token。**

设当前 corpus token sequence 为：

$$
S=(s_1,s_2,\ldots,s_N)
$$

对于一个 token pair：

$$
(a,b)
$$

定义其相邻出现次数：

$$
C(a,b)
=
\sum_{i=1}^{N-1}
\mathbf{1}
[s_i=a\land s_{i+1}=b]
$$

其中 indicator function：

$$
\mathbf{1}[P]
=
\begin{cases}
1,&P\text{ 成立}\\
0,&P\text{ 不成立}
\end{cases}
$$

每一步选择：

$$
(a^*,b^*)
=
\arg\max_{(a,b)} C(a,b)
$$

然后创建：

$$
c=a^* \Vert b^*
$$

这里：

$$
\Vert
$$

表示 byte concatenation。

然后所有非重叠的：

$$
(a^*,b^*)
$$

被替换成：

$$
c
$$

---

# 6. 手算一个 BPE

先用一个极简例子理解。

Corpus：

```text
abababab
```

先暂时忽略 pre-tokenization。

初始化：

```text
a b a b a b a b
```

统计 adjacent pairs：

```text
(a, b) : 4
(b, a) : 3
```

所以：

$$
(a,b)
=
\arg\max C(a,b)
$$

第一次 merge：

```text
a b → ab
```

得到：

```text
ab ab ab ab
```

现在：

```text
(ab, ab) : 3
```

第二次 merge：

```text
ab ab → abab
```

注意必须 **non-overlapping merge**。

所以：

```text
ab ab ab ab
```

变成：

```text
abab abab
```

而不是让一个 token 同时参加左右两个 merge。

最终 vocabulary 从：

```text
a
b
```

逐渐增加：

```text
ab
abab
...
```

---

# 7. BPE Vocabulary Size

Byte-level BPE 初始已经有：

$$
256
$$

个 token。

假设：

$$
V_{\text{target}}
$$

是最终 vocabulary size。

如果没有 special token，则最多进行：

$$
M=V_{\text{target}}-256
$$

次 merge。

例如：

$$
V=10,000
$$

那么：

$$
M=10,000-256=9744
$$

如果有 $S$ 个 special tokens：

$$
M=V-256-S
$$

例如一个 `<|endoftext|>`：

$$
M=10000-256-1=9743
$$

---

# 8. 一个非常重要的区别：BPE Training vs Encoding

这两个阶段不能混淆。

## 8.1 Training

训练 tokenizer 时：

```text
training corpus
      ↓
UTF-8
      ↓
bytes
      ↓
pair statistics
      ↓
most frequent pair
      ↓
merge
      ↓
repeat
```

最终学习：

```text
Vocabulary
+
Ordered Merge Rules
```

例如：

```text
rank 0: (b"t",  b"h")
rank 1: (b"th", b"e")
rank 2: (b"i",  b"n")
...
```

可以理解为：

$$
R(a,b)
=
\text{merge rank}
$$

rank 越小表示越早学到。

---

# 9. BPE Encoding 不是 Longest Match

这是 Tokenizer 最值得理解的问题之一。

假设 vocabulary 里面存在：

```text
a
b
c
ab
bc
abc
```

看到：

```text
abc
```

不能简单说：

> vocabulary 里 `abc` 最长，所以直接选 `abc`。

BPE encoding 的结果取决于 **merge history**。

例如 merge rules：

```text
1. (b, c) → bc
2. (a, b) → ab
3. (a, bc) → abc
```

输入：

```text
a b c
```

第一个有效 merge 是：

```text
b c
```

得到：

```text
a bc
```

随后：

```text
a bc
```

才可能 merge：

```text
abc
```

所以真正决定 encoding 的不仅仅是：

```text
vocab
```

而是：

```text
vocab + ordered merges
```

因此必须保存：

```python
vocab
merges
```

而不能只保存 vocabulary。

---

# 10. Pre-tokenization

真实 tokenizer 通常不会让 BPE 任意跨整个 document 合并。

否则：

```text
hello world
```

可能逐渐出现跨越任意 whitespace / punctuation 的奇怪 token。

因此通常先：

$$
\text{text}
\rightarrow
p_1,p_2,\ldots,p_k
$$

得到若干 pre-token。

然后：

> BPE 只允许在每个 pre-token 内部发生。

例如：

```text
"Hello, world!"
```

可能先切成：

```text
"Hello"
","
" world"
"!"
```

然后分别做 BPE。

所以：

```text
Pre-tokenization
≠
BPE
```

前者控制：

> **BPE 可以在哪里 merge。**

后者学习：

> **哪些 token pair 应该 merge。**

---

# 11. 为什么要保留前导空格？

GPT 系 tokenizer 经常产生类似：

```text
"hello"
" world"
```

而不是：

```text
"hello"
" "
"world"
```

这是因为：

```text
" world"
```

作为一个整体模式在自然语言中非常常见。

它能让 tokenizer 学到：

```text
word boundary + word
```

这样的统计结构。

所以你以后观察 GPT tokenizer 时，经常会发现：

```text
"hello"
```

和：

```text
" hello"
```

对应不同 token。

---

# 12. Special Tokens

语言模型还存在一些具有结构意义的 token，例如：

```text
<|endoftext|>
```

它应该：

```text
"<|endoftext|>"
        ↓
     一个 token
```

而不能被普通 BPE 拆成：

```text
<
|
end
of
text
|
>
```

因此处理顺序应该是：

```text
input text
    ↓
detect special tokens
    ↓
ordinary text
    ↓
pre-tokenization
    ↓
BPE
```

而不是先 BPE 再尝试恢复 special token。

---

# 13. Decode 的数学原理

假设 tokenizer 输出：

$$
t_1,t_2,\ldots,t_n
$$

Vocabulary 定义：

$$
v(t_i)
=
\text{bytes associated with token }t_i
$$

那么 decode 首先做：

$$
b
=
v(t_1)
\Vert
v(t_2)
\Vert
\cdots
\Vert
v(t_n)
$$

最后：

$$
x=\operatorname{UTF8Decode}(b)
$$

也就是：

```python
all_bytes = b"".join(vocab[token_id] for token_id in ids)
text = all_bytes.decode("utf-8")
```

---

# 14. 为什么不能逐 token UTF-8 decode？

这是 byte-level tokenizer 一个很容易写错的地方。

中文：

```text
你
```

UTF-8 由多个 bytes 构成。

BPE 完全可能产生一个 token，它只包含这个 Unicode character 的**部分 UTF-8 bytes**。

因此：

```python
token_bytes.decode("utf-8")
```

可能报错。

正确方式一定是：

```text
token 1 bytes
token 2 bytes
token 3 bytes
      ↓
全部 concatenate
      ↓
完整 UTF-8 byte stream
      ↓
decode UTF-8
```

---

# 15. Tokenizer 最重要的不变量

一个正确 tokenizer 应该满足：

$$
\boxed{
\operatorname{decode}(
\operatorname{encode}(x)
)
=
x
}
$$

即 round-trip invariant。

必须测试：

```text
ASCII
Chinese
Japanese
Korean
emoji
whitespace
tabs
newlines
special tokens
```

例如：

```python
text = "你好，世界 🚀"

assert (
    tokenizer.decode(
        tokenizer.encode(text)
    )
    == text
)
```

---

# 16. Tokenizer 为什么影响 LLM 计算量？

假设 tokenizer A：

```text
一段文本 → 1000 tokens
```

Tokenizer B：

```text
同一段文本 → 700 tokens
```

Attention matrix 从：

$$
1000^2=1,000,000
$$

变成：

$$
700^2=490,000
$$

只有原来的：

$$
\frac{490000}{1000000}=49\%
$$

所以 tokenizer 并不是无关紧要的 preprocessing。

它会影响：

```text
sequence length
training FLOPs
inference FLOPs
KV cache
attention memory
context utilization
dataset token count
```

---

# 17. Vocabulary Size 的 Trade-off

增大 $V$：

```text
更长的常见 byte sequences
→ 可以变成单个 token
→ sequence length T 通常下降
```

但是 embedding 参数量：

$$
Vd_{\text{model}}
$$

LM head：

$$
d_{\text{model}}V
$$

都会随 $V$ 增大。

所以存在 trade-off：

$$
\boxed{
\text{larger }V
\Rightarrow
\text{shorter }T
\quad\text{but}\quad
\text{larger embedding / LM head}
}
$$

Tokenizer 设计本身就是 model design 的一部分。

---

# 18. Compression Ratio

可以定义：

$$
\text{bytes per token}
=
\frac{
\text{UTF-8 byte count}
}{
\text{token count}
}
$$

例如：

```text
1000 bytes
→ 400 tokens
```

那么：

$$
\frac{1000}{400}
=
2.5
$$

即：

```text
2.5 bytes/token
```

一般来说，值更大表示一个 token 平均覆盖更多原始 bytes。

但不能单独依靠 compression ratio 判断 tokenizer 好坏。

---

# 19. From-scratch Python 实现

现在开始实现。

建议目录：

```text
cs336-from-scratch/
├── src/
│   └── tokenizer.py
└── tests/
    └── test_tokenizer.py
```

学习路线本身要求先独立实现 Byte-level BPE，再进入 Transformer，因此这里保持核心 primitive 完全手写。

下面是一版偏 **correctness-first** 的 reference implementation。

```python
# src/tokenizer.py

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import re
from typing import TypeAlias


Token: TypeAlias = bytes
Pair: TypeAlias = tuple[Token, Token]


_PRETOKEN_PATTERN = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+""",
    flags=re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class BPEModel:
    vocab: dict[int, bytes]
    merges: list[Pair]
```

---

# 20. Primitive 1：Pre-tokenization

```python
def pretokenize(text: str) -> list[str]:
    if not text:
        return []

    return [
        match.group(0)
        for match in _PRETOKEN_PATTERN.finditer(text)
    ]
```

先验证一个非常重要的 property：

```python
text = "Hello, world! 123"

pieces = pretokenize(text)

assert "".join(pieces) == text
```

也就是说 pre-tokenization：

> 可以建立边界，但不能损坏原始文本。

---

# 21. Primitive 2：统计相邻 pair

数学上：

$$
C(a,b)
=
\sum_i
\mathbf 1[
s_i=a
\land
s_{i+1}=b
]
$$

代码几乎就是数学公式的直接翻译：

```python
def count_pairs(
    tokens: Sequence[Token],
) -> Counter[Pair]:
    return Counter(
        zip(tokens, tokens[1:])
    )
```

例如：

```python
tokens = [
    b"a",
    b"b",
    b"a",
    b"b",
]

print(count_pairs(tokens))
```

结果：

```text
(a, b) → 2
(b, a) → 1
```

这一个函数你最后应该能够**不看答案直接写出来**。

---

# 22. Primitive 3：Non-overlapping Merge

定义：

$$
(a,b)\rightarrow ab
$$

代码：

```python
def merge_pair(
    tokens: Sequence[Token],
    pair: Pair,
) -> list[Token]:
    left, right = pair
    merged = left + right

    result: list[Token] = []

    i = 0

    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i] == left
            and tokens[i + 1] == right
        ):
            result.append(merged)
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return result
```

关键在：

```python
i += 2
```

因为 merge 必须 non-overlapping。

例如：

```text
a a a
```

merge：

```text
(a, a)
```

正确结果：

```text
aa a
```

而不是让中间的 `a` 被使用两次。

---

# 23. 初始化 Vocabulary

```python
def _initial_vocab(
    special_tokens: Sequence[str],
) -> dict[int, bytes]:
    vocab = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    next_id = 256

    for token in special_tokens:
        encoded = token.encode("utf-8")

        if encoded not in vocab.values():
            vocab[next_id] = encoded
            next_id += 1

    return vocab
```

现在：

```python
vocab[0]
```

是：

```text
b"\x00"
```

而：

```python
vocab[97]
```

就是：

```text
b"a"
```

---

# 24. 将一个 pre-token 转成 byte tokens

```python
def _bytes_to_tokens(
    text: str,
) -> list[bytes]:
    return [
        bytes([value])
        for value in text.encode("utf-8")
    ]
```

例如：

```python
_bytes_to_tokens("abc")
```

得到：

```python
[
    b"a",
    b"b",
    b"c",
]
```

但：

```python
_bytes_to_tokens("你")
```

会得到多个 byte token。

这正是 byte-level 的核心。

---

# 25. BPE Training

我们现在实现：

$$
(a^*,b^*)
=
\arg\max C(a,b)
$$

为了避免每次扫描整个巨大 corpus，这里先把相同 pre-token 压缩成 frequency table。

例如：

```text
" the" 出现 10000 次
```

不需要存 10000 份。

存：

```text
token sequence → frequency
```

即可。

```python
def train_bpe_from_text(
    text: str,
    vocab_size: int,
    special_tokens: Sequence[str] = (),
) -> BPEModel:
    minimum_vocab_size = (
        256 + len(special_tokens)
    )

    if vocab_size < minimum_vocab_size:
        raise ValueError(
            "vocab_size must be at least "
            f"{minimum_vocab_size}"
        )

    vocab = _initial_vocab(special_tokens)
    merges: list[Pair] = []

    word_counts: Counter[tuple[Token, ...]] = Counter()

    pieces = pretokenize(text)

    for piece in pieces:
        token_sequence = tuple(
            _bytes_to_tokens(piece)
        )

        if token_sequence:
            word_counts[token_sequence] += 1

    while len(vocab) < vocab_size:
        pair_counts: Counter[Pair] = Counter()

        for token_sequence, frequency in word_counts.items():
            counts = count_pairs(token_sequence)

            for pair, count in counts.items():
                pair_counts[pair] += count * frequency

        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )

        new_token = (
            best_pair[0]
            + best_pair[1]
        )

        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        updated_counts: Counter[
            tuple[Token, ...]
        ] = Counter()

        for token_sequence, frequency in word_counts.items():
            merged = merge_pair(
                token_sequence,
                best_pair,
            )

            updated_counts[
                tuple(merged)
            ] += frequency

        word_counts = updated_counts

    return BPEModel(
        vocab=vocab,
        merges=merges,
    )
```

这里最重要的不是代码长度，而是理解这一句：

```python
best_pair = max(...)
```

它对应的正是：

$$
\boxed{
(a^*,b^*)
=
\arg\max_{(a,b)}C(a,b)
}
$$

---

# 26. 为什么要记录 merge 顺序？

训练结束：

```python
merges = [
    (b"t", b"h"),
    (b"th", b"e"),
    ...
]
```

可以构造：

```python
merge_ranks = {
    pair: rank
    for rank, pair in enumerate(merges)
}
```

于是：

$$
R(a,b)=\text{rank}
$$

Encoding 时：

> 每次应用当前 sequence 中 rank 最小的可用 pair。

这才能忠实重现训练得到的 BPE algorithm。

---

# 27. BPETokenizer

```python
class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Sequence[Pair],
        special_tokens: Sequence[str] = (),
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)

        self._bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes
            in self.vocab.items()
        }

        self._merge_ranks = {
            pair: rank
            for rank, pair
            in enumerate(self.merges)
        }

        self._special_token_to_id: dict[str, int] = {}

        for token in special_tokens:
            encoded = token.encode("utf-8")

            if encoded not in self._bytes_to_id:
                raise ValueError(
                    f"Special token {token!r} "
                    "is missing from vocabulary"
                )

            self._special_token_to_id[token] = (
                self._bytes_to_id[encoded]
            )

        self.special_tokens = tuple(
            sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
```

---

# 28. Encoding 一个普通 pre-token

先把它变成 bytes：

```text
Unicode
↓
UTF-8
↓
[b1, b2, ..., bn]
```

然后不断寻找：

$$
\arg\min R(a,b)
$$

也就是当前可用 pair 中 rank 最小的那个。

```python
    def _encode_piece(
        self,
        piece: str,
    ) -> list[int]:
        tokens = _bytes_to_tokens(piece)

        while len(tokens) >= 2:
            candidate_pairs = set(
                zip(tokens, tokens[1:])
            )

            valid_pairs = [
                pair
                for pair in candidate_pairs
                if pair in self._merge_ranks
            ]

            if not valid_pairs:
                break

            best_pair = min(
                valid_pairs,
                key=self._merge_ranks.__getitem__,
            )

            tokens = merge_pair(
                tokens,
                best_pair,
            )

        return [
            self._bytes_to_id[token]
            for token in tokens
        ]
```

注意：

训练的时候是：

$$
\arg\max C(a,b)
$$

推理的时候却是：

$$
\arg\min R(a,b)
$$

因为：

- training 在**学习 merge rules**
- encoding 在**重放已经学好的 ordered rules**

这是一个非常值得记住的区别。

---

# 29. 普通文本 Encoding

```python
    def _encode_ordinary(
        self,
        text: str,
    ) -> list[int]:
        ids: list[int] = []

        for piece in pretokenize(text):
            ids.extend(
                self._encode_piece(piece)
            )

        return ids
```

---

# 30. Special-token-aware Encode

我们让 special tokens 优先于普通 BPE。

```python
    def encode(
        self,
        text: str,
    ) -> list[int]:
        if not text:
            return []

        if not self.special_tokens:
            return self._encode_ordinary(text)

        pattern = re.compile(
            "("
            + "|".join(
                re.escape(token)
                for token in self.special_tokens
            )
            + ")"
        )

        ids: list[int] = []

        for part in pattern.split(text):
            if not part:
                continue

            special_id = (
                self._special_token_to_id.get(part)
            )

            if special_id is not None:
                ids.append(special_id)
            else:
                ids.extend(
                    self._encode_ordinary(part)
                )

        return ids
```

数据流：

```text
text
 ↓
special token split
 ↓
ordinary segment
 ↓
pre-tokenization
 ↓
UTF-8 bytes
 ↓
BPE ranks
 ↓
token IDs
```

---

# 31. Decode

Decode 反而非常简单：

```python
    def decode(
        self,
        ids: Sequence[int],
    ) -> str:
        chunks: list[bytes] = []

        for token_id in ids:
            try:
                chunks.append(
                    self.vocab[token_id]
                )
            except KeyError as error:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                ) from error

        byte_string = b"".join(chunks)

        return byte_string.decode("utf-8")
```

注意设计：

```python
b"".join(...)
```

之后才：

```python
.decode("utf-8")
```

不要逐 token decode。

---

# 32. Streaming Interface

大型 dataset 不能全部读进 RAM。

所以最好提供：

```python
    def encode_iterable(
        self,
        texts: Iterable[str],
    ) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text)
```

之后处理大规模数据：

```python
for token_id in tokenizer.encode_iterable(dataset):
    ...
```

而不需要：

```text
entire dataset
→ RAM
→ tokenize
```

---

# 33. 完整 `tokenizer.py`

把前面的代码组合后：

```python
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import re
from typing import TypeAlias


Token: TypeAlias = bytes
Pair: TypeAlias = tuple[Token, Token]


_PRETOKEN_PATTERN = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+""",
    flags=re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class BPEModel:
    vocab: dict[int, bytes]
    merges: list[Pair]


def pretokenize(text: str) -> list[str]:
    if not text:
        return []

    return [
        match.group(0)
        for match in _PRETOKEN_PATTERN.finditer(text)
    ]


def count_pairs(
    tokens: Sequence[Token],
) -> Counter[Pair]:
    return Counter(
        zip(tokens, tokens[1:])
    )


def merge_pair(
    tokens: Sequence[Token],
    pair: Pair,
) -> list[Token]:
    left, right = pair
    merged = left + right

    output: list[Token] = []
    i = 0

    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i] == left
            and tokens[i + 1] == right
        ):
            output.append(merged)
            i += 2
        else:
            output.append(tokens[i])
            i += 1

    return output


def _bytes_to_tokens(
    text: str,
) -> list[bytes]:
    return [
        bytes([value])
        for value in text.encode("utf-8")
    ]


def _initial_vocab(
    special_tokens: Sequence[str],
) -> dict[int, bytes]:
    vocab = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    next_id = 256

    for token in special_tokens:
        encoded = token.encode("utf-8")

        if encoded not in vocab.values():
            vocab[next_id] = encoded
            next_id += 1

    return vocab


def train_bpe_from_text(
    text: str,
    vocab_size: int,
    special_tokens: Sequence[str] = (),
) -> BPEModel:
    minimum_vocab_size = (
        256 + len(special_tokens)
    )

    if vocab_size < minimum_vocab_size:
        raise ValueError(
            "vocab_size must be at least "
            f"{minimum_vocab_size}"
        )

    vocab = _initial_vocab(special_tokens)
    merges: list[Pair] = []

    word_counts: Counter[
        tuple[Token, ...]
    ] = Counter()

    for piece in pretokenize(text):
        token_sequence = tuple(
            _bytes_to_tokens(piece)
        )

        if token_sequence:
            word_counts[token_sequence] += 1

    while len(vocab) < vocab_size:
        pair_counts: Counter[Pair] = Counter()

        for token_sequence, frequency in word_counts.items():
            local_counts = count_pairs(
                token_sequence
            )

            for pair, count in local_counts.items():
                pair_counts[pair] += (
                    count * frequency
                )

        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair,
            ),
        )

        new_token = (
            best_pair[0]
            + best_pair[1]
        )

        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        updated_counts: Counter[
            tuple[Token, ...]
        ] = Counter()

        for token_sequence, frequency in word_counts.items():
            merged = merge_pair(
                token_sequence,
                best_pair,
            )

            updated_counts[
                tuple(merged)
            ] += frequency

        word_counts = updated_counts

    return BPEModel(
        vocab=vocab,
        merges=merges,
    )


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Sequence[Pair],
        special_tokens: Sequence[str] = (),
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)

        self._bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes
            in self.vocab.items()
        }

        self._merge_ranks = {
            pair: rank
            for rank, pair
            in enumerate(self.merges)
        }

        self._special_token_to_id: dict[
            str,
            int,
        ] = {}

        for token in special_tokens:
            encoded = token.encode("utf-8")

            if encoded not in self._bytes_to_id:
                raise ValueError(
                    f"Special token {token!r} "
                    "is missing from vocabulary"
                )

            self._special_token_to_id[token] = (
                self._bytes_to_id[encoded]
            )

        self.special_tokens = tuple(
            sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_piece(
        self,
        piece: str,
    ) -> list[int]:
        tokens = _bytes_to_tokens(piece)

        while len(tokens) >= 2:
            candidate_pairs = set(
                zip(tokens, tokens[1:])
            )

            valid_pairs = [
                pair
                for pair in candidate_pairs
                if pair in self._merge_ranks
            ]

            if not valid_pairs:
                break

            best_pair = min(
                valid_pairs,
                key=self._merge_ranks.__getitem__,
            )

            tokens = merge_pair(
                tokens,
                best_pair,
            )

        return [
            self._bytes_to_id[token]
            for token in tokens
        ]

    def _encode_ordinary(
        self,
        text: str,
    ) -> list[int]:
        ids: list[int] = []

        for piece in pretokenize(text):
            ids.extend(
                self._encode_piece(piece)
            )

        return ids

    def encode(
        self,
        text: str,
    ) -> list[int]:
        if not text:
            return []

        if not self.special_tokens:
            return self._encode_ordinary(text)

        pattern = re.compile(
            "("
            + "|".join(
                re.escape(token)
                for token in self.special_tokens
            )
            + ")"
        )

        ids: list[int] = []

        for part in pattern.split(text):
            if not part:
                continue

            special_id = (
                self._special_token_to_id.get(part)
            )

            if special_id is not None:
                ids.append(special_id)
            else:
                ids.extend(
                    self._encode_ordinary(part)
                )

        return ids

    def decode(
        self,
        ids: Sequence[int],
    ) -> str:
        chunks: list[bytes] = []

        for token_id in ids:
            try:
                chunks.append(
                    self.vocab[token_id]
                )
            except KeyError as error:
                raise ValueError(
                    f"Unknown token ID: "
                    f"{token_id}"
                ) from error

        return b"".join(
            chunks
        ).decode("utf-8")

    def encode_iterable(
        self,
        texts: Iterable[str],
    ) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text)
```

---

# 34. 第一个实验

```python
from src.tokenizer import (
    BPETokenizer,
    train_bpe_from_text,
)


corpus = """
hello world
hello tokenizer
hello world
你好世界
你好 tokenizer
🚀 hello
"""


model = train_bpe_from_text(
    text=corpus,
    vocab_size=300,
    special_tokens=(
        "<|endoftext|>",
    ),
)


tokenizer = BPETokenizer(
    vocab=model.vocab,
    merges=model.merges,
    special_tokens=(
        "<|endoftext|>",
    ),
)


text = "hello 世界 🚀"

ids = tokenizer.encode(text)

print(ids)
print(tokenizer.decode(ids))

assert tokenizer.decode(ids) == text
```

重点不是 IDs 到底是多少。

IDs 本身是 tokenizer training corpus 的产物。

真正应该观察的是：

```text
input
↓
UTF-8
↓
BPE
↓
IDs
↓
decode
↓
原文
```

---

# 35. Tests

建立：

```text
tests/test_tokenizer.py
```

首先测试 primitive。

```python
from src.tokenizer import (
    BPETokenizer,
    count_pairs,
    merge_pair,
    pretokenize,
    train_bpe_from_text,
)


def test_count_pairs() -> None:
    tokens = [
        b"a",
        b"b",
        b"a",
        b"b",
    ]

    counts = count_pairs(tokens)

    assert counts[(b"a", b"b")] == 2
    assert counts[(b"b", b"a")] == 1
```

Merge：

```python
def test_merge_pair() -> None:
    tokens = [
        b"a",
        b"b",
        b"a",
        b"b",
    ]

    result = merge_pair(
        tokens,
        (b"a", b"b"),
    )

    assert result == [
        b"ab",
        b"ab",
    ]
```

最重要的是 overlap：

```python
def test_non_overlapping_merge() -> None:
    tokens = [
        b"a",
        b"a",
        b"a",
    ]

    result = merge_pair(
        tokens,
        (b"a", b"a"),
    )

    assert result == [
        b"aa",
        b"a",
    ]
```

Pre-tokenizer：

```python
def test_pretokenization_preserves_text() -> None:
    text = "Hello, world! 123"

    pieces = pretokenize(text)

    assert "".join(pieces) == text
```

---

# 36. Round-trip Tests

构造一个辅助函数：

```python
def make_tokenizer(
    corpus: str,
    vocab_size: int = 350,
    special_tokens: tuple[str, ...] = (),
) -> BPETokenizer:
    model = train_bpe_from_text(
        text=corpus,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    return BPETokenizer(
        vocab=model.vocab,
        merges=model.merges,
        special_tokens=special_tokens,
    )
```

ASCII：

```python
def test_ascii_round_trip() -> None:
    tokenizer = make_tokenizer(
        "hello world the quick brown fox"
    )

    text = "hello world"

    assert (
        tokenizer.decode(
            tokenizer.encode(text)
        )
        == text
    )
```

中文：

```python
def test_chinese_round_trip() -> None:
    tokenizer = make_tokenizer(
        "你好世界 中文测试 tokenizer",
        vocab_size=400,
    )

    text = "你好，世界"

    assert (
        tokenizer.decode(
            tokenizer.encode(text)
        )
        == text
    )
```

Emoji：

```python
def test_emoji_round_trip() -> None:
    tokenizer = make_tokenizer(
        "🚀 🔥 🤖 hello world",
        vocab_size=350,
    )

    text = "你好 🚀🔥"

    assert (
        tokenizer.decode(
            tokenizer.encode(text)
        )
        == text
    )
```

Whitespace：

```python
def test_whitespace_round_trip() -> None:
    tokenizer = make_tokenizer(
        "hello world\nfoo bar"
    )

    text = "hello   world\n\nfoo\tbar"

    assert (
        tokenizer.decode(
            tokenizer.encode(text)
        )
        == text
    )
```

---

# 37. Special Token Test

```python
def test_special_token() -> None:
    special = "<|endoftext|>"

    tokenizer = make_tokenizer(
        "hello world",
        vocab_size=300,
        special_tokens=(special,),
    )

    ids = tokenizer.encode(special)

    assert len(ids) == 1

    assert tokenizer.decode(ids) == special
```

还应该测试 special token 出现在正文中：

```python
def test_special_token_inside_text() -> None:
    special = "<|endoftext|>"

    tokenizer = make_tokenizer(
        "hello world",
        vocab_size=300,
        special_tokens=(special,),
    )

    text = (
        "hello"
        + special
        + "world"
    )

    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids) == text
```

---

# 38. 运行测试

```bash
pytest -q tests/test_tokenizer.py
```

最终至少要保证：

```text
count_pairs              ✓
non-overlapping merge    ✓
pre-tokenization         ✓
256-byte coverage        ✓
ASCII                    ✓
Chinese                  ✓
emoji                    ✓
whitespace               ✓
newline                  ✓
special token            ✓
encode/decode roundtrip  ✓
encode_iterable          ✓
```

---

# 39. Naive BPE 的复杂度

假设：

- 当前 corpus 一共有 $N$ 个 token
- 需要执行 $M$ 次 merge

如果每次：

```text
重新扫描 corpus
→ count all pairs
→ 找最大 pair
→ 再扫描
→ merge
```

那么粗略复杂度为：

$$
O(MN)
$$

实际还会存在 Python object / Counter / allocation 等大量 overhead。

所以这个实现的定位是：

$$
\boxed{
\text{Reference Implementation}
}
$$

而不是：

$$
\boxed{
\text{Production Tokenizer Trainer}
}
$$

这正符合我们之前确定的“双阶段”学习方式：先手写 primitive 保证数学正确和可解释，再进入 optimized implementation；整条 CS336 路线也是先建立完整 training stack，再进一步研究 systems 和性能。

---

# 40. 后续应该怎样优化？

等 reference implementation 全部测试通过以后，才开始优化。

最重要的观察是：

> 一次 merge 并不会改变整个 corpus 的所有 pair statistics。

假设：

```text
A B C D
```

merge：

```text
B C → BC
```

原本：

```text
(A, B)
(B, C)
(C, D)
```

只会影响局部：

```text
(A, B)
(B, C)
(C, D)
```

变成：

```text
(A, BC)
(BC, D)
```

远处的 pair 根本没有变化。

因此可以维护：

```text
pair
→ count

pair
→ affected pre-tokens
```

merge 后只更新受影响的位置。

这样就不需要：

```text
每次 merge
→ 全 corpus recount
```

这将成为之后真正 CS336 tokenizer optimization 的重点。

---

# 41. Tokenizer 的 Mental Model

最后把整个东西压缩成三层。

## Layer 1：Representation

```text
Unicode text
      ↓
UTF-8
      ↓
bytes
```

数学上：

$$
x
\rightarrow
(b_1,b_2,\ldots,b_n)
$$

---

## Layer 2：Compression

```text
bytes
↓
pre-tokenization
↓
pair statistics
↓
argmax pair
↓
BPE merges
↓
ordered merge rules
```

核心公式：

$$
\boxed{
(a^*,b^*)
=
\arg\max_{(a,b)}C(a,b)
}
$$

---

## Layer 3：Model Interface

```text
token bytes
     ↓
token IDs
     ↓
Tensor [B, T]
     ↓
Embedding
     ↓
[B, T, d_model]
     ↓
Transformer
```

即：

$$
\text{text}
\rightarrow
[T]
\rightarrow
[B,T]
\rightarrow
[B,T,d_{\text{model}}]
$$

---

# 42. Tokenizer 这一章真正需要记住的 8 件事

1. **语言模型处理的是 token ID，不是字符串。**

2. Unicode 文本首先可以表示成 UTF-8 bytes：

$$
\text{Unicode}
\rightarrow
\text{UTF-8 bytes}
$$

3. Byte-level tokenizer 用 $256$ 个初始 token 就可以覆盖任意 UTF-8 文本。

4. BPE training 的核心优化目标是：

$$
\boxed{
(a^*,b^*)
=
\arg\max C(a,b)
}
$$

5. BPE 学习到的不是单纯 vocabulary，而是：

$$
\boxed{
\text{Vocabulary}
+
\text{Ordered Merge Rules}
}
$$

6. Encoding 不是 greedy longest match，而是在重放 learned merge ranks。

7. Decode 必须：

```text
token IDs
→ token bytes
→ concatenate
→ UTF-8 decode
```

不能逐 token UTF-8 decode。

8. Tokenizer 会改变 sequence length $T$，所以会直接影响 Transformer 的：

$$
\text{FLOPs},
\quad
\text{memory},
\quad
\text{latency},
\quad
\text{context utilization}
$$

---

## 43. 这一章的掌握标准

在继续 Transformer 之前，我建议以这四项作为验收。

你应该能够不看笔记写出：

```python
count_pairs(...)
merge_pair(...)
```

能够手算：

$$
(a^*,b^*)
=
\arg\max C(a,b)
$$

能够解释为什么：

```text
BPE ≠ longest match
```

以及能够从头画出：

```text
"你好 🚀"
   ↓
Unicode
   ↓
UTF-8 bytes
   ↓
pre-tokenization
   ↓
byte-level BPE
   ↓
token IDs [T]
   ↓
batch [B, T]
   ↓
Embedding
   ↓
[B, T, d_model]
```

达到这里，Tokenizer 才算真正学完，而不是“会用 tokenizer”。

下一步就可以非常自然地进入：

```text
Token IDs [B, T]
        ↓
Embedding
        ↓
Linear
RMSNorm
SwiGLU
RoPE
Attention
        ↓
Transformer Block
        ↓
Transformer LM
```

这正对应你整条路线里 **BPE Tokenizer → Transformer From Scratch → Language Model Training** 的能力链。
