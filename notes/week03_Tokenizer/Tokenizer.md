# Tokenizer

## Tokenizer 解决什么问题?

语言模型不能直接接受 Python 的 `str` 类型, 如:

```text
"Hello, world!"
```

并且 Transformer 真正接受的是整数:

```text
[15496, 995]
```

所以 tokenizer 本质上实现两个方向:

```text
encode:
string -> token IDs

decode:
token IDs -> string
```

可以形式化理解为:
$$
T:String\to \mathbb{N}^*,\quad T^{-1}:\mathbb{N}^*\to Sring
$$

所以 Tokenizer 将 raw input/bytes 转化成整数 token 序列, 并可以再 decode 回去. 因此最终最基本的正确性条件是:

```Python
text == tokenizer.decode(tokenizer.encode(text))
```

也就是
$$
decode(encode(x))=x
$$

## 为什么不使用一个字符一个 token?

第一种最自然的想法是将其转换成字符, 然后每个字符对应一个 token, 也就是 **character tokenizer**, 即:

```text
Hello

h
e
l
l
o
```

Python 中可以理解为

```Python
ord("a") # 97
chr(97) # "a"
```

而问题在于 Unicode 的字符集非常庞大, 可能远超 256 个字符, 并且很多字符极其稀有. 于是出现两个坏处

- vocabulary 很大
- sequence 仍然很长

所以 character tokenizer 的词表很大, 并且压缩率又不好. 于是现代 tokenizer 通常不会直接使用 Unicode character, 而是转到更底层的

```text
bytes
```

## Unicode, UTF-8 和 bytes

首先要解释几个概念:

- **ASCII:** 使用 7 位(实际常存为 8 位, 最高位为 0)表示 128 个字符包括英文字母(A-Z, a-z), 数字(0-9), 标点符号以及控制字符(如换行 `\n`, 回车 `\r`). 每个字符固定占 1 个字节. 但是只能表示英语和少量符号, 无法表示中文, 日文, emoji 等.

- **Unicode character(Unicode 字符):** 一个抽象的, 与平台无关的文本单位. 它只回答"这是什么字", 不涉及任何存储细节.
- **Unicode code point(Unicode 码点):** Unicode 标准给每个 Unicode character 分配的唯一整数编号. 数学格式为 U+ 前缀加十六进制, 范围从 `U+0000` 到 `U+10FFFF`(共约111万个码位).
>
> 注: 码点仍然是抽象编号, 不是内存里的字节
>
- **UTF-8 encoding(UTF-8 编码):** 一种编码方式, 负责将抽象的码点映射成具体的字节序列. 它是变长编码:
>
> - U+0000-U+007F(ASCII 字符) -> 1 byte
> - U+0080-U+07FF -> 2 bytes
> - U+0800-U+FFFF(含常用中文) -> 3 bytes
> - U+10000-U+10FFFF(含 emoji) -> 4 bytes
> 关键设计: 与 ASCII 完全兼容, ASCII 字符的 UTF-8 编码是就是它原来的单字节值.
>
- **bytes(字节):** 计算机实际存储和传输的最小可寻址单位, 每个字节是8位, 取值范围 `0x00-0xFF`(十进制为 0-255). 文本文件里存的, 网络上传输的, tokenizer 实际处理的, 都是字节, 而不是"字符".

然后要将下面三层分开:

```text
Unicode character -> Unicode code point -> UTF-8 encoding -> bytes
```

例如:

```Python
text="a"
text.encode("utf-8") # b'a'
```

这里的一个 ASCII 字符只需要一个 byte. 但是有的字符需要多个 byte, 例如:

```Python
"🌍".encode("utf-8") # f0 9f 8c 8d
```

此例子说明: UTF-8 中的一个Unicode character 不一定对应一个 byte. 因此要注意
> byte $\neq$ character. 有些 UTF-8 字符必须将多个 bytes 放在一起才能 decode.

## 为什么要从 bytes 开始?

byte 只有
$$
2^{8}=256
$$
种可能. 所以可以天然建立一个完整基础词表:

```text
0    -> b"\x00"
1    -> b"\x01"
...
97    -> b"a"
...
255    -> b"\xff"
```

即

```Python
vocab = {i: bytes([i]) for i in range(256)}
```

它理论上有一个极大的优势
> 理论上任何 UTF-8 文本都能表示, 不再需要传统 word tokenizer 的 `<UNK>` 问题.

但是纯 byte tokenizer 也有缺点. 假设输入 1000 bytes, 则纯 byte tokenizer 会大约产生 1000 个 token. 也就是
$$
bytes/token = 1
$$
而 Transformer attention 的计算又强烈依赖 sequence length. 所以这很昂贵.

## Tokenizer 最核心的 trade-off

Tokenizer 一直在平衡 `Voabulary Size` 和 `Sequence Length` 两个方面.

1. **Vocabulary 小**. 例如 256 byte tokens
    - 优点: embedding / LM head 小; 所有文本都能表示.
    - 缺点: sequence 非常长
2. **Vocabulary 大**. 例如直接把大量单词当作 token, sequence 会短很多, 但是会出现:
    - embedding matrix 大
    - LM head 大
    - softmax 昂贵
    - 大量的 rare words
    - OOV 问题

所以需要一个中间方案, **BPE(Byte Pair Encoding)**:
> 高频字符串 -> 一个 token
>
> 低频字符串 -> 多个 token

## Compression Ratio(压缩率)

定义
$$
\text{Compression Ratio} = \frac{\text{number of bytes}}{\text{number of tokens}}
$$
Compression Ratio 越大, 通常说明:

```text
同样的文本 -> token 更少 -> sequence length 更短
```

> 课程中也称为 **bytes per token**, 并指出提高它可以缩短 Transformer 处理的序列, 但通常又会提高 vocabulary size.

## BPE 的核心思想

BPE 的核心思想是:

```text
从 bytes 开始
↓
寻找最经常一起出现的相邻 token Pair
↓
合并相邻 token Pair 为一个新的 token
↓
重复上述过程
```

### 以 "banana" 为例

假设我们从 "banana" 开始, 则为

```text
"banana"
↓
b a n a n a
↓
若 (a, n) 出现最频繁, 创建 AN=b"an"
↓
b AN AN a
↓
若 (AN, a) 出现最频繁, 创建 ANA=b"ana"
↓
继续 merge.
```

BPE 的 token 粒度不是人为定义的, 而是由训练数据决定. 课程中对 BPE 的直觉描述是:
> common byte sequence -> one token
>
> rare byte sequence -> many tokens

## BPE 的两个完全不同阶段

BPE 有两个阶段, 分别是

1. **Training the tokenizer**
2. **Using the tokenizer**

### Training the tokenizer

```text
输入: training corpus, vocab_size, special_tokens
↓
输出: vocab, merges
```

官方要求大致是:

```Python
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

其中 `merges` 的顺序就是 merge 被学习出来的顺序. 例如

```text
merge #0: (b"t", b"h")
merge #1: (b"th", b"e")
merge #2: (b"i", b"n")
```

于是有

```text
rank(("t","h'))=0
rank(("th","e"))=1
rank(("i","n"))=2
```

此顺序在后续的 encode 时还要使用.

## BPE Training 的完整逻辑

完整的 BPE Training 逻辑如下:

```text
Corpus
↓
remove / isolate special tokens
↓
pre-tokenization
↓
UTF-8 bytes
↓
统计 pre-token frequencies
↓
初始化 byte vocabulary
↓
统计 adjacent pair frequencies
↓
选择最高频率的 pair
↓
merge(合并)
↓
更新 vocabulary + merge list
↓
重新更新 pair frequencies
↓
直到 vocab_size
```

### 初始化 vocabulary

byte-level BPE 天然有 256 个 byte token. 然后加入 special tokens, 例如:

```text
<|endoftext|>
```

之后不断增加 merge token. 如果目标 `vocab_size=10000`, 最终 vocabulary 的总大小应该是 10000, 而不是10000+256.
> `vocab_size` 是包含在 special tokens 在内的最终总词表大小.

### Pre-tokenization

>相比于 tokenizers 来说，pre_tokenizers 是相对而言更加简单更加容易理解的，预分词的作用，就是根据一组规则对输入的文本进行分割，这种预处理是为了确保模型不会在多个“分割”之间构建tokens。
>
>比如如果不进行预分词，而是直接进行分词，那么可能出现这种情况："您好 人没了" -> "您" "好 人" "没了"。
>
>也就是说分词有可能会产生这种与我们日常经验相悖的分词效果，而预分词就可以有效地避免这一点，比如在分词前，先在使用预分词在空格上进行分割："您好 人没了" -> "您好" "人没了"，再进行分词："您好" "人没了" -> "您好" "人" "没了"。
>
>---
>
>转自[tokenizers.pre_tokenizers | 预分词方法介绍]([(1 封私信) tokenizers.pre_tokenizers | 预分词方法介绍 - 知乎](https://zhuanlan.zhihu.com/p/692508797))

不能简单将整个 corpus 的所有 bytes 连成一条序列, 然后随便 merge. 这样可能会学出非常奇怪的跨边界 token. 例如:

```text
hello!world
↓
若完全不做限制: o!w. 甚至: !world...
```

都有可能跨越不希望出现的边界进行 merge. 所以要先进行 pre-tokenization, 将原始文本分割成 `pretokens`. 例如概念上可能得到:

```text
"Hello"
","
" world"
"!"
```

然后 BPE merge 只会发生在一个 pre-token 内部. 不会跨边界进行 merge. 以后在检查 tokenizer 时经常会看到

```text
"Hello"
","
" world"
```

而不是

```text
"Hello"
","
" "
"world"
```

这是因为 **preceding space(前导空格)** 可以成为 token 的一部分.

### Pair Counting

假设当前的 pretoken 表示为

```text
[a,b,a,b]
```

adjacent pairs 为:

```text
(a,b)
(b,a)
(a,b)
```

于是有:

```text
count(a,b)=2
count(b,a)=1
```

BPE要找
$$
\operatorname{arg\,max}_{(x,y)}count(x,y)
$$
然后进行 merge. 而 toy 版通常每轮都

```text
扫描整个数据 -> 统计所有 pair
```

但这是非常慢的 baseline.

### Tie-breaking

假设

```text
(b"a",b"b") -> 100
(b"x",b"y") -> 100
```

两个的 pair frequency 完全相同, 问题是选择哪个?

本节课程的 reference behavior 需要 deterministic tie-breaking:

> **deterministic tie-breaking:** 当 BPE 训练过程中有多个相邻 token pair 的统计频次并列最高时, 采用一条事先规定好的, 无随机性的规则, 从这些并列候选中选出唯一的一个进行合并.
>
> 本节课程选择的是 **lexicographically greater** 的 pair, 并且比较的是原始的 byte tuples.

所以后续不能只写

```Python
max(pair_counts, key=pair_counts.get)
```

然后完全依赖 dictionary insertion order, 否则结果有可能不等于参考的 merges.

### Merge

流程为:

```text
A = b"t"
B = b"h"
↓ merge
(A,B)
↓
新 token: C = A + B = b"th"
```

所以 vocabulary 的本质为:

```text
每个 token = 某一段 bytes
```

因此有:
$$
vocab[new] = vocab[left] + vocab[right]
$$
> 这是 byte-level BPE 最核心的数据结构关系.

### Special Tokens

BPE 通常会加入一些特殊 tokens, 例如 `<|endoftext|>`, 这永远是一个 token, 而不会被拆成更小的 token.

1. **Training 阶段:**special token 不应该参与普通的 BPE merge.

> 并且在 pre-tokenization 前先按 special token 进行切分, 以防止其进入 BPE merge.

1. **Encoding 阶段:** 在编码时, 若输入 `hello<|endoftext|>world`, 应该输出

    ```text
    norm BPE + special token + norm BPE
    ```

    而不是让 special token 进入普通的 merge 流程.

### Overlapping Special Tokens

假设 special tokens 为:

```text
<|endoftext|>

<|endoftext|><|endoftext|>
```

输入:

```text
<|endoftext|><|endoftext|>
```

应该识别第二个更长的完整 special token, 而不是首先拆成两个短 token.
> 课程中所使用的是**最长匹配优先(Longest Match First)**, 具体是: 在执行 `re.split` 或正则匹配之前, 先将所有特殊标记按长度降序排序。这样, 正则引擎会优先尝试匹配更长的标记.
>
>所以在后续实现 special-token matching 时, 要考虑 `overlap + Longest match` 问题.
>
## Tokenizer Training 和 Encoding 的区别

Training:

```text
corpus
↓
统计 frequency
↓
学习 merges

输出: vocab, merges
```

Encoding:

```text
new text + 训练好的 vocab + 训练好的 merges
↓
token IDs
```

> Encoding 绝对不会重新统计语料 frequency.

## Encoding 流程

假设训练出的 merge order:

```text
1. (t, h )
2. (th, e)
3. (i, n)
```

开始 encode:

```text
"the"
↓
t h e
↓ (t, h)
th e
↓ (th, e)
the
```

最终得到一个token.
> 关键是: merge 的优先级来自 training 时产生 merge 的先后顺序.

可以建立:

```Python
merge_rank = {pair:rank}
```

概念上有:

```text
当前所有 adjacent pairs
↓
找存在于 merge_rank 中的 pair
↓
选择 rank 最小, merge, 然后重复
```

## Decode 流程

整体流程为:

```text
token IDs: [100, 2056, 78]
↓
bytes, 进行 bytes 拼接
↓
UTF-8 解码

ids
↓
vocab[id]
↓
b"".join(...)
↓
.decode("utf-8")
↓
str
```

上述不要对每个 token 单独 解码, 因为一个 token 的 bytes 单独看不一定构成完整合法的 UTF-8 character. Decode 的正确想法是:
> 先恢复整个 byte stream, 再做 UTF-8 解码.

## Training 的性能问题

最一般的 BPE:

```text
for every merge:
    扫描整个 corpus
    统计每个 pair 的频率
    找到频率的最大值
    merge 整个 corpus
```

假设:

```text
N =  corpus token 数
M = merge 次数
```

则粗略计算为
$$
O(NM)
$$
但是实际上大量的 Python object/dict 操作会更加慢. 所以后续性能优化也是一部分的工作.
