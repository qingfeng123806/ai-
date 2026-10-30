"""文本处理工具：规范化、分句、分词、重合度、引用解析、安全检查。

全部基于标准库实现，中英文混排场景通用。这些函数是检索、评测、报告三个
子系统共用的底层原语，保持无副作用、可单测。
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Iterable, Iterator, Sequence

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*|\d+(?:\.\d+)?")
SENT_END = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+(?=[A-Z0-9\u4e00-\u9fff])|\n+")
WS = re.compile(r"[ \t\u00a0\u3000]+")

ZH_STOPWORDS = frozenset(
    """的 了 是 在 和 与 或 及 等 有 我 你 他 她 它 我们 你们 他们 这 那 这个 那个 这些 那些
    个 中 上 下 之 也 很 到 说 要 会 对 能 可以 就 都 而 被 把 给 让 从 向 于 为 以 因 由
    吗 呢 吧 啊 呀 哦 嗯 么 着 过 得 地 其 该 此 并 且 但 却 则 若 如 若 何 什么 怎么 如何
    为什么 哪些 哪个 多少 怎样 是否 一下 一个 一些 这样 那样 以及 或者 不过 就是 还是
    请问 想 需要 应该 可能 已经 还是 因为 所以 如果 没有 不是 不能 不会""".split()
)

EN_STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from as is are was were be
    been being do does did not no nor so than that this these those it its i you he she they we
    me my your his her their our them us can could should would may might must will shall have
    has had about into over under again further once here there when where why how what which
    who whom whose all any both each few more most other some such only own same too very just
    dont don't doesnt doesn't im i'm""".split()
)

_ALL_STOP = ZH_STOPWORDS | EN_STOPWORDS
_CJK_STOP_CHARS = frozenset("的了吗呢吧啊呀哦嗯么着过地得")

# 提示注入 / 越狱特征
INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"忽略(?:掉)?\s*(?:之前|以上|前面|上述|你之前|所有)[^。！？\n]{0,10}(?:指令|提示|要求|规则|设定)", "ignore_instructions"),
    (r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?(?:previous|above|prior|earlier)\s+(?:instructions?|prompts?|rules?)", "ignore_instructions"),
    (r"(?:你|从现在起|现在)\s*(?:是|扮演|充当)\s*(?:一个)?\s*(?:管理员|开发者|系统|DAN|无限制)", "role_hijack"),
    (r"(?:输出|打印|显示|重复|告诉我)[^。！？\n]{0,20}(?:提示词|prompt|设定|指令|system)", "prompt_leak"),
    (r"(?:reveal|show|print|repeat)[^.\n]{0,20}(?:system\s+prompt|instructions?)", "prompt_leak"),
    (r"越狱|jailbreak|DAN\s*模式|开发者模式", "jailbreak"),
    (
        r"(?:密钥|密码|私钥|凭据|credential|api\s*key|access\s*key|token|数据库密码)[^。！？\n]{0,14}(?:给我|提供|发我|告诉|输出|交出来)",
        "credential_probe",
    ),
    (
        r"(?:给我|提供|发我|告诉|输出|交出)[^。！？\n]{0,14}(?:密钥|密码|私钥|凭据|api\s*key|token|数据库密码)",
        "credential_probe",
    ),
    (r"<\s*/?\s*(?:system|assistant|tool)\s*>", "role_markup"),
    (r"忽略[^。！？\n]{0,6}(?:安全|合规|审核|限制|策略)", "policy_bypass"),
    (
        r"(?:手机号|身份证|住址|家庭住址|银行卡号|薪资|工资条)[^。！？\n]{0,12}(?:查一下|查询|是多少|告诉我|给我|发我|查查)",
        "pii_probe",
    ),
    (
        r"(?:查一下|查询|查查|告诉我|给我|发我)[^。！？\n]{0,12}(?:手机号|身份证|住址|银行卡|个人信息)",
        "pii_probe",
    ),
)

# 个人敏感信息
PII_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("phone", r"(?<!\d)1[3-9]\d{9}(?!\d)", "手机号"),
    ("email", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "邮箱"),
    ("id_card", r"(?<!\d)\d{17}[\dXx](?!\d)", "身份证号"),
    ("bank_card", r"(?<!\d)\d{16,19}(?!\d)", "银行卡号"),
    ("ipv4", r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", "IP 地址"),
)

TOXIC_TERMS = ("傻逼", "去死", "滚蛋", "智障", "脑残", "fuck", "shit", "idiot")

REFUSAL_MARKERS = (
    "无法回答", "不能回答", "无法确定", "没有找到", "未找到", "资料中没有", "文档中没有",
    "知识库中没有", "不在知识库", "无法从", "恕难", "抱歉", "无法提供", "无法确认",
    "无法执行", "不能执行", "无法协助", "无法满足", "不予提供", "没有足够",
    "i cannot", "i can't", "i don't know", "not enough information", "unable to",
    "no information", "cannot answer",
)

CITATION_MARK = re.compile(r"\[(\d{1,2})\]")


# --------------------------------------------------------------------------
# 基础规范化
# --------------------------------------------------------------------------

def normalize_ws(text: str) -> str:
    """压缩行内空白，保留段落换行（至多一个空行）。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WS.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip()


def short_hash(value: str, size: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:size]


def stable_rng(*parts: object) -> random.Random:
    """由给定种子派生稳定的随机数发生器（跨进程可复现）。"""
    key = "|".join(str(p) for p in parts)
    return random.Random(int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


# --------------------------------------------------------------------------
# 分句与分词
# --------------------------------------------------------------------------

def sentences(text: str) -> list[str]:
    """中英文分句，过滤空串与纯标点片段。"""
    if not text:
        return []
    parts = SENT_END.split(text)
    out: list[str] = []
    for part in parts:
        if part is None:
            continue
        s = part.strip()
        if len(s) < 2:
            continue
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", s):
            continue
        out.append(s)
    return out


def cjk_terms(run: str, bigrams: bool = True) -> list[str]:
    """把一段连续中文切成 bigram（长度不足时退化为 unigram）。"""
    if not run:
        return []
    if len(run) == 1:
        return [] if run in _CJK_STOP_CHARS else [run]
    out: list[str] = []
    if bigrams:
        for i in range(len(run) - 1):
            pair = run[i : i + 2]
            if pair[0] in _CJK_STOP_CHARS or pair[1] in _CJK_STOP_CHARS:
                continue
            out.append(pair)
    for ch in run:
        if ch not in _CJK_STOP_CHARS:
            out.append(ch)
    return out


def terms(text: str) -> list[str]:
    """内容词序列：英文小写词 + 中文 unigram/bigram，去停用词。"""
    if not text:
        return []
    out: list[str] = []
    for match in LATIN_WORD.finditer(text):
        word = match.group(0).lower()
        if len(word) < 2 and not word.isdigit():
            continue
        if word in EN_STOPWORDS:
            continue
        out.append(word)
    for run in CJK_RUN.findall(text):
        out.extend(cjk_terms(run))
    return out


def term_set(text: str) -> set[str]:
    return set(terms(text))


def keywords(text: str, top_n: int = 8) -> list[str]:
    """基于词频的简易关键词抽取（用于合成测试集与标签）。"""
    counts: dict[str, int] = {}
    for tok in terms(text):
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [tok for tok, _ in ranked[:top_n]]


def char_ngrams(text: str, low: int = 2, high: int = 3) -> list[str]:
    """字符 n-gram（去除空白），用于哈希向量化。"""
    compact = re.sub(r"\s+", "", text.lower())
    out: list[str] = []
    for n in range(low, high + 1):
        for i in range(len(compact) - n + 1):
            out.append(compact[i : i + n])
    return out


# --------------------------------------------------------------------------
# 重合度 / 覆盖率
# --------------------------------------------------------------------------

def coverage(needles: Iterable[str], haystack: set[str], weights: dict[str, float] | None = None) -> float:
    """needles 中被 haystack 覆盖的比例（可按 idf 加权）。"""
    items = list(needles)
    if not items:
        return 0.0
    if not haystack:
        return 0.0
    hit = 0.0
    total = 0.0
    for item in items:
        w = (weights or {}).get(item, 1.0)
        total += w
        if item in haystack:
            hit += w
    return clamp(safe_div(hit, total))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return clamp(safe_div(inter, union))


def containment(needles: set[str], haystack: set[str]) -> float:
    if not needles:
        return 0.0
    return clamp(safe_div(len(needles & haystack), len(needles)))


def build_idf(documents: Sequence[str]) -> dict[str, float]:
    """在语料上估计 idf，供加权覆盖率 / 离线判定使用。"""
    df: dict[str, int] = {}
    n = max(1, len(documents))
    for doc in documents:
        for tok in set(terms(doc)):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log(1.0 + n / (1.0 + c)) for tok, c in df.items()}


def idf_weight(tok: str, idf: dict[str, float] | None) -> float:
    if not idf:
        return 1.0
    return max(0.05, idf.get(tok, 1.0))


# --------------------------------------------------------------------------
# 引用 / 去重 / 拒答
# --------------------------------------------------------------------------

def extract_citations(text: str) -> list[int]:
    return [int(m.group(1)) for m in CITATION_MARK.finditer(text or "")]


def strip_citations(text: str) -> str:
    return CITATION_MARK.sub("", text or "").strip()


def citation_sentences(answer: str) -> list[tuple[str, list[int]]]:
    """把答案拆成 (句子, 该句引用编号列表)。"""
    out: list[tuple[str, list[int]]] = []
    for sent in sentences(answer):
        out.append((sent, extract_citations(sent)))
    return out


def is_refusal(answer: str) -> bool:
    """判断答案是否为拒答：短答案命中拒答词，或开头即声明无法回答。"""
    text = (answer or "").strip().lower()
    if not text:
        return True
    if any(marker in text for marker in REFUSAL_MARKERS) and len(text) <= 80:
        return True
    head = text[:120]
    return any(marker in head for marker in REFUSAL_MARKERS)


def dedupe_sentences(text: str, threshold: float = 0.86) -> str:
    """删除近似重复句子，抑制“重复回答”类问题。"""
    seen: list[set[str]] = []
    kept: list[str] = []
    for sent in sentences(text):
        sig = term_set(sent)
        if any(jaccard(sig, prev) >= threshold for prev in seen):
            continue
        seen.append(sig)
        kept.append(sent)
    if not kept:
        return text.strip()
    joined = "".join(kept) if _needs_cjk_join(kept) else " ".join(kept)
    return joined


def _needs_cjk_join(parts: Sequence[str]) -> bool:
    return bool(parts) and all(part[-1] in "。！？；" for part in parts) and not any(
        part[-1] in ".!?" for part in parts
    )


def detect_injection(text: str) -> list[str]:
    hits: list[str] = []
    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, text or "", flags=re.IGNORECASE):
            hits.append(label)
    return hits


def scan_pii(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for kind, pattern, label in PII_PATTERNS:
        for match in re.finditer(pattern, text or ""):
            found.append({"type": kind, "label": label, "value": _mask(match.group(0))})
    return found


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def toxicity_hits(text: str) -> list[str]:
    low = (text or "").lower()
    return [term for term in TOXIC_TERMS if term in low]


def contains_any(text: str, needles: Iterable[str]) -> bool:
    low = (text or "").lower()
    return any(n.lower() in low for n in needles if n)


def keyword_coverage(expected: Sequence[str], answer: str) -> float:
    """期望要点在答案中的覆盖率（支持同义扩展由调用方预置）。"""
    if not expected:
        return 1.0
    ans_terms = term_set(answer)
    ans_low = (answer or "").lower()
    hit = 0
    for kw in expected:
        if kw.lower() in ans_low or term_set(kw) <= ans_terms:
            hit += 1
    return clamp(safe_div(hit, len(expected)))


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def iter_lines(text: str) -> Iterator[str]:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            yield line
