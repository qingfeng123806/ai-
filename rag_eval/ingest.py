"""文档加载与切片：4 种切片策略 + 多格式加载器。

切片策略影响召回粒度与上下文完整性，是 A/B 实验中最常见的变量之一，因此
四种策略共享同一输出契约（``Chunk``），可在实验矩阵里任意切换：

- ``fixed``：固定字符窗口 + 重叠，实现最简单、行为可预测；
- ``recursive``：按段落/句子/标点递归切分后贪心合并，尽量不切断语义；
- ``heading``：先按 Markdown 标题分节，再在节内递归切分，保留章节路径；
- ``semantic``：按相邻句子向量相似度断点，主题漂移处切分。
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import IngestCfg
from .embeddings import HashingEmbedder, cosine
from .text import normalize_ws, sentences, short_hash, truncate
from .types import Chunk, Document


class LoaderError(RuntimeError):
    """文档无法解析（依赖缺失或格式损坏）。"""


# --------------------------------------------------------------------------
# 加载器
# --------------------------------------------------------------------------

class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return html.unescape("".join(self.parts))


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def _html_to_text(raw: str) -> str:
    parser = _HTMLText()
    parser.feed(raw)
    return parser.text()


def _pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as exc:
            raise LoaderError(
                f"解析 PDF 需要 pypdf：{path.name}。请执行 pip install pypdf，或将文档转为 Markdown/TXT。"
            ) from exc
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(f"\n\n## 第 {i + 1} 页\n\n" + (page.extract_text() or ""))
        except Exception:
            continue
    return "".join(pages)


def _docx_to_text(path: Path) -> str:
    try:
        import docx  # type: ignore
    except Exception as exc:
        raise LoaderError(
            f"解析 DOCX 需要 python-docx：{path.name}。请执行 pip install python-docx。"
        ) from exc
    document = docx.Document(str(path))
    parts: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        if style.startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "1"
            parts.append("#" * min(6, int(level)) + " " + text)
        else:
            parts.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines()[:30]:
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        raw = _read_text(path)
        text = raw
    elif suffix in (".html", ".htm"):
        raw = _html_to_text(_read_text(path))
        text = raw
    elif suffix == ".pdf":
        text = _pdf_to_text(path)
    elif suffix == ".docx":
        text = _docx_to_text(path)
    elif suffix == ".json":
        text = _read_text(path)
    else:
        raise LoaderError(f"不支持的文档格式：{suffix}（{path.name}）")
    text = normalize_ws(text)
    if not text:
        raise LoaderError(f"文档内容为空：{path}")
    title = _extract_title(text, path.stem)
    doc_id = short_hash(str(path.resolve()).lower() + "|" + str(len(text)), 12)
    return Document(doc_id=doc_id, title=title, source=str(path.as_posix()), text=text, meta={"suffix": suffix})


def load_corpus(paths: Sequence[str], include_globs: Sequence[str] | None = None, root: Path | None = None) -> list[Document]:
    """加载目录或文件列表，返回去重后的文档集合。"""
    root = root or Path.cwd()
    globs = list(include_globs or ["*.md", "*.markdown", "*.txt", "*.html", "*.htm", "*.pdf", "*.docx"])
    files: list[Path] = []
    for item in paths:
        p = Path(item)
        if not p.is_absolute():
            p = root / p
        if p.is_dir():
            for pattern in globs:
                files.extend(sorted(p.rglob(pattern)))
        elif p.is_file():
            files.append(p)
        else:
            raise LoaderError(f"语料路径不存在：{p}")
    seen: set[Path] = set()
    docs: list[Document] = []
    errors: list[str] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            docs.append(load_document(resolved))
        except LoaderError as exc:
            errors.append(str(exc))
    if errors and not docs:
        raise LoaderError("；".join(errors))
    return docs


# --------------------------------------------------------------------------
# 切片
# --------------------------------------------------------------------------

@dataclass
class ChunkStats:
    documents: int = 0
    chunks: int = 0
    chars: int = 0
    avg_chars: float = 0.0
    min_chars: int = 0
    max_chars: int = 0
    chunker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "chars": self.chars,
            "avg_chars": round(self.avg_chars, 1),
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "chunker": self.chunker,
        }


def _finalize(doc: Document, pieces: Iterable[tuple[str, int, int, list[str]]], chunker: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, (text, start, end, headings) in enumerate(pieces):
        text = text.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{idx:03d}",
                doc_id=doc.doc_id,
                title=doc.title,
                source=doc.source,
                text=text,
                index=idx,
                start=max(0, start),
                end=max(0, end),
                meta={"headings": headings, "chunker": chunker},
            )
        )
    return chunks


def _snap_end(text: str, end: int, limit: int) -> int:
    """把窗口右边界推进到最近的句末，避免答案被截断在半句话。"""
    window = text[end : min(len(text), end + 40)]
    best = -1
    for mark in "。！？；\n.!?;":
        pos = window.find(mark)
        if pos >= 0 and (best < 0 or pos < best):
            best = pos
    if best >= 0:
        candidate = end + best + 1
        if candidate <= limit:
            return candidate
    return end


def chunk_fixed(doc: Document, cfg: IngestCfg) -> list[Chunk]:
    text = doc.text
    size = max(50, cfg.chunk_size)
    overlap = max(0, min(cfg.chunk_overlap, size // 2))
    step = max(1, size - overlap)
    pieces: list[tuple[str, int, int, list[str]]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        if end < n:
            end = _snap_end(text, end, min(n, start + size + 40))
        piece = text[start:end]
        if len(piece.strip()) >= cfg.min_chunk_chars or not pieces:
            pieces.append((piece, start, end, []))
        if end >= n:
            break
        start = max(start + step, end - overlap) if end - start > overlap else end
    return _finalize(doc, pieces, "fixed")


def _split_recursive(text: str, separators: Sequence[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), max(1, size))]
    sep = separators[0]
    rest = separators[1:]
    if sep == "":
        parts = [text[i : i + size] for i in range(0, len(text), max(1, size))]
        return parts
    pieces = text.split(sep)
    out: list[str] = []
    for i, piece in enumerate(pieces):
        chunk = piece + (sep if i < len(pieces) - 1 else "")
        if not chunk:
            continue
        if len(chunk) <= size:
            out.append(chunk)
        else:
            out.extend(_split_recursive(chunk, rest, size))
    return out


def chunk_recursive(doc: Document, cfg: IngestCfg) -> list[Chunk]:
    size = max(80, cfg.chunk_size)
    separators = ["\n\n", "\n", "。", "！", "？", "；", ". ", "，", " ", ""]
    atoms = _split_recursive(doc.text, separators, size)
    merged: list[str] = []
    buf = ""
    for atom in atoms:
        if not buf:
            buf = atom
        elif len(buf) + len(atom) <= size:
            buf += atom
        else:
            merged.append(buf)
            tail = buf[-min(cfg.chunk_overlap, len(buf) // 2) :] if cfg.chunk_overlap else ""
            buf = tail + atom if tail and len(tail + atom) <= size * 1.5 else atom
    if buf.strip():
        merged.append(buf)
    pieces: list[tuple[str, int, int, list[str]]] = []
    cursor = 0
    for piece in merged:
        text = piece.strip()
        if not text:
            continue
        if len(text) < cfg.min_chunk_chars and pieces:
            prev_text, prev_start, _, prev_head = pieces[-1]
            pieces[-1] = (prev_text + "\n" + text, prev_start, cursor + len(piece), prev_head)
            continue
        start = doc.text.find(text[: min(40, len(text))], cursor)
        start = cursor if start < 0 else start
        end = start + len(text)
        pieces.append((text, start, end, []))
        cursor = max(cursor, end)
    return _finalize(doc, pieces, "recursive")


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_heading(doc: Document, cfg: IngestCfg) -> list[Chunk]:
    lines = doc.text.splitlines()
    sections: list[tuple[list[str], list[str]]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    heading_started = False
    for line in lines:
        match = _HEADING_RE.match(line.strip())
        if match:
            if heading_started and buffer:
                sections.append(([h for _, h in stack], buffer))
                buffer = []
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            heading_started = True
            buffer.append(line)
        else:
            buffer.append(line)
    if buffer:
        sections.append(([h for _, h in stack], buffer))

    pieces: list[tuple[str, int, int, list[str]]] = []
    cursor = 0
    for headings, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        prefix = "【" + " / ".join(headings) + "】\n" if headings and cfg.heading_aware else ""
        body_with_prefix = prefix + body
        parts = _split_recursive(body_with_prefix, ["\n\n", "\n", "。", "；", "，", ""], max(120, cfg.chunk_size))
        buf = ""
        for part in parts:
            if not buf:
                buf = part
            elif len(buf) + len(part) <= cfg.chunk_size:
                buf += part
            else:
                text = buf.strip()
                if text:
                    start = doc.text.find(text[-min(30, len(text)) :], max(0, cursor - 200))
                    start = cursor if start < 0 else max(0, start - max(0, len(text) - 30))
                    pieces.append((text, start, start + len(text), headings))
                    cursor = start + len(text)
                buf = part
        text = buf.strip()
        if text:
            start = doc.text.find(text[-min(30, len(text)) :], max(0, cursor - 200))
            start = cursor if start < 0 else max(0, start - max(0, len(text) - 30))
            pieces.append((text, start, start + len(text), headings))
            cursor = start + len(text)
    return _finalize(doc, pieces, "heading")


def chunk_semantic(doc: Document, cfg: IngestCfg) -> list[Chunk]:
    """按相邻句子相似度切分：相似度跌破阈值处视为主题漂移点。"""
    sents = sentences(doc.text)
    if not sents:
        return chunk_heading(doc, cfg)
    embedder = HashingEmbedder(dim=256)
    vectors = embedder.embed_documents(sents)
    groups: list[list[int]] = [[0]]
    for i in range(1, len(sents)):
        sim = cosine(vectors[i - 1], vectors[i])
        current_len = sum(len(sents[j]) for j in groups[-1])
        if sim < cfg.semantic_threshold or current_len >= cfg.chunk_size:
            groups.append([i])
        else:
            groups[-1].append(i)
    pieces: list[tuple[str, int, int, list[str]]] = []
    cursor = 0
    for group in groups:
        text = "".join(sents[i] for i in group).strip()
        if not text:
            continue
        if len(text) < cfg.min_chunk_chars and pieces:
            prev_text, prev_start, _, prev_headings = pieces[-1]
            pieces[-1] = (prev_text + text, prev_start, cursor + len(text), prev_headings)
            continue
        start = doc.text.find(text[: min(30, len(text))], cursor)
        start = cursor if start < 0 else start
        pieces.append((text, start, start + len(text), []))
        cursor = max(cursor, start + len(text))
    return _finalize(doc, pieces, "semantic")


CHUNKERS = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "heading": chunk_heading,
    "semantic": chunk_semantic,
}


def chunk_documents(docs: Sequence[Document], cfg: IngestCfg) -> list[Chunk]:
    fn = CHUNKERS.get(cfg.chunker)
    if fn is None:
        raise ValueError(f"未知切片策略 {cfg.chunker}，可用：{sorted(CHUNKERS)}")
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(fn(doc, cfg))
    # 超长切片兜底截断，避免单块撑爆上下文预算
    for chunk in chunks:
        if len(chunk.text) > cfg.max_chunk_chars:
            chunk.text = truncate(chunk.text, cfg.max_chunk_chars)
    return chunks


def corpus_signature(docs: Sequence[Document], cfg: IngestCfg) -> str:
    """语料 + 切片参数的内容指纹，用于判断索引是否需要重建。"""
    hasher = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        hasher.update(doc.doc_id.encode())
        hasher.update(str(len(doc.text)).encode())
    hasher.update(
        f"{cfg.chunker}|{cfg.chunk_size}|{cfg.chunk_overlap}|{cfg.min_chunk_chars}|{cfg.heading_aware}|{cfg.semantic_threshold}".encode()
    )
    return hasher.hexdigest()[:16]


def ingest_stats(docs: Sequence[Document], chunks: Sequence[Chunk], chunker: str = "") -> ChunkStats:
    sizes = [len(c.text) for c in chunks] or [0]
    return ChunkStats(
        documents=len(docs),
        chunks=len(chunks),
        chars=sum(sizes),
        avg_chars=sum(sizes) / len(sizes),
        min_chars=min(sizes),
        max_chars=max(sizes),
        chunker=chunker,
    )
