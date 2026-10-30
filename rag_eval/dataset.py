"""测试集管理：多格式加载、拆分统计、去重合并、子集抽样。

支持 JSONL / JSON / CSV；一个「套件」可以是单文件，也可以是目录（递归加载）。
用例的 ``split`` 字段决定它进入核心回归集、边缘场景集还是对抗集；
若文件本身按拆分命名（core.jsonl / edge.jsonl / adversarial-*.jsonl）也会自动推断。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import ConfigError
from .text import short_hash, stable_rng, truncate
from .types import SPLITS, SPLIT_LABELS, CaseSpec


class DatasetError(ValueError):
    """测试集文件缺失或格式不合法。"""


def _infer_split(path: Path) -> str:
    name = path.stem.lower()
    if "adv" in name or "attack" in name or "安全" in name:
        return "adversarial"
    if "edge" in name or "边界" in name:
        return "edge"
    if "core" in name or "regression" in name or "回归" in name:
        return "core"
    return "core"


def _load_jsonl(path: Path) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    default_split = _infer_split(path)
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}:{lineno} JSON 解析失败：{exc}") from exc
        if not isinstance(data, dict):
            raise DatasetError(f"{path.name}:{lineno} 期望 JSON 对象")
        if "split" not in data:
            data["split"] = default_split
        try:
            cases.append(CaseSpec.from_dict(data))
        except ValueError as exc:
            raise DatasetError(f"{path.name}:{lineno} {exc}") from exc
    return cases


def _load_json(path: Path) -> list[CaseSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("cases") or payload.get("data") or []
    else:
        items = payload
    if not isinstance(items, list):
        raise DatasetError(f"{path.name} 期望用例数组")
    default_split = _infer_split(path)
    cases: list[CaseSpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item.setdefault("split", default_split)
        cases.append(CaseSpec.from_dict(item))
    return cases


def _load_csv(path: Path) -> list[CaseSpec]:
    default_split = _infer_split(path)
    cases: list[CaseSpec] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            keywords = [k.strip() for k in (row.get("expected_keywords") or "").split(";") if k.strip()]
            contexts = [c.strip() for c in (row.get("reference_contexts") or "").split("||") if c.strip()]
            cases.append(
                CaseSpec(
                    case_id=(row.get("case_id") or "auto-" + short_hash(question, 8)).strip(),
                    question=question,
                    split=(row.get("split") or default_split).strip() or default_split,
                    reference_answer=(row.get("reference_answer") or "").strip(),
                    reference_contexts=contexts,
                    expected_keywords=keywords,
                    tags=[t.strip() for t in (row.get("tags") or "").split(",") if t.strip()],
                    source=(row.get("source") or "human").strip(),
                    difficulty=(row.get("difficulty") or "medium").strip(),
                    should_refuse=(row.get("should_refuse") or "").strip().lower() in ("1", "true", "yes", "是"),
                    adversarial_type=(row.get("adversarial_type") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                )
            )
    return cases


def load_cases_from_file(path: str | Path) -> list[CaseSpec]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"测试集不存在：{path}")
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return _load_jsonl(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise DatasetError(f"不支持的测试集格式：{suffix}")


def load_cases(paths: str | Sequence[str] | Path) -> list[CaseSpec]:
    """加载一个或多个测试集文件/目录，自动去重并校验 id 唯一性。"""
    if isinstance(paths, (str, Path)):
        items: list[Path] = [Path(paths)]
    else:
        items = [Path(p) for p in paths]
    cases: list[CaseSpec] = []
    for item in items:
        if item.is_dir():
            files = sorted(
                p for p in item.rglob("*") if p.suffix.lower() in (".jsonl", ".json", ".csv")
            )
            if not files:
                raise DatasetError(f"目录中没有测试集文件：{item}")
            for file in files:
                cases.extend(load_cases_from_file(file))
        else:
            cases.extend(load_cases_from_file(item))
    if not cases:
        raise DatasetError("测试集为空")
    seen: dict[str, CaseSpec] = {}
    for case in cases:
        if case.case_id in seen:
            existing = seen[case.case_id]
            if existing.question != case.question:
                # id 冲突但内容不同：补后缀而不是覆盖，避免静默丢用例
                case.case_id = f"{case.case_id}-{short_hash(case.question, 6)}"
            else:
                continue
        seen[case.case_id] = case
    return list(seen.values())


def write_cases(path: str | Path, cases: Sequence[CaseSpec]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    return path


def filter_cases(cases: Sequence[CaseSpec], splits: Sequence[str] | None = None, max_cases: int = 0, seed: int = 42) -> list[CaseSpec]:
    selected = list(cases)
    if splits:
        wanted = {s.lower() for s in splits}
        selected = [c for c in selected if c.split in wanted]
    if max_cases and len(selected) > max_cases:
        rng = stable_rng("subset", seed)
        selected = sorted(rng.sample(selected, max_cases), key=lambda c: c.case_id)
    return selected


@dataclass
class DatasetStats:
    total: int
    splits: dict[str, int]
    sources: dict[str, int]
    adversarial_types: dict[str, int]
    with_reference: int
    should_refuse: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "splits": self.splits,
            "split_labels": {k: SPLIT_LABELS.get(k, k) for k in self.splits},
            "sources": self.sources,
            "adversarial_types": self.adversarial_types,
            "with_reference": self.with_reference,
            "should_refuse": self.should_refuse,
        }


def dataset_stats(cases: Sequence[CaseSpec]) -> DatasetStats:
    splits = {s: 0 for s in SPLITS}
    sources: dict[str, int] = {}
    adv: dict[str, int] = {}
    with_ref = 0
    refuse = 0
    for case in cases:
        splits[case.split] = splits.get(case.split, 0) + 1
        sources[case.source] = sources.get(case.source, 0) + 1
        if case.reference_answer or case.reference_contexts:
            with_ref += 1
        if case.should_refuse:
            refuse += 1
        if case.adversarial_type:
            adv[case.adversarial_type] = adv.get(case.adversarial_type, 0) + 1
    return DatasetStats(
        total=len(cases),
        splits=splits,
        sources=sources,
        adversarial_types=adv,
        with_reference=with_ref,
        should_refuse=refuse,
    )


def merge_sets(human: Sequence[CaseSpec], synthetic: Sequence[CaseSpec], synthetic_weight: float = 1.0) -> list[CaseSpec]:
    """人工标注集优先：合成的同问题用例会被人工版本替换。"""
    by_question: dict[str, CaseSpec] = {}
    for case in human:
        by_question[case.question.strip()] = case
    for case in synthetic:
        key = case.question.strip()
        if key in by_question:
            continue
        if synthetic_weight < 1.0 and stable_rng("merge", key).random() > synthetic_weight:
            continue
        by_question[key] = case
    return sorted(by_question.values(), key=lambda c: (SPLITS.index(c.split), c.case_id))


def suite_summary_line(cases: Sequence[CaseSpec]) -> str:
    stats = dataset_stats(cases)
    parts = [f"{SPLIT_LABELS.get(k, k)} {v}" for k, v in stats.splits.items() if v]
    return f"{stats.total} 例（" + "，".join(parts) + "）"


def preview(cases: Sequence[CaseSpec], limit: int = 5) -> list[dict[str, str]]:
    return [
        {
            "case_id": c.case_id,
            "split": c.split,
            "question": truncate(c.question, 80),
        }
        for c in cases[:limit]
    ]
