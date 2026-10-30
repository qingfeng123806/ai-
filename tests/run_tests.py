"""标准库测试运行器：无需 pytest 也能跑完整测试集。

用法：
    python -m tests.run_tests            # 全部
    python -m tests.run_tests text stats # 只跑名字匹配的模块
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def discover(patterns: list[str]) -> list[str]:
    modules = []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        name = path.stem
        if patterns and not any(pattern in name for pattern in patterns):
            continue
        modules.append(f"tests.{name}")
    return modules


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    base = Path(__file__).resolve().parent.parent / "artifacts" / ".test-tmp"
    if base.exists() and not argv:
        import shutil

        shutil.rmtree(base, ignore_errors=True)
    modules = discover(argv)
    if not modules:
        print("没有匹配的测试模块")
        return 1
    total = passed = 0
    failures: list[tuple[str, str, str]] = []
    started = time.perf_counter()
    for module_name in modules:
        module = importlib.import_module(module_name)
        tests = [
            (name, fn)
            for name, fn in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_")
        ]
        print(f"\n{module_name}  ({len(tests)} 个用例)")
        for name, fn in tests:
            total += 1
            t0 = time.perf_counter()
            try:
                fn()
            except Exception:
                failures.append((module_name, name, traceback.format_exc()))
                print(f"  FAIL  {name}  ({time.perf_counter() - t0:.3f}s)")
            else:
                passed += 1
                print(f"  ok    {name}  ({time.perf_counter() - t0:.3f}s)")
    elapsed = time.perf_counter() - started
    print("\n" + "=" * 66)
    if failures:
        print(f"失败 {len(failures)} / {total} 个用例：\n")
        for module_name, name, tb in failures:
            print(f"--- {module_name}::{name} ---")
            print(tb)
        print(f"用时 {elapsed:.2f}s")
        return 1
    import shutil

    shutil.rmtree(base, ignore_errors=True)  # 全部通过时清理临时目录
    print(f"全部通过：{passed}/{total} 个用例，用时 {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
