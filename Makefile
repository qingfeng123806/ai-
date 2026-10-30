# kb-rag-eval 常用任务入口
# 说明：命令统一写成 `python -m ...`，在 Windows / Linux 下行为一致；
# 如需指定解释器，用 `make PY=python3.12 <target>` 覆盖。

PY ?= python
CONFIG ?= configs/baseline.toml
DATASET ?= data/testsets/core.jsonl
EXPERIMENT ?= configs/experiments/ab_chunking.toml
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: install doctor ingest testset eval ab serve test clean

## install: 安装可选的 API 依赖（核心链路零第三方依赖，可跳过）
install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

## doctor: 检查运行环境与各后端（BGE / Chroma / Milvus / LLM 密钥）可用性
doctor:
	$(PY) -m rag_eval doctor

## ingest: 构建或复用索引（内容指纹命中时直接复用缓存）
ingest:
	$(PY) -m rag_eval ingest --config $(CONFIG)

## testset: 查看/校验测试集概况
testset:
	$(PY) -m rag_eval testset

## eval: 在指定测试集上跑一次评测
eval:
	$(PY) -m rag_eval eval --config $(CONFIG) --dataset $(DATASET)

## ab: 运行一组 A/B 实验
ab:
	$(PY) -m rag_eval ab --experiment $(EXPERIMENT) --config $(CONFIG)

## serve: 启动 HTTP API 与看板
serve:
	$(PY) -m rag_eval serve --host $(HOST) --port $(PORT)

## test: 运行单元测试
test:
	$(PY) -m pytest -q

## clean: 清理缓存与派生产物（保留 artifacts/runs 中的原始评测记录）
clean:
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in map(pathlib.Path, ['artifacts/.cache', 'artifacts/reports', '.pytest_cache', '.ruff_cache'])]"
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
