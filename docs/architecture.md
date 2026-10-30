# 系统架构

本文档描述 `kb-rag-eval` 的分层结构、数据流、降级策略与扩展点，内容与 `rag_eval/` 下的实现一致。

## 1. 设计目标

1. **离线可跑通。** 核心链路只依赖 Python 3.11+ 标准库：无网络、无第三方包、无模型权重时，仍能完成
   「建索引 → 检索 → 生成 → 评测 → 报告 → 看板」全流程，可直接用作 CI 的回归门禁。
2. **在线自动增强。** BGE、Chroma、Milvus、OpenAI 兼容接口、RAGAs、DeepEval 均以可选适配器接入；
   检测到依赖与凭据就自动启用，缺失则降级，调用失败再降级，评测本身不因后端缺失而中断。
3. **可追溯。** 每次运行绑定配置指纹、后端清单、逐用例答案与引用；配置差异、指标口径、坏例证据都可回放。

## 2. 分层结构

```
+----------------------------------------------------------------------------------+
|  接口层  cli.py（python -m rag_eval）  api.py（传输无关的 dispatch）              |
|          dashboard.py（标准库 HTTP）  fastapi_app.py（ASGI，可选依赖）            |
|          static/（看板 index.html / app.js / styles.css，hash 路由）              |
|          路由：/api/health /api/overview /api/runs /api/compare /api/experiments  |
|                /api/metrics /api/datasets /api/report /api/eval /api/jobs ...     |
+----------------------------------------------------------------------------------+
|  编排层  pipeline.py   建索引（load → chunk → embed → store）/ 问答（retrieve →     |
|                        rerank → generate）；index 按内容指纹缓存                  |
|          runner.py     一次评测 = 数据加载 + 逐用例执行 + judge + 指标 + 持久化    |
|          experiments.py A/B 矩阵、独立索引、配对检验、综合评分、冠军判定           |
+----------------------------------------------------------------------------------+
|  能力层  ingest.py     5 类加载器（md/html/txt/pdf/docx）+ 4 种切片器              |
|          embeddings.py BGE / 哈希向量 + 缓存                                      |
|          store.py      memory / chroma / milvus                                   |
|          retrieve.py   BM25 + 稠密 + RRF + MMR + 重排序                           |
|          generate.py   Prompt 注册表 + 引用约束 + 在线/离线生成                    |
|          judge.py      LLM-as-Judge / 启发式量规                                  |
|          adapters.py   RAGAs / DeepEval 翻译层                                    |
|          metrics/      32 个指标 + 坏例规则 + 聚合与置信区间                       |
|          stats.py      均值/分位/Bootstrap/配对检验/Cohen's d                      |
|          text.py       分词、分句、覆盖率、引用解析、PII/注入检测（公共原语）       |
+----------------------------------------------------------------------------------+
|  数据层  db.py   SQLite（runs / case_results / experiments / artifacts，WAL）      |
|          dataset.py  JSONL / JSON / CSV 测试集加载、去重、拆分过滤、统计           |
|          types.py    Document / Chunk / Retrieved / CaseSpec / CaseResult / RunRecord |
+----------------------------------------------------------------------------------+
```

## 3. 数据流

### 3.1 建索引

```
data/corpus/*.md
   │  load_corpus(paths, include_globs)
   ▼
Document[]  ── chunk_documents(cfg.ingest) ──►  Chunk[]
   │  4 种切片：fixed / recursive / heading / semantic
   ▼
build_embedder(cfg.embedding)
   │  auto: BGE 预热成功 → BGE；失败 → 哈希向量（字符 2-3 gram，blake2b 带符号哈希，次线性 TF，L2 归一化）
   ▼
build_store(cfg.store, dim, persist_dir=index_dir)
   │  auto: chromadb 可导入 → Chroma；否则内存向量库（float32 + JSONL 持久化）
   ▼
artifacts/index/<signature>/  （chunks.jsonl + vectors.f32 + index.meta.json）
```

索引目录名为 16 位内容指纹：`sha256(语料指纹 + 切片参数 + 向量器签名)[:16]`，其中语料指纹由文档 id 与长度、
切片参数（`chunker|chunk_size|chunk_overlap|min_chunk_chars|heading_aware|semantic_threshold`）决定。
因此换切片策略或换向量模型必然生成新目录，旧索引不被覆盖；参数与语料不变时直接命中缓存（`index.meta.json` 中的
`signature` 校验通过且 `store.load()` 成功）。

### 3.2 在线问答

```
question
   │  embed_query
   ▼
稠密召回 candidate_k ─┐
                     ├─ RRF 融合（dense_weight/(k+rank) + lexical_weight/(k+rank)，k=rrf_k，融合分归一到 1.0）
BM25 召回 candidate_k ┘
   │  （可选 MMR 去重，min_score 过滤）
   ▼
rerank：BGE CrossEncoder / FlagReranker → sigmoid 映射到 0-1；
        缺失时词法重排（0.55 × 词覆盖与短语命中 + 0.25 × 稠密分先验 + 0.20 × 标题命中）
   ▼
top_k 片段 → format_contexts 渲染为 [1] 《标题》/ 章节 编号块
   ▼
generate：在线走 OpenAI 兼容 Chat Completions（JSON 模式下也用于 Judge）；
          离线抽取式生成器按 Prompt style 摘句 + 逐句挂引用 + 拒答护栏
   ▼
Generation{answer, citations, refused, tokens, cost, latency}
```

每个阶段单独计时（`dense_ms` / `lexical_ms` / `fuse_ms` / `rerank_ms` / `generation_ms`），
`LatencyBreakdown` 汇成 `retrieval_ms`、`rerank_ms`、`generation_ms`、`total_ms`，供性能指标使用。

### 3.3 评测运行

```
data/testsets/*.jsonl
   │  dataset.load_cases（自动去重、按文件名推断 split）
   ▼
CLI 按 --splits / --max-cases 过滤（dataset.filter_cases 提供同样的过滤能力）
   ▼
对每个用例：pipeline.query → CaseResult（答案、引用、片段、延迟、token、成本）
   │  judge.evaluate（LLM 或启发式）
   ▼
evaluate_case：逐指标计算 → 截断 [0,1] → 按 DEFAULT_FLAG_RULES 标记坏例 → 逐句支撑证据
   ▼
aggregate_metrics：均值 + Bootstrap 置信区间；聚合指标按用例重采样
   ▼
RunRecord + 逐用例 JSONL（artifacts/runs/<run_id>.jsonl / .meta.json）+ SQLite
```

单用例异常被记录为 `error` 字段而不是抛出；单指标异常记录到 `details.metric_errors`；
外部框架（RAGAs / DeepEval）失败只影响 external 指标。整体原则：**评测系统本身要比被评测系统更稳**。

## 4. 运行模式与降级矩阵

`RAGPipeline.mode` 由「生成器是否在线」与「向量器是否为真实模型」推导：

| 条件 | mode |
| --- | --- |
| BGE 生效且 LLM 可用 | `live` |
| 二者只有一个生效 | `hybrid` |
| 都为内置实现 | `offline` |

| 组件 | 配置项 | 首选后端 | 缺失/失败时 | 触发降级的时机 |
| --- | --- | --- | --- | --- |
| 向量化 | `embedding.provider` = `auto`/`bge`/`hashing` | BGE（sentence-transformers → FlagEmbedding） | 哈希向量（384 维，确定性） | `auto` 下预热 `embed_query("预热")` 失败即降级；`bge` 显式指定时不降级 |
| 向量库 | `store.provider` = `auto`/`memory`/`chroma`/`milvus` | Chroma（持久化） | 内存向量库（余弦暴力检索） | `auto` 下 `import chromadb` 失败即降级；`milvus` 需显式指定 |
| 重排序 | `rerank.provider` = `auto`/`bge`/`lexical`/`none` | BGE 交叉编码器 | 词法重排 | `auto` 下模型加载失败即降级；运行期加载失败退回词法 |
| 生成 | `generate.provider` = `auto`/`openai`/`offline` | OpenAI 兼容接口 | 离线抽取式生成器 | 无 API Key 时 `build_llm` 返回 `None`；调用抛 `LLMError` 时该用例回退，`provider` 记为 `offline-extractive(fallback)` |
| Judge | `evaluate.judge_provider` = `auto`/`offline` | 复用 LLM 客户端 | 启发式量规 | 无 LLM 或解析失败时回退，原因写入 `reasons.error` |
| RAGAs / DeepEval | `evaluate.ragas` / `evaluate.deepeval` | 外部框架 | 不参与计算 | 未安装依赖或缺 API Key 时状态标记为不可用并记录原因 |

所有实际生效的后端都会写入 `RunRecord.providers`（如 `embedding: hashing:hashing-char2_3`），
`RunRecord.mode` 与 `metadata.adapters` 一并持久化，报告可据此判断数字的适用边界。

## 5. 持久化

SQLite 单文件（默认 `artifacts/eval.db`，WAL 模式），四张表：

| 表 | 主键 | 内容 |
| --- | --- | --- |
| `runs` | `run_id` | 配置指纹与完整配置、指标 JSON、后端清单、模式、耗时、实验归属 |
| `case_results` | (`run_id`,`case_id`) | 逐用例完整载荷 JSON（答案、引用、片段、分数、证据、坏例标记） |
| `experiments` | `experiment_id` | 实验配置、各变体结果、汇总（含冠军与排名） |
| `artifacts` | 自增 | 运行产物登记（如 `results.jsonl` 路径） |

JSON 字段统一序列化，指标增减不需要 schema 迁移；看板读取与评测写入可并发（WAL）。
另外每次运行都会落盘一对文件：`artifacts/runs/<run_id>.jsonl`（逐用例）与
`artifacts/runs/<run_id>.meta.json`（运行记录），即使数据库损坏也可人工复盘。

## 6. 扩展点

| 扩展 | 做法 | 位置 |
| --- | --- | --- |
| 新增切片策略 | 写 `chunk_*(doc, cfg) -> list[Chunk]` 并注册进 `CHUNKERS`，同时在 `config.CHUNKERS` 增加名称 | `ingest.py` |
| 新增向量后端 | 实现 `Embedder` 协议（`name/kind/dim/embed_documents/embed_query/info`），在 `build_embedder` 中分派 | `embeddings.py` |
| 新增向量库 | 继承 `VectorStore` 实现 `add/search/count/persist/load/all_chunks/stats` | `store.py` |
| 新增指标 | `@case_metric(MetricSpec(name=..., group=..., direction=...))`，函数签名为 `(CaseResult, EvalContext) -> float \| None` | `metrics/` 下任一模块 |
| 新增 Prompt | 在 `PROMPTS` 注册 `PromptTemplate(name, system, user, style)`，配置 `generate.prompt` 引用 | `generate.py` |
| 新增测试集格式 | 在 `load_cases_from_file` 增加后缀分支 | `dataset.py` |
| 新增外部评测框架 | 参照 `RagasAdapter`：`status` + `available` + `evaluate(cases, results) -> {case_id: {metric: score}}` | `adapters.py` |
| 新增实验维度 | 在实验 TOML 的 `overrides` 中覆盖任意 section 字段，无需改代码 | `configs/experiments/` |

## 7. 关键取舍

- **为什么用内容指纹而不是配置哈希做索引键？** 配置里包含 `top_k`、`prompt` 等与索引无关的字段；
  只对「影响切片与向量」的参数取指纹，才能让 TopK / Prompt / 模型实验复用同一份索引。
- **为什么 RRF 而不是加权分数相加？** 余弦相似度在 0~1，BM25 无上界，直接相加会让某一路分数支配排序；
  RRF 只用名次，天然消除量纲差异。
- **为什么离线生成器要带拒答护栏？** 知识域外问题若强行摘句，会产出「看似有据实则无关」的答案；
  生成器因此加入注入检测、域外检测（问题锚点词大量不在语料 idf 中）、证据分阈值与相对阈值。
- **为什么指标用词法原语？** 离线环境无法调用 embedding 相似度或 LLM 判定；词法覆盖 + idf 加权在中文 FAQ 语料上
  与人工判断的一致性足够做回归对比，而且逐句可解释、结果可复现。

## 8. 已知限制

- 内存向量库为暴力检索，数万片段内可用；更大规模需切换 Chroma / Milvus。
- 哈希向量只捕捉字面 n-gram 相似度，同义改写召回弱于 BGE。
- 词法重排依赖查询词与片段词面重合，跨语言或强改写的排序收益有限。
- SQLite 与文件产物均为单机；多实例部署需要共享存储与外部数据库。
