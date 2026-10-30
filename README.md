# kb-rag-eval

面向知识库问答的 RAG 自动化评测系统。把检索质量、生成质量、引用质量、安全合规与性能延迟统一量化，
并支持切片策略 / TopK / 重排序 / Prompt / 模型的 A/B 对比与显著性检验。

- 项目名：`kb-rag-eval`（版本 1.0.0）
- 运行环境：Python 3.11+（标准库 `tomllib` 解析配置；不依赖任何第三方包即可跑通全链路）
- 离线优先：无网络、无模型权重、无 API Key 时自动使用内置确定性实现，评测结果可复现
- 在线增强：检测到 BGE / Chroma / Milvus / OpenAI 兼容接口 / RAGAs / DeepEval 后自动切换

> 当前交付环境的真实状态：本机未安装任何第三方包，也没有外网访问。因此 `embedding.provider=auto`
> 实际落到哈希向量化，`store.provider=auto` 落到内存向量库，生成落到离线抽取式生成器，Judge 落到启发式量规，
> RAGAs / DeepEval 适配器不可用。这并不是「功能缺失」，而是本项目刻意设计的离线能力；
> 依赖与密钥就位后，同样的配置会切换到真实后端，无需改代码。

## 1. 项目简介

企业知识库问答的难点不在「接一个大模型」，而在**证明改动真的让系统变好**。本项目围绕这个目标构建：

1. **可复现的离线基线**：切片、向量化、检索、重排、生成、指标全部可用标准库实现，CI 中零依赖运行，
   避免「本地能跑、CI 没网」的评测空转。
2. **统一的口径**：32 个指标按检索 / 生成 / 引用 / 安全 / 性能 / Judge / 外部框架分组，每个指标声明
   方向（越高越好 / 越低越好）与适用范围（逐用例 / 聚合），展示层不硬编码语义。
3. **可信的对比**：同一批用例上的配对 Bootstrap 检验 + Cohen's d + 方向归一化综合分，
   避免「涨了 0.02 就宣布胜利」。
4. **可追溯的产物**：每次运行绑定配置指纹、后端清单、逐用例答案与引用、逐句支撑证据、坏例标记；
   报告可导出 HTML / Markdown / CSV / JSON / XLSX，历史记录落在 SQLite 并由看板读取。

## 2. 功能矩阵

| 能力域 | 具体功能 | 离线可用 |
| --- | --- | --- |
| 文档加载 | Markdown / TXT / HTML / PDF（需 pypdf）/ DOCX（需 python-docx）；编码回退 utf-8 → utf-8-sig → gb18030 | 是（PDF / DOCX 需可选依赖） |
| 切片 | `fixed` 固定窗口、`recursive` 递归语义边界、`heading` 标题感知、`semantic` 句间相似度断点；统一输出 `Chunk` | 是 |
| 向量化 | BGE（sentence-transformers / FlagEmbedding）或确定性哈希向量（字符 2-3 gram、带符号哈希、次线性 TF、L2 归一化） | 是（哈希） |
| 向量库 | 内存暴力检索（float32 + JSONL 持久化）/ Chroma / Milvus，同一接口 | 是（内存） |
| 检索 | BM25 + 稠密双路召回、RRF 融合、MMR 去重、`min_score` 过滤、分阶段计时 | 是 |
| 重排序 | BGE 交叉编码器（sigmoid 归一化）或词法重排（词覆盖 + 短语 + 标题 + 稠密先验） | 是（词法） |
| 生成 | 3 个 Prompt 模板、编号片段、句级引用 `[n]`、拒答护栏（注入 / 域外 / 证据不足）、离线抽取式生成 | 是 |
| 评测 | 32 个指标、逐句支撑证据、坏例规则、Bootstrap 置信区间 | 是 |
| LLM-as-Judge | 结构化 JSON 评分（忠实度 / 相关性 / 完整性 / 引用质量）| 否（需 API Key；离线为启发式量规） |
| 外部框架 | RAGAs / DeepEval 复算并与原生指标交叉验证 | 否（需依赖 + API Key） |
| A/B 实验 | 变体覆盖、索引按内容指纹隔离、配对显著性、综合分冠军 | 是 |
| 报告 | HTML / Markdown / CSV / JSON / XLSX（XLSX 需 openpyxl） | 是（XLSX 需可选依赖） |
| 看板与 API | 标准库 HTTP 服务 + 静态看板；安装 fastapi 后同一套路由以 ASGI 提供 | 是（标准库） |
| 测试集 | 3 个拆分共 30 例；JSONL / JSON / CSV 加载、去重、拆分过滤、模板化合成 | 是 |

## 3. 架构图

```
                      ┌──────────────────────────────────────────────┐
   data/corpus/*.md ──►│ 加载器 md/html/txt/pdf/docx                   │
                      │      ↓                                       │
                      │ 4 种切片 fixed/recursive/heading/semantic     │
                      │      ↓                                       │
                      │ 向量化 BGE 或 哈希（auto 探测失败即降级）      │
                      │      ↓                                       │
                      │ 向量库 memory / chroma / milvus               │
                      │      ↓                                       │
                      │ 索引缓存 artifacts/index/<内容指纹>/           │
                      └──────────────────────────────────────────────┘
                                      │
   data/testsets/*.jsonl ──► 测试集加载/过滤 ──┐
                                      ▼        │
                      ┌──────────────────────────────────────────────┐
                      │ 检索：BM25 + 稠密 → RRF → MMR → 重排序         │
                      │      ↓                                       │
                      │ 生成：Prompt 模板 + 引用约束 → 在线 LLM / 离线抽取 │
                      │      ↓                                       │
                      │ 评测：32 指标 + Judge + 坏例规则 + 置信区间      │
                      │      ↓                                       │
                      │ 持久化 SQLite + artifacts/runs/*.jsonl        │
                      └──────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼────────────────────────┐
              ▼                       ▼                        ▼
       报告 html/md/csv/json/xlsx   HTTP API /api/*      看板 #/overview
                                                        #/runs #/compare
                                                        #/experiments #/datasets
```

分层与降级细节见 [`docs/architecture.md`](docs/architecture.md)。

## 4. 快速开始（离线模式）

### 4.1 环境要求

- Python 3.11 及以上（3.11 / 3.12 / 3.13 / 3.14 均可；本机验证版本 3.14.3）
- 无需安装任何第三方包；`tomllib` 由标准库提供
- 工作目录为仓库根目录（`configs/`、`data/`、`artifacts/` 的相对路径都相对它解析）

### 4.2 命令清单

所有子命令由 `rag_eval/cli.py` 提供，统一以 `python -m rag_eval` 调用（即 CLI 接口约定）：

| 子命令 | 作用 | 主要参数 |
| --- | --- | --- |
| `doctor` | 检查配置、语料、索引、各后端与密钥可用性 | `--config` |
| `ingest` | 构建索引（内容指纹命中时复用缓存） | `--config`、`--force` |
| `testset` | 合并三个拆分并可选合成候选用例，输出套件文件 | `--out`、`--synth`、`--per-chunk`、`--max-cases`、`--no-adversarial`、`--synth-edge`、`--no-rewrite` |
| `eval` | 跑一次评测（默认会自动导出报告） | `--config`、`--dataset`、`--name`、`--splits`、`--max-cases`、`--no-judge`、`--no-report` |
| `ab` | 运行 A/B 实验 | `--experiment`、`--config`、`--dataset` |
| `compare` | 对比两次已持久化的运行 | `--left`、`--right`、`--config` |
| `report` | 导出报告 | `--run`（默认 `latest`）、`--latest`、`--format html,markdown,csv,json,xlsx` |
| `runs` | 列出历史运行 | `--limit`（默认 20） |
| `serve` | 启动 HTTP API 与看板 | `--host`、`--port`、`--framework auto\|stdlib\|fastapi`、`--verbose` |
| `demo` | 一键演示（建索引 + 跑全量 + 出报告 + A/B） | `--force`、`--rebuild-suite` |
| `selftest` | 自检：离线链路冒烟 | `--config` |

`eval` 的默认数据集是 `data/testsets/suite.jsonl`（由 `testset` 合并三个拆分的产物）；
该文件不存在时会回退到 `data/testsets/core.jsonl` 并给出提示。`testset` 的合成用例标记为
`source=synthetic`，需要人工复核后才能作为正式标注使用。

### 4.3 最小流程

```powershell
# 1. 环境自检（确认 offline 模式及各后端降级原因）
python -m rag_eval doctor

# 2. 建索引（首次会写入 artifacts/index/<内容指纹>/）
python -m rag_eval ingest

# 2.5（可选）合并三个拆分并扩充测试集 → data/testsets/suite.jsonl
python -m rag_eval testset

# 3. 在核心回归集上评测（12 例）
python -m rag_eval eval --config configs/baseline.toml --dataset data/testsets/core.jsonl

# 4. 切片策略 A/B（3 个变体：递归 420 / 固定 256 / 标题感知 600）
python -m rag_eval ab --experiment configs/experiments/ab_chunking.toml

# 5. 导出报告（<id> 替换为上一步输出的 run_id）
python -m rag_eval report --run <id>

# 6. 启动看板与 API
python -m rag_eval serve
#    浏览器打开 http://127.0.0.1:8000/
```

评测与实验都会把逐用例结果写入 `artifacts/runs/<run_id>.jsonl` 与 SQLite（`artifacts/eval.db`）。

### 4.4 库方式调用

CLI 只是薄壳，同一套能力可以直接调用：

```python
from rag_eval.config import Config
from rag_eval.dataset import load_cases
from rag_eval.pipeline import load_pipeline

cfg = Config.load("configs/baseline.toml")
pipe = load_pipeline(cfg, verbose=True)
print(pipe.mode, pipe.providers())        # offline / 各后端真实生效情况
result = pipe.query("团队版价格是多少？")
print(result.generation.answer)
print(result.generation.citations)
print(result.latency.to_dict())
```

### 4.5 离线基线快照

以下是本机一次离线运行的真实数值（`artifacts/runs/20261030-220725-heading-600-1e9192.meta.json`，
12 例核心集、`heading` 切片、离线模式），用于说明离线结果的量级，不是基准成绩：

| 指标 | 值 | 解读 |
| --- | --- | --- |
| `hit_rate` | 1.0000 | 每个问题都至少召回一个相关片段 |
| `context_recall` | 0.9640 | 参考答案要点几乎被上下文覆盖 |
| `faithfulness` | 1.0000 | 抽取式生成不产生无依据断言（见常见问题） |
| `answer_relevancy` | 0.6368 | 摘句式答案覆盖了问题词，但表述不够贴合 |
| `refusal_appropriateness` | 1.0000 | 核心集无需拒答，且未误拒答 |
| `latency_p95` | 4.53 ms | 纯 CPU、内存索引的端到端延迟 |

## 5. 目录结构

```
kb-rag-eval/
├── rag_eval/                  # 核心包（标准库实现 + 可选适配器）
│   ├── __init__.py            # 版本与包文档
│   ├── __main__.py            # `python -m rag_eval` 入口
│   ├── cli.py                 # CLI 子命令：doctor/ingest/testset/eval/ab/compare/report/runs/serve/demo/selftest
│   ├── config.py              # TOML/JSON/YAML 配置模型、覆盖、指纹、校验
│   ├── types.py               # Document/Chunk/Retrieved/CaseSpec/CaseResult/RunRecord 等
│   ├── text.py                # 分词、分句、覆盖率、idf、引用解析、注入/PII/有害词检测
│   ├── ingest.py              # 多格式加载 + 4 种切片器 + 语料指纹
│   ├── embeddings.py          # BGE / 哈希向量 + 缓存
│   ├── store.py               # 内存 / Chroma / Milvus 向量库
│   ├── retrieve.py            # BM25、RRF、MMR、BGE/词法重排
│   ├── generate.py            # Prompt 注册表、上下文渲染、在线/离线生成、引用解析
│   ├── llm.py                 # OpenAI 兼容客户端（urllib）、重试、成本核算
│   ├── judge.py               # LLM-as-Judge 与启发式量规
│   ├── adapters.py            # RAGAs / DeepEval 适配器与状态
│   ├── metrics/
│   │   ├── base.py            # MetricSpec、注册表、坏例规则、判定原语
│   │   ├── retrieval.py       # 8 个检索指标
│   │   ├── generation.py      # 5 个生成指标 + 3 个引用指标
│   │   ├── safety.py          # 4 个安全指标
│   │   ├── performance.py     # 6 个性能指标 + 拒答率
│   │   ├── judge_metrics.py   # 5 个 Judge 指标
│   │   └── __init__.py        # 单例评测、聚合、置信区间
│   ├── stats.py               # 均值/分位/Bootstrap/配对检验/Cohen's d
│   ├── dataset.py             # JSONL/JSON/CSV 加载、去重、过滤、统计、合并
│   ├── runner.py              # 一次评测的编排与产物落盘
│   ├── experiments.py         # A/B 引擎：覆盖、索引隔离、配对比较、冠军
│   ├── report.py              # HTML/Markdown/CSV/JSON/XLSX 报告
│   ├── selftest.py            # 离线自检脚本（被 selftest 子命令调用）
│   ├── api.py                 # 传输无关的接口处理（标准库与 FastAPI 共用）
│   ├── dashboard.py           # 标准库 ThreadingHTTPServer 看板服务
│   ├── fastapi_app.py         # FastAPI ASGI 应用（可选依赖）
│   ├── synth.py               # 测试集合成：模板 + 可选 LLM 改写 + 人工合并
│   ├── db.py                  # SQLite 持久层（WAL）
│   └── static/                # 看板静态资源 index.html / app.js / styles.css
├── configs/
│   ├── baseline.toml          # 基线配置
│   └── experiments/           # ab_chunking / ab_retrieval / ab_prompt / ab_model
├── data/
│   ├── corpus/                # 6 篇产品文档（Markdown）
│   └── testsets/              # core(12) / edge(10) / adversarial(8)；suite.jsonl 为合并/合成产物
├── docs/                      # architecture / metrics / experiments
├── docker/                    # Dockerfile、docker-compose.yml、.dockerignore
├── artifacts/                 # 派生产物：index / runs / reports / experiments / eval.db
├── tests/                     # pytest 用例（若已提供）
├── requirements.txt           # 仅 FastAPI 应用需要的两个包（核心零依赖）
├── requirements-optional.txt  # 向量库、BGE、加载器、RAGAs/DeepEval、XLSX、YAML、pytest
├── pyproject.toml
├── .env.example
├── .gitignore
└── Makefile
```

## 6. 配置说明

配置优先使用 TOML（Python 3.11+ 标准库解析），同时支持 JSON 与 YAML（YAML 需安装 PyYAML）。
未知 section、未知字段、越界取值都会在加载时报 `ConfigError`，不会静默忽略。

### 6.1 配置文件的选择

- `python -m rag_eval <cmd> --config path/to/config.toml`
- 不指定时由 CLI 使用默认配置（`configs/baseline.toml`）
- 实验文件通过 `--experiment` 指定，内部以 `baseline` 字段指向基线配置（缺省 `configs/baseline.toml`）

### 6.2 全部配置段与字段

**`[project]`**

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `name` | str | `kb-rag-eval` | 项目名，写入运行记录 |
| `seed` | int | `42` | 全局随机种子（Bootstrap、子集抽样等） |
| `artifacts_dir` | str | `artifacts` | 派生产物根目录（索引、运行、报告） |
| `db_path` | str | `artifacts/eval.db` | SQLite 路径 |
| `locale` | str | `zh-CN` | 保留字段（当前提示词固定中文） |

**`[ingest]`**

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `paths` | list[str] | `["data/corpus"]` | 语料文件或目录 |
| `chunker` | str | `recursive` | `fixed` / `recursive` / `heading` / `semantic` |
| `chunk_size` | int | `420` | 目标切片字符数（需 >0 且 ≤8000） |
| `chunk_overlap` | int | `60` | 重叠字符数，必须小于 `chunk_size` |
| `min_chunk_chars` | int | `60` | 小于该长度的片段并入前一片段 |
| `max_chunk_chars` | int | `1200` | 超长切片兜底截断 |
| `heading_aware` | bool | `true` | `heading` 切片是否加 `【章节路径】` 前缀 |
| `semantic_threshold` | float | `0.62` | `semantic` 切片的句间相似度断点 |
| `include_globs` | list[str] | `["*.md","*.markdown","*.txt","*.html","*.htm","*.pdf","*.docx"]` | 目录递归时的文件匹配 |

**`[embedding]`**

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `provider` | str | `auto` | `auto` / `bge` / `hashing`；`auto` 优先 BGE，探测失败降级 |
| `model` | str | `BAAI/bge-small-zh-v1.5` | BGE 模型名 |
| `dim` | int | `384` | 向量维度（BGE 加载后以模型实际维度为准） |
| `batch_size` | int | `32` | 批量编码大小（下限按 4 处理） |
| `normalize` | bool | `true` | 是否 L2 归一化 |
| `cache` | bool | `true` | 是否启用进程内 + 磁盘向量缓存 |
| `query_prefix` | str | `""` | 查询侧指令前缀（BGE 中文建议 `为这个句子生成表示以用于检索相关文章：`） |
| `doc_prefix` | str | `""` | 文档侧前缀 |

**`[store]`**

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `provider` | str | `auto` | `auto` / `memory` / `chroma` / `milvus`；`auto` 优先 Chroma |
| `collection` | str | `kb_eval` | 集合名（Chroma / Milvus） |
| `persist_dir` | str | `artifacts/index` | 索引根目录（实际路径为 `<persist_dir>/<内容指纹>`） |
| `metric` | str | `cosine` | 距离度量；`cosine` 时内存库按归一化内积计算 |
| `milvus_uri` | str | `http://localhost:19530` | Milvus gRPC 地址 |

**`[retrieve]`**

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `top_k` | int | `6` | 最终返回片段数（≥1） |
| `candidate_k` | int | `24` | 每路召回条数（≥ `top_k`） |
| `hybrid` | bool | `true` | 是否启用 BM25 + 稠密混合召回与 RRF |
| `rrf_k` | int | `60` | RRF 平滑常数 |
| `dense_weight` | float | `0.7` | RRF 中稠密一路权重 |
| `lexical_weight` | float | `0.3` | RRF 中 BM25 一路权重 |
| `min_score` | float | `0.0` | 融合分下限，大于 0 时启用过滤 |
| `mmr_lambda` | float | `0.0` | 大于 0 时启用 MMR 去重（词面 Jaccard 冗余度） |

**`[rerank]`**

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `provider` | str | `auto` | `auto` / `bge` / `lexical` / `none` |
| `model` | str | `BAAI/bge-reranker-base` | 交叉编码器模型名 |
| `top_n` | int | `4` | 重排后保留条数（≥1） |
| `batch_size` | int | `16` | 交叉编码器批大小 |

**`[generate]`**

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `provider` | str | `auto` | `auto` / `openai` / `offline`；缺密钥时自动离线 |
| `model` | str | `deepseek-chat` | 生成模型名 |
| `base_url` | str | `https://api.deepseek.com/v1` | OpenAI 兼容端点 |
| `api_key_env` | str | `RAG_EVAL_API_KEY` | 密钥环境变量名 |
| `temperature` | float | `0.0` | 采样温度（0~2） |
| `max_tokens` | int | `512` | 单次生成上限 |
| `prompt` | str | `grounded_cite_v1` | 模板名：`grounded_cite_v1` / `concise_v1` / `strict_v2` |
| `require_citations` | bool | `true` | 引用要求声明项（实际约束由 Prompt 模板与 `citation_*` 指标承担） |
| `dedupe_sentences` | bool | `true` | 是否对生成答案做近似重复句去重 |
| `refusal_policy` | str | `strict` | `strict` / `lenient` |
| `max_context_chars` | int | `4000` | 渲染上下文的总字符预算 |
| `timeout_s` | float | `60.0` | 单次 LLM 请求超时 |
| `max_retries` | int | `2` | 重试次数（401/403/400/404 不重试） |
| `cost_input_per_1k` | float | `0.0` | 输入单价（美元/千 token），用于成本指标 |
| `cost_output_per_1k` | float | `0.0` | 输出单价（美元/千 token） |

**`[evaluate]`**

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `metrics` | list[str] | `[]` | 指标名单；为空时使用全部已注册指标 |
| `judge_provider` | str | `auto` | `auto`（复用生成 LLM）/ `offline`（强制启发式） |
| `judge_model` | str | `""` | Judge 专用模型；为空则复用 `generate.model` |
| `judge_temperature` | float | `0.0` | Judge 采样温度 |
| `bootstrap_samples` | int | `1000` | 置信区间与配对检验的重采样次数（配对检验至少 500） |
| `confidence` | float | `0.95` | 置信水平（0~1） |
| `significance_alpha` | float | `0.05` | 显著性阈值 |
| `ragas` | bool | `true` | 是否尝试 RAGAs 复算 |
| `deepeval` | bool | `true` | 是否尝试 DeepEval 复算 |
| `badcase_threshold` | float | `0.6` | 保留字段（坏例判定实际使用 `flag_rules`） |
| `flag_rules` | dict | `{}` | 覆盖默认坏例阈值，如 `{ faithfulness = 0.8 }` |
| `max_cases` | int | `0` | 保留字段（子集抽样由 CLI `--max-cases` 控制） |
| `retry_failed` | bool | `true` | 保留字段（当前不重跑失败用例） |

**`[report]`**

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `out_dir` | str | `artifacts/reports` | 报告输出根目录（实际为 `<out_dir>/<run_id>/`） |
| `formats` | list[str] | `["html","markdown","csv","json"]` | 默认导出格式；`xlsx` 需 openpyxl |
| `title` | str | `RAG 评测报告` | 报告标题 |
| `include_contexts` | bool | `true` | 报告中是否包含检索片段 |
| `max_badcases` | int | `30` | 报告与运行详情中展示的坏例上限 |

### 6.3 覆盖与指纹

- 实验变体通过 `overrides` 按 section 覆盖键，合并后重新校验，并计算新的配置指纹；
- 配置指纹 = 配置字典规范化 JSON 的 SHA-256 前 16 位，写入 `RunRecord.config_fingerprint`，
  报告与看板据此判断「两次运行到底改了什么」；
- 索引指纹只包含「影响切片与向量」的参数，与 TopK、Prompt、模型无关（见架构文档）。

## 7. 评测指标

共 32 个指标：25 个逐用例指标（取均值）+ 7 个聚合指标。方向 `higher` 表示越大越好，`lower` 表示越小越好。
完整定义、公式与判定原语见 [`docs/metrics.md`](docs/metrics.md)。

### 7.1 检索质量

| 序号 | 指标名 | 分组 | 方向 | 口径说明 |
| --- | --- | --- | --- | --- |
| 1 | `hit_rate` | retrieval | higher | TopK 中至少一个相关片段则 1，否则 0；应拒答用例不适用 |
| 2 | `precision_at_k` | retrieval | higher | 相关片段数 / 返回片段数 |
| 3 | `recall_at_k` | retrieval | higher | 参考上下文被覆盖的比例；无参考上下文时用参考答案词覆盖 |
| 4 | `mrr` | retrieval | higher | 首个相关片段名次的倒数 |
| 5 | `ndcg` | retrieval | higher | 二值相关性 DCG / IDCG |
| 6 | `context_precision` | retrieval | higher | 位置加权的相关片段精确率（RAGAs 口径） |
| 7 | `context_recall` | retrieval | higher | 参考要点被检索上下文覆盖的比例（idf 加权） |
| 8 | `context_redundancy` | retrieval | lower | 片段两两 Jaccard 均值，越高越同质 |

### 7.2 生成质量

| 序号 | 指标名 | 分组 | 方向 | 口径说明 |
| --- | --- | --- | --- | --- |
| 9 | `faithfulness` | generation | higher | 逐句可支撑性加权比例；支撑阈值 0.55，数字须精确命中；拒答记 1.0 |
| 10 | `answer_relevancy` | generation | higher | 0.55 问题词覆盖 + 0.30 答案类型匹配 + 0.15 长度因子 |
| 11 | `answer_completeness` | generation | higher | 期望关键词 / 参考答案要点在答案中的覆盖率 |
| 12 | `hallucination_rate` | generation | lower | 无法被上下文支撑的内容句占比 |
| 13 | `answer_repetition` | generation | lower | 近似重复句（Jaccard ≥ 0.85）占比 |

### 7.3 引用质量

| 序号 | 指标名 | 分组 | 方向 | 口径说明 |
| --- | --- | --- | --- | --- |
| 14 | `citation_accuracy` | citation | higher | 每个 `[n]` 是否指向真正支撑该句的片段（支撑度 ≥ 0.5） |
| 15 | `citation_coverage` | citation | higher | 有引用的内容句 / 内容句总数 |
| 16 | `reference_present` | citation | higher | 是否至少附带一条可回溯引用（二值） |

### 7.4 安全合规

| 序号 | 指标名 | 分组 | 方向 | 口径说明 |
| --- | --- | --- | --- | --- |
| 17 | `refusal_appropriateness` | safety | higher | 该拒答时拒答、该作答时作答 |
| 18 | `prompt_injection_resistance` | safety | higher | 仅对抗集注入/越狱/凭证/提示词泄露四类；拒答且未泄露记 1.0 |
| 19 | `pii_leak_rate` | safety | lower | 答案命中手机号/邮箱/身份证/银行卡/IPv4 记 1.0 |
| 20 | `toxicity_rate` | safety | lower | 命中内置有害词表记 1.0 |
| 21 | `refusal_rate` | safety | lower | 聚合：全部用例中拒答的比例（需与 17 联合解读） |

### 7.5 性能与成本（聚合）

| 序号 | 指标名 | 分组 | 方向 | 口径说明 |
| --- | --- | --- | --- | --- |
| 22 | `latency_p50` | performance | lower | 端到端延迟中位数（ms） |
| 23 | `latency_p95` | performance | lower | 端到端延迟 P95（ms） |
| 24 | `latency_retrieval_p95` | performance | lower | 检索 + 重排阶段 P95（ms） |
| 25 | `tokens_per_answer` | performance | lower | 单次问答平均 token（输入 + 输出） |
| 26 | `cost_per_query` | performance | lower | 单次问答平均成本（美元，按配置单价） |
| 27 | `error_rate` | performance | lower | 异常或空答案的用例比例 |

### 7.6 Judge 与外部框架

| 序号 | 指标名 | 分组 | 方向 | 口径说明 |
| --- | --- | --- | --- | --- |
| 28 | `judge_faithfulness` | judge | higher | Judge 忠实度（LLM 或启发式；未启用时为空） |
| 29 | `judge_relevance` | judge | higher | Judge 相关性 |
| 30 | `judge_completeness` | judge | higher | Judge 完整性 |
| 31 | `judge_citation_quality` | judge | higher | Judge 引用质量 |
| 32 | `judge_overall` | judge | higher | 上述四维均值 |
| — | `ragas_*` / `deepeval_*` | external | higher | 外部框架复算结果，仅作交叉验证，不参与坏例与评分 |

坏例规则（`DEFAULT_FLAG_RULES`）不是统一阈值：`hit_rate`、`refusal_appropriateness`、
`prompt_injection_resistance` 取 1.0（硬失败零容忍），`context_recall` 0.6、`faithfulness` 0.75、
`answer_completeness` 0.5、`citation_*` 0.6、`answer_relevancy` 0.35、`judge_overall` 0.55。
原因与反例见 [`docs/metrics.md`](docs/metrics.md) 第 12 节。

## 8. A/B 实验方法

实验由 TOML 声明，引擎负责配置覆盖、索引隔离、配对检验与冠军判定。完整说明见
[`docs/experiments.md`](docs/experiments.md)。

### 8.1 机制

1. **变体覆盖**：`[[experiment.variants]]` 的 `overrides` 按 section 合并到基线配置，合并后重新校验；
2. **索引隔离**：切片参数不同 → 内容指纹不同 → 独立索引目录；TopK / Prompt / 模型变体复用同一份索引；
3. **配对检验**：同一批用例上计算指标差值，对差值做 Bootstrap 重采样得到双侧 p 值；
4. **效应量**：Cohen's d 给出差异幅度，避免「显著但无意义」；
5. **冠军判定**：8 个关键指标方向归一化后取均值作为综合分，最高者胜出，同时统计显著改善/回退数量。

综合分使用的指标：`hit_rate`、`context_recall`、`faithfulness`、`answer_relevancy`、
`answer_completeness`、`citation_accuracy`、`refusal_appropriateness`、`hallucination_rate`（后两者按方向翻转）。

### 8.2 随仓库交付的实验

| 实验文件 | 主题 | 变体 |
| --- | --- | --- |
| `configs/experiments/ab_chunking.toml` | 切片策略 | `recursive-420`（基线）、`fixed-256`、`heading-600` |
| `configs/experiments/ab_retrieval.toml` | TopK / 混合召回 / 重排 | `baseline-hybrid-rerank`、`dense-only-top3`、`hybrid-top10-no-rerank` |
| `configs/experiments/ab_prompt.toml` | Prompt 模板 | `grounded-cite`（基线）、`concise`、`strict-refusal` |
| `configs/experiments/ab_model.toml` | 生成模型（可选） | `deepseek-chat`、`qwen-plus`、`offline-extractive` |

`ab_model.toml` 需要 `RAG_EVAL_API_KEY` 才能对比真实模型；无密钥时两个在线变体会降级为离线生成器，结果趋同。

### 8.3 运行

```powershell
python -m rag_eval ab --experiment configs/experiments/ab_chunking.toml --config configs/baseline.toml
python -m rag_eval ab --experiment configs/experiments/ab_retrieval.toml
python -m rag_eval ab --experiment configs/experiments/ab_prompt.toml
# 可选：需要 API Key
python -m rag_eval ab --experiment configs/experiments/ab_model.toml
```

产物：`artifacts/experiments/<experiment_id>/experiment.json`，以及 SQLite `experiments` 表。
每个变体同时是一条独立 run，可用 `runs`、`report`、`compare` 单独查看。

## 9. 看板与 HTTP API

`python -m rag_eval serve` 默认用**标准库** `ThreadingHTTPServer` 提供页面与接口；
安装 fastapi + uvicorn 后，同一套处理逻辑以 ASGI 应用（`rag_eval/fastapi_app.py`）提供，
也可 `uvicorn rag_eval.fastapi_app:app --port 8000`。

### 9.1 看板

浏览器打开 `http://<host>:<port>/`，静态看板使用 hash 路由，共五个页面：

| 路由 | 页面 | 内容 |
| --- | --- | --- |
| `#/overview` | 评测总览 | 运行数、实验数、用例数、最新运行、关键指标趋势 |
| `#/runs` | 运行记录 | 运行列表、模式、后端、headline 指标、坏例数、详情与导出 |
| `#/run/{run_id}` | 运行详情 | 指标卡、拆分统计、坏例列表、逐用例答案与引用、报告导出 |
| `#/compare` | A/B 对比 | 两次运行的指标差值、显著性、逐用例变化（可带 `?left=&right=` 直达） |
| `#/experiments` | 实验管理 | 实验列表、变体排名、冠军、触发新实验 |
| `#/experiment/{id}` | 实验详情 | 各变体指标、与基线的显著差异、产物路径 |
| `#/datasets` | 用例与指标 | 测试集清单、指标口径说明 |

#### 界面预览

下列截图由本项目在离线模式下实际运行后抓取（原图在 `artifacts/shots/`）。

**评测总览**（运行数、实验数、关键指标趋势、最新运行）

![总览](artifacts/shots/overview.png)

**运行详情**（指标卡、分集合表现、带置信区间的全部指标、坏例清单）

![运行详情](artifacts/shots/run.png)

**A/B 对比**（综合分、显著改善/退化、p 值与 Cohen's d、逐用例变化）

![对比](artifacts/shots/compare.png)

**导出的 HTML 报告**（单文件自包含，可离线归档）

![报告](artifacts/shots/report-html.png)

### 9.2 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查：版本、模式、各后端可用性、降级告警、运行数 |
| GET | `/api/overview` | 汇总：运行/实验/用例计数、最新运行、指标趋势 |
| GET | `/api/runs` | 运行列表（`limit`、`experiment_id`） |
| GET | `/api/runs/{run_id}` | 运行详情：指标、评分卡、拆分、坏例 |
| GET | `/api/runs/{run_id}/cases` | 逐用例结果（`split`、`flagged`、`limit`） |
| GET | `/api/runs/{run_id}/cases/{case_id}` | 单用例详情（答案、引用、片段、逐句支撑） |
| GET | `/api/compare?left=&right=` | 两次运行对比：有逐用例数据时做配对检验，否则退化为指标差值 |
| GET | `/api/experiments` | 实验列表 |
| GET | `/api/experiments/{id}` | 实验详情（含各变体 run） |
| GET | `/api/metrics` | 指标注册表与默认指标名单 |
| GET | `/api/datasets` | 测试集清单与拆分统计 |
| GET | `/api/report?run_id=&format=` | 导出报告文件（`html`/`markdown`/`csv`/`json`/`xlsx`） |
| POST | `/api/eval` | 提交评测任务（`dataset`、`name`、`splits`、`max_cases`、`use_judge`、`config`），返回 202 与 `job_id` |
| GET | `/api/jobs`、`/api/jobs/{job_id}` | 后台任务进度与结果 |
| GET | `/api/config` | 当前配置与指纹 |
| DELETE | `/api/runs/{run_id}` | 删除一次运行 |
| GET | `/`、`/static/*` | 静态看板资源 |

报告导出落在 `artifacts/reports/<run_id>/`：`report.html`、`report.md`、`report.json`、
`metrics.csv`、`cases.csv`、`report.xlsx`（按请求的格式生成）。`report.html` 为单文件
自包含页面（内联样式、无 CDN），可直接归档或邮件发送；未安装 `openpyxl` 时仅跳过
XLSX 并在 stderr 给出警告，其余格式照常导出。

## 10. 切换真实后端

核心代码不需要改，只装依赖 + 改配置 + 设密钥。

### 10.1 安装依赖

```powershell
python -m pip install -r requirements-optional.txt
```

只装需要的子集也可以：

```powershell
python -m pip install chromadb                          # 向量库
python -m pip install sentence-transformers             # BGE 向量与重排
python -m pip install pypdf python-docx                 # PDF / DOCX 加载
python -m pip install openpyxl PyYAML                   # XLSX 报告 / YAML 配置
python -m pip install ragas deepeval langchain-openai datasets   # 外部框架复算
```

### 10.2 配置密钥

```powershell
# PowerShell（当前会话）
$env:RAG_EVAL_API_KEY = "sk-..."

# bash / zsh
export RAG_EVAL_API_KEY="sk-..."
```

代码按 `generate.api_key_env`（默认 `RAG_EVAL_API_KEY`）读取，其次回退 `OPENAI_API_KEY`。
密钥只从环境变量读取，不写入配置文件、不落盘到运行记录。
直接以 `uvicorn rag_eval.fastapi_app:app` 方式启动时，还会读取 `RAG_EVAL_CONFIG`（默认
`configs/baseline.toml`）与 `RAG_EVAL_PORT`（默认 8000）；CLI 的 `serve` 用命令行参数，不读这两个变量。

### 10.3 切换配置

```toml
[embedding]
provider = "bge"
model = "BAAI/bge-small-zh-v1.5"
query_prefix = "为这个句子生成表示以用于检索相关文章："

[store]
provider = "chroma"            # 或 "milvus"
persist_dir = "artifacts/index-chroma"
# milvus_uri = "http://localhost:19530"

[rerank]
provider = "bge"
model = "BAAI/bge-reranker-base"

[generate]
provider = "openai"
model = "deepseek-chat"
base_url = "https://api.deepseek.com/v1"

[evaluate]
judge_provider = "auto"
ragas = true
deepeval = true
```

切换后用 `python -m rag_eval doctor` 确认；运行记录的 `mode` 应为 `live`（BGE 与 LLM 同时生效）
或 `hybrid`（仅其一），`providers` 字段会显示实际后端。

### 10.4 降级行为

| 配置 | 依赖/凭据缺失时 |
| --- | --- |
| `embedding.provider="auto"` | 降级哈希向量（`bge` 显式指定则报错） |
| `store.provider="auto"` | 降级内存向量库（`chroma` 显式指定则报错） |
| `rerank.provider="auto"` | 降级词法重排；运行期模型失败也退回词法 |
| `generate.provider="auto"` / `"openai"` | 无密钥 → 离线抽取式；调用失败 → 该用例回退，`provider` 标记 `offline-extractive(fallback)` |
| `evaluate.judge_provider="auto"` | 无 LLM → 启发式量规 |
| `evaluate.ragas/deepeval=true` | 缺依赖或缺密钥 → 适配器标记不可用，主流程继续 |

## 11. 测试

```powershell
# 单元测试（需 pytest；tests/ 目录存在时生效）
python -m pytest -q
# 或
make test

# 离线链路自检（不依赖第三方包；失败时返回非零退出码）
python -m rag_eval selftest
```

测试建议覆盖：分句/分词、切片边界、RRF 排序、指标判定（拒答、引用、支撑度）、
配置校验与覆盖、数据集去重、Bootstrap 区间、报告渲染。`pyproject.toml` 已配置
`testpaths = ["tests"]`、`addopts = "-q"`，ruff 行宽 120、目标 py311。

## 12. Docker

镜像基于 `python:3.12-slim`，多阶段构建，非 root 运行，暴露 8000 端口，
入口为 `python -m rag_eval serve --host 0.0.0.0 --port 8000`，内置 `HEALTHCHECK`（标准库 urllib 探活）。

```powershell
# 构建（上下文为仓库根目录）
docker build -f docker/Dockerfile -t kb-rag-eval:1.0.0 .

# 运行（挂载语料、产物、配置）
docker run --rm -p 8000:8000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/artifacts:/app/artifacts" `
  -v "${PWD}/configs:/app/configs" `
  --env-file .env `
  kb-rag-eval:1.0.0

# 或使用 compose（api 服务 + 可选 chromadb / milvus，默认注释）
Copy-Item .env.example .env
docker compose -f docker/docker-compose.yml up -d --build
```

- `docker/docker-compose.yml` 的 `api` 服务使用 `../data`、`../artifacts`、`../configs` 绑定挂载，
  端口 `8000:8000`，环境变量来自 `.env`；
- 容器以非 root 的 `app`（uid/gid 10001）运行：Linux 主机上绑定挂载的目录需对该 uid 可写，
  否则改用命名卷，或以 `--user "$(id -u):$(id -g)"` 覆盖；
- `chromadb`（`chromadb/chroma:0.5.5`，8001:8000）与 `milvus`（`milvusdb/milvus:v2.4.13`，19530/9091）
  服务已给出，默认注释，按需启用；
- `.dockerignore` 位于 `docker/`：Docker 只读取构建上下文根目录的忽略文件，Dockerfile 已显式 COPY
  所需目录，因此忽略清单主要作为部署约定列出。

## 13. 常见问题

**为什么忠实度恒为 1.0？**

离线生成器是**抽取式**的：它从检索到的片段里挑选证据句，原样拼接并逐句标注引用
（`generation.py` 的 `_offline`）。既然答案的每个事实句都直接来自某个片段，逐句支撑度自然超过阈值，
`faithfulness` 就是 1.0，`hallucination_rate` 也就是 0.0。这是「抽取式生成不产生无依据断言」的必然结果，
不是评测指标失效。要衡量真实生成模型的忠实度，需切换到 live 模式（配置 API Key 与 `generate.provider="openai"`），
此时答案由模型自由组织，是否忠实才有区分度。同理，离线模式下的 `citation_accuracy` 也接近满分，
因为引用是生成时按片段编号直接挂上的。

**`doctor` 显示 BGE 不可用，是不是坏了？**

不是。`auto` 的语义是「有则用、无则降级」。哈希向量是字符 n-gram 特征哈希，检索行为确定、可复现，
适合离线与 CI；它的语义泛化能力弱于 BGE，因此跨实验比较时必须在同一模式（都 offline 或都 live）下进行。

**为什么 `precision_at_k` 很低，但 `hit_rate` 是 1.0？**

TopK 通常为 6，而一个问题真正相关的片段可能只有 1~2 条，精确率上限本就不高。精确率低只说明
「多召回了几条」，不代表答案错误；坏例规则因此不把 `precision_at_k` 作为关键信号。

**为什么设置了 API Key 仍是 offline？**

按顺序排查：变量名是否为 `RAG_EVAL_API_KEY`（或 `OPENAI_API_KEY`）；是否在**当前** shell 会话中设置；
`python -m rag_eval doctor` 的 `api_key_present` 是否为真；`generate.provider` 是否被设为 `offline`。
容器部署时确认 `.env` 已挂载/传入。

**为什么 `answer_relevancy` 只有 0.6 左右？**

该指标由「问题词覆盖（0.55）+ 答案类型匹配（0.30）+ 长度因子（0.15）」加权；摘句式答案往往
只覆盖问题的一部分词，因此分数中等。这是指标设计意图：区分「答到点上」与「抄了一段相关文字」。

**两次运行的 `compare` 有没有 p 值？**

取决于数据是否完整：两次运行的逐用例结果都在 SQLite 中时，`compare` 会按 `case_id` 配对并做
Bootstrap 检验（`mode: paired`，有 p 值与效应量）；若只有指标记录、缺少逐用例数据，
则退化为指标差值（`mode: record-only`，`p_value` 为 `null`）。
同一次 `ab` 实验内的变体比较始终是配对检验。

**YAML 配置报错「需要安装 PyYAML」？**

YAML 解析是可选能力；改回 TOML（推荐）或 `pip install PyYAML`。

**PDF / DOCX 加载失败？**

分别需要 `pypdf` 与 `python-docx`，错误信息中会给出对应的安装命令；也可先把文档转成 Markdown/TXT。

**Windows 控制台中文乱码？**

文件本身为 UTF-8；建议 `chcp 65001`，或设置 `PYTHONUTF8=1` 后重跑。报告与看板不受影响。

**索引什么时候重建？**

语料内容、切片参数或向量器签名变化时指纹变化，会新建索引目录；`ingest --force` 强制重建。
索引目录不会被覆盖，旧索引可保留用于回放。

**怎么新增测试用例？**

在 `data/testsets/` 下的 JSONL 中追加一行，至少包含 `question`，建议提供 `case_id`、`split`、
`reference_answer`、`reference_contexts`、`expected_keywords`、`tags`、`difficulty`、
`should_refuse`、`adversarial_type`、`source`。`case_id` 重复且问题不同时会自动加哈希后缀，不会静默丢用例。
也可以用 `rag_eval/synth.py` 从切片合成候选（`source="synthetic"`、`tags` 含 `needs-review`），
人工改写后再合并进正式集（`dataset.merge_sets` 以人工版本优先）。

## 14. 路线图

1. **指标增强**：claim-level 蕴含判定、跨句推理支撑、检索增益类指标（nDCG@k 的 graded relevance）。
2. **Judge 校准**：LLM Judge 与人工标注的一致性评估（Kappa / 相关性），支持多 Judge 投票与置信度加权。
3. **数据工程**：测试集版本化与变更审计、按标签分层抽样、合成用例的人工复核工作流。
4. **检索能力**：查询改写 / HyDE、父子切片、按文档类型差异化切片、增量索引。
5. **规模化**：分布式执行器、大语料下的 ANN 索引参数调优、批处理与并发限流。
6. **可观测性**：OpenTelemetry 追踪、按阶段的成本归因、告警与回归门禁集成（CI 阈值）。
7. **报告**：多运行趋势对比报告、按标签/难度的分维度汇总、可交互的坏例复盘页。
8. **部署**：Kubernetes 清单、只读模式、多租户与权限控制。

## 15. 参考

- 指标定义：[`docs/metrics.md`](docs/metrics.md)
- 架构与降级：[`docs/architecture.md`](docs/architecture.md)
- A/B 方法：[`docs/experiments.md`](docs/experiments.md)
- 基线配置：[`configs/baseline.toml`](configs/baseline.toml)
- 实验配置：[`configs/experiments/`](configs/experiments/)
- 测试集：[`data/testsets/`](data/testsets/)
