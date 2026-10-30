# A/B 实验方法

`kb-rag-eval` 把「改一个变量、在同一批测试用例上对比」做成一等公民：实验由 TOML 声明，
引擎负责生成各变体配置、隔离索引、跑完整评测、做配对显著性检验，并给出冠军与排名。

## 1. 实验配置结构

实验文件由 `Config.load_experiment()` 解析，仅支持 `.toml` / `.json`，顶层是 `[experiment]` 表
（JSON 可直接给对象，或包一层 `experiment`）：

| 字段 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| `name` | 是 | — | 实验名，同时进入实验 ID |
| `dataset` | 是 | — | 测试集路径（单文件或目录） |
| `variants` | 是 | — | 至少 2 个，`name` 唯一 |
| `baseline` | 否 | `configs/baseline.toml` | 基线配置文件路径；实际对照取「第一个成功运行的变体」 |
| `description` | 否 | 空 | 实验说明 |
| `seed` | 否 | 42 | 实验种子（声明项；运行时随机种子取自基线配置的 `project.seed`） |
| `output_dir` | 否 | `artifacts/experiments` | 实验产物根目录 |
| `repeat` | 否 | 1 | 重复次数（当前版本记为配置项，不改变执行次数） |

每个 `[[experiment.variants]]` 支持三个字段：`name`（必填）、`description`、`overrides`。
校验规则来自 `ExperimentCfg.from_dict` / `VariantCfg.from_dict`：未知字段直接报错（避免拼写错误被静默忽略）、
变体数 < 2 报错、名称重复报错、`overrides` 必须是表。

## 2. 覆盖（overrides）如何生效

```toml
[[experiment.variants]]
name = "fixed-256"
description = "固定窗口 256 字，重叠 40"

[experiment.variants.overrides.ingest]
chunker = "fixed"
chunk_size = 256
chunk_overlap = 40
```

`Config.apply_overrides()` 的语义是**按 section 覆盖键**：

1. 以基线配置 `to_dict()` 为底；
2. 对 `overrides` 中的每个 section，用其键值更新该 section 的同名字段；
3. 合并结果重新走 `Config.from_dict()`，因此会再次执行完整校验（枚举取值、`top_k ≤ candidate_k`、
   `chunk_overlap < chunk_size` 等）；
4. 返回新配置（原配置不变），并计算新的配置指纹。

注意：覆盖粒度是「section 内的键」，值是整体替换而非递归合并。因此 `flag_rules` 这类字典字段
需要写全量，不能只写要改的那一项。未知 section 会抛 `ConfigError`，不会被忽略。

## 3. 索引隔离与复用

索引目录由内容指纹决定（见 `docs/architecture.md` 第 3.1 节）：

- 切片参数不同的变体（如 `recursive` 420 / `fixed` 256 / `heading` 600）指纹不同，各自建独立索引，
  互不污染，也不会互相覆盖；
- TopK、Prompt、生成模型等**不影响切片与向量**的变体，指纹相同，直接复用同一份索引；
- 引擎内部再按配置指纹缓存 `RAGPipeline` 实例：两个变体合并后配置完全相同时不会重复建索引；
- 重复运行同一实验时，索引命中磁盘缓存（`index.meta.json` 的 `signature` 校验通过），只有首次需要编码全量切片。

这条设计让「切片实验」和「检索实验」的耗时结构完全不同：前者每组都要重建索引，后者只在第一次建。

## 4. 执行流程

`experiments.run_experiment()` 的顺序：

```
读取实验配置 → 载入测试集（去重后固定顺序）
  └─ 对每个 variant：
       cfg   = baseline.apply_overrides(variant.overrides)
       pipe  = 缓存命中 ? 复用 : RAGPipeline(cfg, build_index(cfg))
       run   = run_evaluation(cfg, 同一批 cases, RunOptions(variant=name, experiment_id=...))
       记录 run.record.metadata["overrides"] = variant.overrides
  └─ 取第一个成功的变体为对照，其余每个变体与之做配对比较
  └─ 汇总排名 / 冠军 / 显著改善与显著回退计数
  └─ 写 artifacts/experiments/<experiment_id>/experiment.json + SQLite experiments 表
```

容错策略：

- 单个变体抛异常时记为 `error` 并继续跑后续变体，实验状态标记为 `partial`；
- 对照取「第一个成功运行的变体」而不是配置里的 `baseline` 字段，避免基线本身崩溃导致整个实验无对照；
- 若所有变体都失败，`summary.winner` 为空字符串。

## 5. 统计方法

### 5.1 配对 Bootstrap 显著性检验

同一批用例、同一提示词，只有配置不同，因此使用**配对**设计：对每个用例取两组指标值之差，
对差值做有放回重采样，统计「重采样均值绝对值 ≥ 观测差值绝对值」的比例作为双侧 p 值
（`stats.paired_bootstrap_pvalue`）。

- 重采样次数：`max(500, evaluate.bootstrap_samples)`，默认 1000 → 至少 1000 次；
- 显著性阈值：`evaluate.significance_alpha`（默认 0.05）；
- 两组差值全为 0 时 p 值直接返回 1.0，避免出现「零方差导致假显著」；
- 配对设计的好处：用例难度带来的方差被差分抵消，几十条用例也能看出方向性差异。

### 5.2 效应量

`stats.cohens_d` 用两组均值差除以合并标准差。p 值回答「是否可能是偶然」，Cohen's d 回答「差异有多大」。
经验参考：|d| < 0.2 可忽略，0.2~0.5 小，0.5~0.8 中，> 0.8 大。

### 5.3 综合评分与冠军

`experiments.COMPOSITE_METRICS` 固定为 8 个关键质量指标：

```
hit_rate, context_recall, faithfulness, answer_relevancy,
answer_completeness, citation_accuracy, refusal_appropriateness, hallucination_rate
```

综合分 = 这些指标的方向归一化值的算术平均：`higher` 取原值，`lower` 取 `1 − 原值`，
缺失指标跳过。冠军取有效变体中综合分最高者，`summary.ranking` 给出完整排序。

综合分只用于**排序**，不是质量结论：它把 8 个不同量纲的指标等权平均，可能掩盖「某一维度严重退化」。
判断某变体是否真的更好，应同时看：

1. 是否有指标显著改善（`comparisons[].improved`）；
2. 是否有指标显著回退（`comparisons[].degraded`）；
3. 关键安全指标是否保持 1.0；
4. 延迟与成本是否可接受。

### 5.4 结果字段

每个 `comparisons[]` 元素包含：

| 字段 | 含义 |
| --- | --- |
| `left` / `right` | 对照与被测变体的 run_id、variant、name |
| `n_cases` | 参与配对比较的用例数（两组交集） |
| `deltas[]` | 每个指标的 left/right/delta/relative/p_value/significant/effect_size/better/n |
| `improved` / `degraded` | 显著变好、显著变差的指标列表 |
| `case_diffs[]` | 综合质量变化最大的 20 个用例（case_id、left、right、delta），用于人工比对答案 |
| `composite` | 两组的综合分 |
| `variant` | 被测变体名 |

`experiment.json` 顶层另有 `summary`：冠军、排名、各变体 headline 指标、耗时、显著改善/回退计数。

## 6. 随仓库交付的实验

### 6.1 `configs/experiments/ab_chunking.toml`（切片策略）

数据集 `data/testsets/core.jsonl`，基线 `configs/baseline.toml`。

| 变体 | 覆盖项 | 观察点 |
| --- | --- | --- |
| `recursive-420` | 无（基线：递归切片 420 字，重叠 60） | 语义边界完整，长答案更易被单片段覆盖 |
| `fixed-256` | `ingest.chunker=fixed`、`chunk_size=256`、`chunk_overlap=40` | 粒度更细，`precision_at_k` 通常上升、上下文完整性下降 |
| `heading-600` | `ingest.chunker=heading`、`chunk_size=600`、`chunk_overlap=80` | 章节前缀进入切片，标题相关问题的 `hit_rate` / `mrr` 通常更好 |

### 6.2 `configs/experiments/ab_retrieval.toml`（召回与重排）

| 变体 | 覆盖项 | 观察点 |
| --- | --- | --- |
| `baseline-hybrid-rerank` | 无（混合召回 candidate 24，重排 top_n=4） | 质量上限最高，延迟也最高 |
| `dense-only-top3` | `retrieve.top_k=3`、`candidate_k=3`、`hybrid=false`；`rerank.provider=none`、`top_n=3` | 低延迟下限，观察 `context_recall` 与 `latency_p95` 的取舍 |
| `hybrid-top10-no-rerank` | `retrieve.top_k=10`、`candidate_k=30`；`rerank.provider=none`、`top_n=10` | 高召回场景，观察 `context_redundancy` 是否恶化 |

### 6.3 `configs/experiments/ab_prompt.toml`（Prompt 模板）

| 变体 | 覆盖项 | 观察点 |
| --- | --- | --- |
| `grounded-cite` | 无（`grounded_cite_v1`：句级引用、信息不足即拒答） | 引用覆盖与拒答恰当性的平衡点 |
| `concise` | `generate.prompt=concise_v1` | 更短答案提升 `answer_relevancy`，可能牺牲 `answer_completeness` |
| `strict-refusal` | `generate.prompt=strict_v2`、`refusal_policy=strict` | 拒答更多：`refusal_appropriateness` 与 `refusal_rate` 需要联合看 |

Prompt 的 `style` 同时驱动在线与离线生成器（在线写入指令，离线按 style 近似执行：句数上限、
证据分阈值），因此即便在离线模式，这个实验也能产生可比较的行为差异。

### 6.4 `configs/experiments/ab_model.toml`（可选，需要 API Key）

| 变体 | 覆盖项 |
| --- | --- |
| `deepseek-chat` | `generate.provider=openai`、`model=deepseek-chat`、`base_url=https://api.deepseek.com/v1` |
| `qwen-plus` | `generate.provider=openai`、`model=qwen-plus`、`base_url=https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `offline-extractive` | `generate.provider=offline` |

该实验在无密钥时也能跑：两个在线变体会因缺少密钥自动降级到离线生成器（`provider` 记为
`offline-extractive(fallback)`），三组结果趋同。要做有意义的模型对比，必须先设置 `RAG_EVAL_API_KEY`
（或 `OPENAI_API_KEY`），并用 `python -m rag_eval doctor` 确认 LLM 可用。

## 7. 运行与查看

```bash
# 运行实验（CLI 接口约定）
python -m rag_eval ab --experiment configs/experiments/ab_chunking.toml --config configs/baseline.toml

# 查看历史运行与实验
python -m rag_eval runs --limit 20
```

产物：

- `artifacts/experiments/<experiment_id>/experiment.json`：完整实验记录（变体、比较、汇总）；
- SQLite `experiments` 表：状态、变体摘要、汇总，供看板 `/api/experiments` 与 `/api/experiments/{id}` 读取；
- 每个变体同时是一条独立 run，可在 `artifacts/runs/` 与 `/api/runs` 中检索。

跨实验的两次运行对比用 `compare`：

```bash
python -m rag_eval compare --left <run_id_a> --right <run_id_b>
```

它从 SQLite 读取两次运行的逐用例结果，按 `case_id` 配对后做配对 Bootstrap 检验
（`api.compare_persisted`，结果 `mode: paired`），因此有 p 值与显著性结论；
若某次运行没有逐用例数据（例如只保留了指标记录），则退化为指标级差值
（`experiments.compare_records`，`mode: record-only`，`p_value` 与 `significant` 为 `null`）。

## 8. 实验设计建议

1. **一次只改一个维度。** 切片、TopK、Prompt 同时改，即使综合分提升也无法归因。
2. **用核心集做回归，用对抗集做安全验证。** 切片与检索实验建议用 `core.jsonl`；
   Prompt 与拒答策略实验建议 `core.jsonl + adversarial.jsonl`（可用 `--splits` 过滤）。
3. **样本量要诚实。** 核心集 12 例、全集 30 例属于「冒烟级」规模，p 值容易不稳定；
   结论应表述为「在该测试集上观察到」，而不是普遍结论。扩充用例后再复测更可靠。
4. **关注回退。** 只看 `improved` 会高估收益；`degraded` 列表里往往藏着真实的取舍。
5. **延迟与成本是硬约束。** 检索类变体的质量提升通常伴随 `latency_p95` 上升，
   上线判断应把性能指标和综合分放在一起看。
6. **复现性。** 固定 `seed`、固定测试集文件、固定配置后，指标与置信区间应可复现；
   若不可复现，先检查是否有未固定顺序的外部依赖（如在线模型的温度）。

## 9. 常见问题

**为什么两个变体的指标完全一样？**
检查 `overrides` 是否写在了正确的 section 下（例如把 `chunk_size` 写到 `[retrieve]` 不会生效，
因为该字段属于 `[ingest]`）。未知字段会报错，但「写到合法 section 的错误键」不会改变行为。

**为什么 p 值总是 1.0？**
两组逐用例差值全为 0 时（配置未实际生效、或两组输出完全相同）会直接返回 1.0。

**为什么综合分排序与某个单项指标排序不一致？**
综合分是 8 个指标方向归一化后的等权平均，单项极值会被平均掉；以显著性与业务约束为准。

**为什么实验状态是 `partial`？**
至少一个变体抛异常（如 Milvus 连接失败、BGE 权重下载失败且被显式指定为 `bge`）。
每个变体的 `error` 字段记录了异常类型与消息。
