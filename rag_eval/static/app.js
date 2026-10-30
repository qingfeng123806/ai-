/* RAG 评测看板前端：原生 JS + 内联 SVG，无构建、无 CDN 依赖。 */
(() => {
  "use strict";

  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
  const view = () => $("#view");

  const METRIC_LABELS = {
    hit_rate: "命中率", precision_at_k: "检索精确率", recall_at_k: "检索召回率",
    mrr: "MRR", ndcg: "nDCG", context_precision: "上下文精确率", context_recall: "上下文召回率",
    context_redundancy: "上下文冗余度", faithfulness: "忠实度", answer_relevancy: "答案相关性",
    answer_completeness: "答案完整性", hallucination_rate: "幻觉率", answer_repetition: "重复度",
    citation_accuracy: "引用准确率", citation_coverage: "引用覆盖率", reference_present: "引用存在率",
    judge_faithfulness: "Judge 忠实度", judge_relevance: "Judge 相关性", judge_completeness: "Judge 完整性",
    judge_citation_quality: "Judge 引用质量", judge_overall: "Judge 综合",
    refusal_appropriateness: "拒答恰当性", prompt_injection_resistance: "注入抵抗力",
    pii_leak_rate: "PII 泄露率", toxicity_rate: "有害内容率",
    latency_p50: "延迟 P50", latency_p95: "延迟 P95", latency_retrieval_p95: "检索延迟 P95",
    tokens_per_answer: "平均 Token", cost_per_query: "单次成本", error_rate: "错误率", refusal_rate: "拒答率",
  };
  const GROUP_LABELS = {
    retrieval: "检索质量", generation: "生成质量", citation: "引用质量", judge: "LLM-as-Judge",
    safety: "安全合规", performance: "性能与成本", external: "外部框架复算",
  };
  const SPLIT_LABELS = { core: "核心回归集", edge: "边缘场景集", adversarial: "对抗集" };
  const GROUP_ORDER = ["retrieval", "generation", "citation", "judge", "safety", "performance", "external"];

  const esc = (value) =>
    String(value == null ? "" : value).replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));

  const pct = (value, digits = 1) =>
    value == null || Number.isNaN(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;

  const num = (value, digits = 2) =>
    value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);

  function fmtMetric(metric) {
    if (!metric) return "—";
    const name = metric.name || "";
    const value = Number(metric.value);
    if (name.startsWith("latency")) return `${num(value, 1)} ms`;
    if (name === "tokens_per_answer") return num(value, 0);
    if (name === "cost_per_query") return `$${Number(value).toFixed(5)}`;
    return pct(value);
  }

  function label(name) { return METRIC_LABELS[name] || name; }

  async function getJSON(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).error || detail; } catch (_) {}
      throw new Error(detail);
    }
    return response.json();
  }

  let toastTimer = null;
  function toast(message, isError = false) {
    let el = $(".toast");
    if (!el) {
      el = document.createElement("div");
      el.className = "toast";
      document.body.appendChild(el);
    }
    el.textContent = message;
    el.className = `toast show${isError ? " err" : ""}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = "toast"; }, 3600);
  }

  function spinner(text = "加载中…") {
    return `<div class="empty"><span class="spinner"></span> ${esc(text)}</div>`;
  }

  function empty(text) { return `<div class="empty">${esc(text)}</div>`; }

  function pill(text, kind = "") { return `<span class="pill ${kind}">${esc(text)}</span>`; }

  // ---------------------------------------------------------------- 图表
  function lineChart(points, { width = 760, height = 170, keys, colors } = {}) {
    if (!points.length) return empty("暂无趋势数据");
    const pad = { l: 38, r: 14, t: 12, b: 18 };
    const w = width - pad.l - pad.r;
    const h = height - pad.t - pad.b;
    const series = keys.map((key, idx) => ({
      key,
      color: colors[idx] || "#5aa9ff",
      values: points.map((p) => (p[key] == null ? null : Number(p[key]))),
    }));
    const yMax = 1;
    const x = (i) => pad.l + (points.length === 1 ? w / 2 : (i / (points.length - 1)) * w);
    const y = (v) => pad.t + h - (v / yMax) * h;
    let svg = `<svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">`;
    for (let g = 0; g <= 4; g++) {
      const yy = pad.t + (h / 4) * g;
      svg += `<line x1="${pad.l}" y1="${yy}" x2="${width - pad.r}" y2="${yy}" stroke="#1f293d" stroke-width="1"/>`;
      svg += `<text x="6" y="${yy + 3}">${Math.round((1 - g / 4) * 100)}%</text>`;
    }
    series.forEach((s) => {
      const segments = [];
      s.values.forEach((v, i) => {
        if (v == null) return;
        segments.push(`${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`);
      });
      if (segments.length) {
        svg += `<path d="${segments.join(" ")}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
      }
      s.values.forEach((v, i) => {
        if (v == null) return;
        svg += `<circle cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="3" fill="${s.color}"><title>${esc(points[i].name)} · ${esc(s.key)} = ${(v * 100).toFixed(1)}%</title></circle>`;
      });
    });
    svg += "</svg>";
    return svg;
  }

  function chartLegend(keys, colors, labels) {
    return `<div class="row" style="gap:16px;margin-top:10px">${keys.map((key, idx) =>
      `<span class="small muted"><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${colors[idx]};margin-right:6px"></span>${esc(labels[idx])}</span>`
    ).join("")}</div>`;
  }

  function barCell(value, low, high) {
    const v = Math.max(0, Math.min(1, Number(value) || 0));
    const lo = Math.max(0, Math.min(1, Number(low ?? v)));
    const hi = Math.max(0, Math.min(1, Number(high ?? v)));
    return `<div class="ci-bar"><i style="left:${(lo * 100).toFixed(1)}%;width:${Math.max(1.2, (hi - lo) * 100).toFixed(1)}%"></i></div>`;
  }

  // ---------------------------------------------------------------- 路由
  const routes = [
    [/^#\/?$/, () => renderOverview()],
    [/^#\/overview$/, () => renderOverview()],
    [/^#\/runs$/, () => renderRuns()],
    [/^#\/run\/([^/]+)$/, (m) => renderRun(m[1])],
    [/^#\/compare$/, () => renderCompare()],
    [/^#\/experiments$/, () => renderExperiments()],
    [/^#\/experiment\/([^/]+)$/, (m) => renderExperiment(m[1])],
    [/^#\/datasets$/, () => renderDatasets()],
  ];

  async function router() {
    const raw = location.hash || "#/overview";
    // hash 中可能带查询串（如 #/compare?left=..&right=..），路由匹配只看路径部分
    const hash = raw.split("?")[0];
    $$("nav a").forEach((a) => a.classList.toggle("active", hash.startsWith(a.getAttribute("href"))));
    for (const [pattern, handler] of routes) {
      const match = hash.match(pattern);
      if (match) {
        view().innerHTML = spinner();
        try { await handler(match); } catch (exc) { view().innerHTML = empty(`加载失败：${exc.message}`); }
        return;
      }
    }
    view().innerHTML = empty("页面不存在");
  }

  // ---------------------------------------------------------------- 总览
  async function renderOverview() {
    const [overview, health] = await Promise.all([getJSON("/api/overview"), getJSON("/api/health")]);
    const latest = overview.latest_run;
    const headline = latest ? latest.metrics.filter((m) => ["faithfulness", "answer_relevancy", "context_recall", "citation_accuracy"].includes(m.name)) : [];
    const warns = (health.warnings || []).map((w) => `<div class="small muted">· ${esc(w)}</div>`).join("");

    view().innerHTML = `
      <div class="page-head">
        <div>
          <h1>评测总览</h1>
          <div class="sub">检索质量 · 生成质量 · 安全合规 · 性能延迟，统一量化并支持 A/B 对比</div>
        </div>
        <div class="row">
          ${pill(`模式 ${health.mode}`, "info")}
          ${pill(`v${health.version}`)}
          <button class="btn primary" id="btn-new-run">发起评测</button>
        </div>
      </div>
      ${warns ? `<div class="panel"><h2>运行环境降级提示<span class="hint">离线模式下使用内置确定性后端</span></h2>${warns}</div>` : ""}
      <div class="cards">
        <div class="card"><div class="k">评测运行</div><div class="v">${overview.runs}</div><div class="d">累计 run 数</div></div>
        <div class="card"><div class="k">A/B 实验</div><div class="v">${overview.experiments}</div><div class="d">对比实验数</div></div>
        <div class="card"><div class="k">用例结果</div><div class="v">${overview.case_results}</div><div class="d">已评测用例</div></div>
        <div class="card"><div class="k">最新运行</div><div class="v" style="font-size:15px">${latest ? esc(latest.variant || latest.name) : "—"}</div><div class="d">${latest ? esc(latest.created_at.replace("T", " ").slice(0, 19)) : "尚无运行"}</div></div>
      </div>

      <div class="grid-2">
        <div class="panel">
          <h2>核心指标趋势<span class="hint">按时间顺序，最近 ${overview.trend.length} 次运行</span></h2>
          ${lineChart(overview.trend, {
            keys: ["faithfulness", "context_recall", "answer_relevancy"],
            colors: ["#3ddc97", "#5aa9ff", "#ffc857"],
          })}
          ${chartLegend(["faithfulness", "context_recall", "answer_relevancy"], ["#3ddc97", "#5aa9ff", "#ffc857"], ["忠实度", "上下文召回率", "答案相关性"])}
        </div>
        <div class="panel">
          <h2>最新运行<span class="hint">${latest ? esc(latest.run_id) : ""}</span></h2>
          ${latest ? `<div class="cards" style="grid-template-columns:1fr 1fr;margin:0">
            ${headline.map((m) => `<div class="card"><div class="k">${esc(label(m.name))}</div><div class="v" style="font-size:20px">${fmtMetric(m)}</div>
              <div class="bar"><i style="width:${(Number(m.value) * 100).toFixed(1)}%"></i></div></div>`).join("")}
          </div>
          <div class="row" style="margin-top:14px">
            <a class="btn" href="#/run/${esc(latest.run_id)}">查看详情</a>
          </div>` : empty("尚未运行评测：执行 python -m rag_eval eval 或在看板发起评测")}
        </div>
      </div>

      <div class="panel">
        <h2>最近运行</h2>
        ${runsTable((await getJSON("/api/runs?limit=8")).runs)}
      </div>
    `;
    $("#btn-new-run").addEventListener("click", openNewRunModal);
  }

  function runsTable(runs) {
    if (!runs || !runs.length) return empty("暂无运行记录");
    return `<table>
      <thead><tr><th>运行</th><th>变体</th><th>数据集</th><th class="num">用例</th><th class="num">坏例</th>
      <th class="num">忠实度</th><th class="num">相关性</th><th class="num">上下文召回</th><th>模式</th><th>时间</th></tr></thead>
      <tbody>
      ${runs.map((r) => {
        const h = r.headline || {};
        return `<tr>
          <td><a href="#/run/${esc(r.run_id)}">${esc(r.name)}</a><div class="mono">${esc(r.run_id)}</div></td>
          <td>${esc(r.variant || "—")}</td>
          <td class="mono">${esc((r.dataset || "").split("/").pop())}</td>
          <td class="num">${r.n_cases}</td>
          <td class="num">${r.badcases ? `<span class="pill bad">${r.badcases}</span>` : `<span class="pill ok">0</span>`}</td>
          <td class="num">${pct(h.faithfulness)}</td>
          <td class="num">${pct(h.answer_relevancy)}</td>
          <td class="num">${pct(h.context_recall)}</td>
          <td>${pill(r.mode)}</td>
          <td class="mono">${esc((r.created_at || "").replace("T", " ").slice(5, 16))}</td>
        </tr>`;
      }).join("")}
      </tbody></table>`;
  }

  async function renderRuns() {
    const data = await getJSON("/api/runs?limit=100");
    view().innerHTML = `
      <div class="page-head">
        <div><h1>运行记录</h1><div class="sub">每次评测都绑定配置指纹与实际后端，可追溯、可复现</div></div>
        <button class="btn primary" id="btn-new-run">发起评测</button>
      </div>
      <div class="panel">${runsTable(data.runs)}</div>`;
    $("#btn-new-run").addEventListener("click", openNewRunModal);
  }

  // ---------------------------------------------------------------- 运行详情
  async function renderRun(runId) {
    const [detail, allRuns] = await Promise.all([
      getJSON(`/api/runs/${encodeURIComponent(runId)}`),
      getJSON("/api/runs?limit=100"),
    ]);
    const run = detail.run;
    const scorecard = detail.scorecard || [];
    const groups = detail.metrics || {};
    const splits = detail.splits || {};

    const metricRows = GROUP_ORDER.flatMap((group) => {
      const items = groups[group] || [];
      if (!items.length) return [];
      const head = `<tr><td colspan="6" class="metric-group-title">${esc(GROUP_LABELS[group] || group)}</td></tr>`;
      return [head, ...items.map((m) => `<tr>
        <td>${esc(m.name)}<div class="small muted">${esc(label(m.name) === m.name ? "" : label(m.name))}</div></td>
        <td class="small muted">${esc(m.description || "")}</td>
        <td class="num">${fmtMetric(m)}${barCell(m.value, m.ci_low, m.ci_high)}</td>
        <td class="num small muted">${m.name.startsWith("latency") ? `${num(m.ci_low, 1)}–${num(m.ci_high, 1)}` : `[${pct(m.ci_low)}, ${pct(m.ci_high)}]`}</td>
        <td class="num">${m.n ?? ""}</td>
        <td class="small muted">${m.direction === "lower" ? "越低越好" : "越高越好"}</td>
      </tr>`)];
    }).join("");

    const providers = Object.entries(run.providers || {}).map(([k, v]) => `<span class="tag">${esc(k)} <b>${esc(v)}</b></span>`).join(" ");

    view().innerHTML = `
      <div class="page-head">
        <div>
          <h1>${esc(run.name)}</h1>
          <div class="sub mono">${esc(run.run_id)} · ${esc(run.dataset)} · ${run.n_cases} 例 · 耗时 ${num(run.duration_s, 2)}s</div>
        </div>
        <div class="row">
          <a class="btn" href="#/compare?left=${encodeURIComponent(run.run_id)}">加入对比</a>
          <a class="btn" href="/api/report?run_id=${encodeURIComponent(run.run_id)}&format=html">导出 HTML</a>
          <a class="btn" href="/api/report?run_id=${encodeURIComponent(run.run_id)}&format=markdown">Markdown</a>
          <a class="btn" href="/api/report?run_id=${encodeURIComponent(run.run_id)}&format=xlsx">XLSX</a>
        </div>
      </div>

      <div class="row" style="margin-bottom:16px">
        ${pill(`模式 ${run.mode}`, run.mode === "offline" ? "warn" : "ok")}
        ${pill(`变体 ${run.variant || "baseline"}`)}
        ${pill(`配置 ${String(run.config_fingerprint || "").slice(0, 12)}`)}
        ${pill(`坏例 ${detail.badcase_total}`, detail.badcase_total ? "bad" : "ok")}
        ${providers}
      </div>

      <div class="cards">
        ${scorecard.map((item) => `<div class="card">
          <div class="k">${esc(item.label)}</div>
          <div class="v">${esc(item.display)}</div>
          <div class="bar"><i style="width:${(item.value * 100).toFixed(1)}%"></i></div>
        </div>`).join("")}
      </div>

      <div class="grid-2">
        <div class="panel">
          <h2>分集合表现</h2>
          <table><thead><tr><th>集合</th><th class="num">用例</th><th class="num">坏例</th><th class="num">坏例率</th></tr></thead>
          <tbody>${Object.entries(splits).map(([split, item]) => `<tr>
            <td>${esc(SPLIT_LABELS[split] || split)}</td><td class="num">${item.n}</td>
            <td class="num">${item.bad}</td><td class="num">${item.n ? ((item.bad / item.n) * 100).toFixed(1) : "0.0"}%</td>
          </tr>`).join("")}</tbody></table>
        </div>
        <div class="panel">
          <h2>Judge 与外部框架<span class="hint">交叉校验与复算来源</span></h2>
          <div class="small muted">Judge 来源：${esc((run.metadata || {}).judge ? run.metadata.judge.source : "—")}</div>
          <div class="small muted" style="margin-top:6px">外部适配器：${((run.metadata || {}).adapters || []).length ? ((run.metadata.adapters || []).map((a) => esc(`${a.name}:${a.available ? "启用" : "未启用"}`)).join("，")) : "未启用（缺少依赖或无 API Key）"}</div>
          <div class="small muted" style="margin-top:6px">检索：${esc((run.providers || {}).reranker || "")}</div>
        </div>
      </div>

      <div class="panel">
        <h2>全部指标<span class="hint">数值与 95% Bootstrap 置信区间</span></h2>
        <table>
          <colgroup>
            <col style="width:23%" /><col style="width:39%" /><col style="width:12%" />
            <col style="width:13%" /><col style="width:5%" /><col style="width:8%" />
          </colgroup>
          <thead><tr><th>指标</th><th>口径</th><th class="num">数值</th><th class="num">95% CI</th><th class="num">n</th><th>方向</th></tr></thead>
        <tbody>${metricRows}</tbody></table>
      </div>

      <div class="panel">
        <h2>坏例分析<span class="hint">共 ${detail.badcase_total} 条，点击展开答案与检索片段对比</span></h2>
        <div class="row" style="margin-bottom:12px">
          <select id="filter-split">
            <option value="">全部集合</option>
            ${Object.keys(SPLIT_LABELS).map((s) => `<option value="${s}">${SPLIT_LABELS[s]}</option>`).join("")}
          </select>
          <input type="text" id="filter-kw" placeholder="按问题/用例 ID 搜索" style="min-width:220px" />
          <span class="spacer"></span>
          <span class="small muted" id="filter-count"></span>
        </div>
        <div id="case-list"></div>
      </div>
    `;

    const cases = detail.badcases || [];
    const renderList = () => {
      const split = $("#filter-split").value;
      const kw = ($("#filter-kw").value || "").trim().toLowerCase();
      const filtered = cases.filter((c) =>
        (!split || c.split === split) &&
        (!kw || c.case_id.toLowerCase().includes(kw) || (c.question || "").toLowerCase().includes(kw)));
      $("#filter-count").textContent = `显示 ${filtered.length} / ${cases.length} 条`;
      $("#case-list").innerHTML = filtered.length ? filtered.map(caseBlock).join("") : empty("没有匹配的坏例");
    };
    $("#filter-split").addEventListener("change", renderList);
    $("#filter-kw").addEventListener("input", renderList);
    renderList();
  }

  function caseBlock(c) {
    const scores = Object.entries(c.scores || {})
      .filter(([k]) => ["faithfulness", "answer_relevancy", "answer_completeness", "citation_accuracy", "context_recall"].includes(k))
      .map(([k, v]) => `<span class="tag">${esc(label(k))} <b>${num(v)}</b></span>`).join(" ");
    const flags = (c.flags || []).map((f) => pill(label(f), "bad")).join(" ");
    const judge = ((c.details || {}).judge || {});
    const reason = ((judge.reasons || {}).overall) || "";
    const supports = ((c.details || {}).support || []).map((s) =>
      `<div class="sup"><span class="mark ${s.supported ? "ok" : "no"}">${s.supported ? "支撑" : "未支撑"}</span>
       <span>${esc(s.sentence)}</span><span class="muted">support=${num(s.support)}</span></div>`).join("");
    const relevance = ((c.details || {}).relevance || []);
    const contexts = (c.contexts || []).map((item) => {
      const rel = relevance.find((r) => r.chunk_id === item.chunk_id);
      return `<div class="ctx ${rel && rel.relevant ? "hit" : ""}">
        <div class="h">[${item.rank}] ${esc(item.title)} · ${esc(item.chunk_id)} · score ${num(item.score)} · ${rel && rel.relevant ? "相关" : "不相关"}</div>
        ${esc(item.text || item.preview || "")}</div>`;
    }).join("");
    const reference = (c.case || {}).reference_answer || "";
    return `<details class="badcase">
      <summary>
        <span class="mono">${esc(c.case_id)}</span>
        ${pill(SPLIT_LABELS[c.split] || c.split)}
        <span class="q">${esc(c.question)}</span>
        ${flags}
      </summary>
      <div class="body">
        <div class="grid-2">
          <div>
            <span class="lbl">系统答案</span>
            <pre class="answer">${esc(c.answer || "")}</pre>
            ${scores ? `<div class="row" style="margin-top:10px">${scores}</div>` : ""}
            ${reason ? `<div class="small muted" style="margin-top:8px">Judge：${esc(reason)}</div>` : ""}
          </div>
          <div>
            <span class="lbl">参考答案</span>
            <pre>${esc(reference || "—")}</pre>
            <span class="lbl" style="margin-top:10px">逐句支撑度</span>
            ${supports || `<div class="small muted">拒答或无事实性断言</div>`}
          </div>
        </div>
        <span class="lbl" style="margin-top:12px">检索片段对比（${(c.contexts || []).length} 条）</span>
        ${contexts || `<div class="small muted">无检索结果</div>`}
        <div class="small muted" style="margin-top:8px">延迟 ${num((c.latency || {}).total_ms, 1)} ms · 检索 ${num((c.latency || {}).retrieval_ms, 1)} ms · 重排 ${num((c.latency || {}).rerank_ms, 1)} ms · 生成 ${num((c.latency || {}).generation_ms, 1)} ms</div>
      </div>
    </details>`;
  }

  // ---------------------------------------------------------------- 对比
  async function renderCompare() {
    const runs = (await getJSON("/api/runs?limit=60")).runs;
    const params = new URLSearchParams((location.hash.split("?")[1] || ""));
    const left = params.get("left") || (runs[0] && runs[0].run_id) || "";
    const right = params.get("right") || (runs[1] && runs[1].run_id) || "";

    const options = (selected) => runs.map((r) =>
      `<option value="${esc(r.run_id)}" ${r.run_id === selected ? "selected" : ""}>${esc(r.name)} · ${esc(r.variant || "")} · ${esc(r.created_at.slice(5, 16))}</option>`).join("");

    view().innerHTML = `
      <div class="page-head">
        <div><h1>A/B 对比</h1><div class="sub">同一批用例上的配对比较：配对 Bootstrap 显著性检验 + 效应量</div></div>
      </div>
      <div class="panel">
        <div class="row">
          <select id="left-select" style="min-width:320px"><option value="">选择基线…</option>${options(left)}</select>
          <span class="muted">vs</span>
          <select id="right-select" style="min-width:320px"><option value="">选择实验组…</option>${options(right)}</select>
          <button class="btn primary" id="btn-compare">对比</button>
        </div>
      </div>
      <div id="compare-body">${empty("选择两次运行后点击对比")}</div>
    `;

    const doCompare = async () => {
      const l = $("#left-select").value;
      const r = $("#right-select").value;
      if (!l || !r) { toast("请选择两次运行", true); return; }
      if (l === r) { toast("需要选择不同的运行", true); return; }
      history.replaceState(null, "", `#/compare?left=${encodeURIComponent(l)}&right=${encodeURIComponent(r)}`);
      $("#compare-body").innerHTML = spinner("计算配对显著性…");
      try {
        const data = await getJSON(`/api/compare?left=${encodeURIComponent(l)}&right=${encodeURIComponent(r)}`);
        $("#compare-body").innerHTML = compareBody(data);
      } catch (exc) {
        $("#compare-body").innerHTML = empty(`对比失败：${exc.message}`);
      }
    };
    $("#btn-compare").addEventListener("click", doCompare);
    if (left && right && left !== right) doCompare();
  }

  function compareBody(data) {
    const c = data.composite || {};
    const sig = (d) => d.significant == null
      ? `<span class="pill">仅记录</span>`
      : (d.significant ? `<span class="pill ${d.better === "right" ? "ok" : "bad"}">显著</span>` : `<span class="pill">不显著</span>`);
    const deltas = (data.deltas || []).filter((d) => d.direction !== "lower" || d.metric.includes("latency") || d.metric.includes("rate") || d.metric.includes("redundancy") || d.metric.includes("repetition"));
    const rows = (data.deltas || []).map((d) => {
      const cls = d.better === "right" ? "delta-up" : d.better === "left" ? "delta-down" : "";
      const arrow = d.better === "right" ? "▲" : d.better === "left" ? "▼" : "＝";
      return `<tr>
        <td>${esc(d.metric)}<div class="small muted">${esc(label(d.metric) === d.metric ? "" : label(d.metric))}</div></td>
        <td class="num">${pct(d.left)}</td>
        <td class="num">${pct(d.right)}</td>
        <td class="num ${cls}">${arrow} ${(d.delta >= 0 ? "+" : "")}${(d.delta * 100).toFixed(1)}%</td>
        <td class="num small muted">${d.p_value == null ? "—" : Number(d.p_value).toFixed(4)}</td>
        <td class="num small muted">${d.effect_size == null ? "—" : Number(d.effect_size).toFixed(2)}</td>
        <td>${sig(d)}</td>
      </tr>`;
    }).join("");

    const improvements = (data.improved || []).map((d) => pill(`${label(d.metric)} ↑`, "ok"));
    const regressions = (data.degraded || []).map((d) => pill(`${label(d.metric)} ↓`, "bad"));

    // 卡片里最多展示 5 个，其余折叠为计数，避免长列表撑破卡片
    const pillsBrief = (items, total) => {
      if (!items.length) return "无";
      const shown = items.slice(0, 5).join(" ");
      return items.length > 5 ? `${shown} <span class="muted small">等 ${total} 项</span>` : shown;
    };
    const cases = (data.case_diffs || []).slice(0, 12).map((c) => `<tr>
      <td class="mono">${esc(c.case_id)}</td>
      <td>${esc(c.question || "")}</td>
      <td class="num">${num(c.left)}</td>
      <td class="num">${num(c.right)}</td>
      <td class="num ${c.delta > 0 ? "delta-up" : c.delta < 0 ? "delta-down" : ""}">${(c.delta >= 0 ? "+" : "")}${num(c.delta)}</td>
    </tr>`).join("");

    return `
      <div class="cards">
        <div class="card"><div class="k">基线综合分</div><div class="v">${num(c.left, 3)}</div><div class="d">${esc((data.left_run || {}).name || "")}</div></div>
        <div class="card"><div class="k">实验组综合分</div><div class="v">${num(c.right, 3)}</div><div class="d">${esc((data.right_run || {}).name || "")}</div></div>
        <div class="card"><div class="k">显著改善指标</div><div class="v">${(data.improved || []).length}</div><div class="d">${pillsBrief(improvements, (data.improved || []).length)}</div></div>
        <div class="card"><div class="k">显著退化指标</div><div class="v">${(data.degraded || []).length}</div><div class="d">${pillsBrief(regressions, (data.degraded || []).length)}</div></div>
        <div class="card"><div class="k">配对用例</div><div class="v">${data.n_cases}</div><div class="d">${esc(data.mode || "")}</div></div>
      </div>
      <div class="panel">
        <h2>指标差异<span class="hint">右列为实验组；p 值来自配对 Bootstrap（零假设中心化）</span></h2>
        <table><thead><tr><th>指标</th><th class="num">基线</th><th class="num">实验组</th><th class="num">Δ</th><th class="num">p 值</th><th class="num">Cohen's d</th><th>结论</th></tr></thead>
        <tbody>${rows || `<tr><td colspan="7">${empty("无共同指标")}</td></tr>`}</tbody></table>
      </div>
      <div class="panel">
        <h2>逐用例变化<span class="hint">按综合质量分排序，便于抽查「变好的」和「变差的」样本</span></h2>
        <table><thead><tr><th>用例</th><th>问题</th><th class="num">基线</th><th class="num">实验组</th><th class="num">Δ</th></tr></thead>
        <tbody>${cases || `<tr><td colspan="5">${empty("无逐用例数据")}</td></tr>`}</tbody></table>
      </div>`;
  }

  // ---------------------------------------------------------------- 实验
  async function renderExperiments() {
    const data = await getJSON("/api/experiments");
    const items = data.experiments || [];
    view().innerHTML = `
      <div class="page-head">
        <div><h1>A/B 实验</h1><div class="sub">切片策略 / TopK / 重排序 / Prompt / 模型的对照实验</div></div>
        <div class="row"><span class="small muted">通过 CLI 运行：python -m rag_eval ab --experiment configs/experiments/ab_chunking.toml</span></div>
      </div>
      ${items.length ? `<div class="panel"><table>
        <thead><tr><th>实验</th><th>数据集</th><th class="num">变体</th><th>冠军</th><th class="num">显著改善</th><th class="num">显著退化</th><th>状态</th><th>时间</th></tr></thead>
        <tbody>${items.map((e) => {
          const s = e.summary || {};
          return `<tr>
            <td><a href="#/experiment/${esc(e.experiment_id)}">${esc(e.name)}</a><div class="mono">${esc(e.experiment_id)}</div></td>
            <td class="mono">${esc((e.dataset || "").split("/").pop())}</td>
            <td class="num">${(e.variants || []).length}</td>
            <td>${s.winner ? pill(s.winner, "ok") : "—"}</td>
            <td class="num">${s.significant_improvements ?? "—"}</td>
            <td class="num">${s.significant_regressions ?? "—"}</td>
            <td>${pill(e.status)}</td>
            <td class="mono">${esc((e.created_at || "").replace("T", " ").slice(5, 16))}</td>
          </tr>`;
        }).join("")}</tbody></table></div>` : `<div class="panel">${empty("尚无实验记录")}</div>`}
    `;
  }

  async function renderExperiment(experimentId) {
    const data = await getJSON(`/api/experiments/${encodeURIComponent(experimentId)}`);
    const summary = data.summary || {};
    const rows = summary.ranking || [];
    const runs = data.runs || [];
    const runOf = (variant) => runs.find((r) => r.variant === variant);

    view().innerHTML = `
      <div class="page-head">
        <div><h1>${esc(data.name)}</h1><div class="sub mono">${esc(data.experiment_id)} · ${esc(data.dataset)}</div></div>
        <a class="btn" href="#/experiments">返回列表</a>
      </div>
      <div class="panel">
        <h2>排名<span class="hint">综合分 = 关键质量指标的方向归一化均值</span></h2>
        <table><thead><tr><th>变体</th><th>说明</th><th class="num">综合分</th><th class="num">用例</th><th class="num">耗时</th><th>关键指标</th><th></th></tr></thead>
        <tbody>${rows.map((row, idx) => {
          const r = runOf(row.variant);
          const h = row.headline || {};
          return `<tr>
            <td>${idx === 0 ? "🏆 " : ""}${esc(row.variant)}${row.variant === summary.baseline ? `<span class="pill info">基线</span>` : ""}</td>
            <td class="small muted">${esc(row.description || "")}</td>
            <td class="num">${num(row.composite, 3)}${barCell(row.composite, row.composite, row.composite)}</td>
            <td class="num">${row.n_cases ?? ""}</td>
            <td class="num">${num(row.duration_s, 2)}s</td>
            <td class="small muted">忠实 ${pct(h.faithfulness)} · 相关 ${pct(h.answer_relevancy)} · 召回 ${pct(h.context_recall)} · 引用 ${pct(h.citation_accuracy)}</td>
            <td>${r ? `<a class="btn ghost" href="#/run/${esc(r.run_id)}">详情</a>` : ""}</td>
          </tr>`;
        }).join("")}</tbody></table>
      </div>
      ${(summary.rows || []).filter((r) => Object.keys(r.overrides || {}).length).length ? `<div class="panel">
        <h2>配置差异<span class="hint">每个变体相对基线的覆盖项</span></h2>
        ${(summary.rows || []).filter((r) => Object.keys(r.overrides || {}).length).map((r) => `
          <div class="small" style="margin-bottom:8px"><b>${esc(r.variant)}</b>：${Object.entries(r.overrides).map(([sec, kv]) =>
            `${esc(sec)}.${Object.entries(kv).map(([k, v]) => `${esc(k)}=${esc(v)}`).join(", ")}`).join("；")}</div>`).join("")}
      </div>` : ""}
      <div class="panel">
        <h2>与基线的显著性对比</h2>
        ${(summary.rows || []).filter((r) => r.variant !== summary.baseline && runOf(r.variant)).map((r) => `
          <div class="small" style="margin:10px 0 4px"><b>${esc(r.variant)}</b> vs ${esc(summary.baseline)}</div>
          <div id="cmp-${esc(r.variant)}" class="small muted">加载中…</div>`).join("") || empty("无对比数据")}
      </div>
    `;

    for (const row of rows) {
      if (row.variant === summary.baseline) continue;
      const base = runOf(summary.baseline);
      const target = runOf(row.variant);
      const holder = document.getElementById(`cmp-${row.variant}`);
      if (!base || !target || !holder) continue;
      try {
        const cmp = await getJSON(`/api/compare?left=${encodeURIComponent(base.run_id)}&right=${encodeURIComponent(target.run_id)}`);
        const sig = (cmp.deltas || []).filter((d) => d.significant);
        holder.innerHTML = sig.length
          ? sig.map((d) => `${pill(`${label(d.metric)} ${d.delta > 0 ? "+" : ""}${(d.delta * 100).toFixed(1)}%`, d.better === "right" ? "ok" : "bad")}`).join(" ")
          : "无显著差异（样本量或效应量不足）";
      } catch (exc) {
        holder.textContent = `对比失败：${exc.message}`;
      }
    }
  }

  // ---------------------------------------------------------------- 用例集
  async function renderDatasets() {
    const [data, metrics] = await Promise.all([getJSON("/api/datasets"), getJSON("/api/metrics")]);
    const groups = {};
    (metrics.metrics || []).forEach((m) => { (groups[m.group] = groups[m.group] || []).push(m); });
    view().innerHTML = `
      <div class="page-head">
        <div><h1>用例集与指标字典</h1><div class="sub">核心回归集 / 边缘场景集 / 对抗集，以及全部可用指标的口径</div></div>
      </div>
      <div class="panel">
        <h2>测试集</h2>
        <table><thead><tr><th>文件</th><th>名称</th><th class="num">用例数</th><th>拆分</th></tr></thead>
        <tbody>${(data.datasets || []).map((d) => `<tr>
          <td class="mono">${esc(d.path)}</td><td>${esc(d.name)}</td><td class="num">${d.count ?? "—"}</td>
          <td>${Object.entries(d.splits || {}).map(([k, v]) => pill(`${k} ${v}`)).join(" ")}</td>
        </tr>`).join("")}</tbody></table>
      </div>
      <div class="panel">
        <h2>指标字典<span class="hint">${(metrics.metrics || []).length} 个指标</span></h2>
        ${GROUP_ORDER.filter((g) => groups[g]).map((g) => `
          <div class="metric-group-title">${esc(GROUP_LABELS[g] || g)}</div>
          <table><thead><tr><th>指标</th><th>类型</th><th>方向</th><th>口径</th></tr></thead>
          <tbody>${groups[g].map((m) => `<tr>
            <td>${esc(m.name)}<div class="small muted">${esc(label(m.name) === m.name ? "" : label(m.name))}</div></td>
            <td class="small muted">${m.scope === "aggregate" ? "聚合" : "逐用例"}</td>
            <td class="small muted">${m.direction === "lower" ? "越低越好" : "越高越好"}</td>
            <td class="small muted">${esc(m.description)}</td>
          </tr>`).join("")}</tbody></table>`).join("")}
      </div>
    `;
  }

  // ---------------------------------------------------------------- 发起评测
  async function openNewRunModal() {
    let datasets = { datasets: [] };
    try { datasets = await getJSON("/api/datasets"); } catch (_) {}
    const modal = document.createElement("div");
    modal.className = "modal show";
    modal.innerHTML = `<div class="box">
      <h3>发起一次评测</h3>
      <div class="field"><label>测试集</label>
        <select id="m-dataset">${(datasets.datasets || []).map((d) =>
          `<option value="${esc(d.path)}">${esc(d.path)}（${d.count ?? "?"} 例）</option>`).join("")}</select></div>
      <div class="field"><label>运行名称</label><input id="m-name" type="text" value="dashboard-${new Date().toTimeString().slice(0, 5).replace(":", "")}" /></div>
      <div class="field"><label>集合过滤（留空为全部）</label>
        <select id="m-splits" multiple size="3" style="min-height:70px">
          <option value="core">核心回归集</option><option value="edge">边缘场景集</option><option value="adversarial">对抗集</option>
        </select></div>
      <div class="field"><label>最多用例数（0 = 不限）</label><input id="m-max" type="number" value="0" min="0" /></div>
      <div class="field"><label><input id="m-judge" type="checkbox" checked /> 启用 LLM-as-Judge（无 Key 时用启发式量规）</label></div>
      <div class="row"><span class="spacer"></span>
        <button class="btn" id="m-cancel">取消</button>
        <button class="btn primary" id="m-start">开始评测</button></div>
    </div>`;
    document.body.appendChild(modal);
    const close = () => modal.remove();
    $("#m-cancel", modal).addEventListener("click", close);
    modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
    $("#m-start", modal).addEventListener("click", async () => {
      const splits = Array.from($("#m-splits", modal).selectedOptions).map((o) => o.value);
      const payload = {
        dataset: $("#m-dataset", modal).value,
        name: $("#m-name", modal).value || "dashboard-run",
        splits: splits.length ? splits : null,
        max_cases: Number($("#m-max", modal).value || 0),
        use_judge: $("#m-judge", modal).checked,
      };
      $("#m-start", modal).disabled = true;
      $("#m-start", modal).textContent = "运行中…";
      try {
        const job = await getJSON("/api/eval", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        close();
        pollJob(job.job_id);
      } catch (exc) {
        toast(`启动失败：${exc.message}`, true);
        $("#m-start", modal).disabled = false;
        $("#m-start", modal).textContent = "开始评测";
      }
    });
  }

  async function pollJob(jobId) {
    toast(`已提交评测任务 ${jobId}`);
    const started = Date.now();
    const tick = async () => {
      try {
        const job = await getJSON(`/api/jobs/${jobId}`);
        if (job.status === "done") {
          toast(`评测完成：${job.message}`);
          location.hash = `#/run/${job.result.run_id}`;
          return;
        }
        if (job.status === "failed") {
          toast(`评测失败：${job.error}`, true);
          return;
        }
        if (Date.now() - started > 1000 * 60 * 30) { toast("任务超时", true); return; }
        setTimeout(tick, 900);
      } catch (exc) {
        toast(`轮询失败：${exc.message}`, true);
      }
    };
    tick();
  }

  window.addEventListener("hashchange", router);
  document.addEventListener("DOMContentLoaded", router);
  if (document.readyState !== "loading") router();

  // 兜底错误提示：任何未捕获异常都在页面上暴露出来，而不是留一个空白页
  window.addEventListener("error", (event) => {
    const box = view();
    if (box && !box.querySelector(".panel, .cards, .badcase")) {
      box.innerHTML = `<div class="empty">页面渲染出错：${esc((event.error && event.error.message) || event.message)}</div>`;
    }
    console.error(event.error || event.message);
  });
  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason && (event.reason.stack || event.reason.message || event.reason);
    console.error(reason);
  });
})();
