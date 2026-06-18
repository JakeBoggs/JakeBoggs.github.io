(function () {
  if (!document.querySelector("[data-benchmark-dashboard]")) return;

  const state = {
    data: null,
    rankingView: "bar",
    frontierDeltaMode: "time",
    filters: { benchmark: "", category: "", lab: "", openness: "" },
    charts: { ranking: null, time: null, metr: null, eci: null, frontier: null, frontierDelta: null },
  };

  const fmt = new Intl.NumberFormat("en-US");
  const dateFull = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
  const $ = (sel) => document.querySelector(sel);
  const pct = (v) => v == null ? "" : `${(v * 100).toFixed(1)}%`;

  // Brand-ish colors per lab. Falls back to a deterministic hash color.
  const LAB_COLORS = {
    "OpenAI": "#10A37F",
    "Anthropic": "#D97757",
    "Google": "#4285F4",
    "Google DeepMind": "#4285F4",
    "xAI": "#1F1F1F",
    "DeepSeek": "#4D6BFE",
    "Meta": "#0467DF",
    "Meta AI": "#0467DF",
    "Mistral": "#FA520F",
    "Mistral AI": "#FA520F",
    "Alibaba": "#615CED",
    "Qwen": "#615CED",
    "Z.ai": "#06B6D4",
    "Zhipu": "#06B6D4",
    "GLM": "#06B6D4",
    "Moonshot AI": "#7E22CE",
    "Moonshot": "#7E22CE",
    "Kimi": "#7E22CE",
    "Cohere": "#E879F9",
    "NVIDIA": "#76B900",
    "OpenBMB": "#F59E0B",
    "Microsoft": "#5E5E5E",
    "Tencent": "#1F8AE0",
    "Reka": "#FF4081",
    "Inflection": "#9333EA",
    "AI21": "#EF4444",
    "Human": "#16A34A",
  };
  const US_LABS = new Set(["OpenAI", "Anthropic", "Google", "Google DeepMind", "xAI", "Meta", "Meta AI", "Microsoft", "Amazon", "Cohere", "AI21 Labs", "NVIDIA"]);
  const CHINA_LABS = new Set(["DeepSeek", "Alibaba", "Qwen", "Z.ai", "Zhipu", "Zhipu AI", "GLM", "Moonshot AI", "Moonshot", "Kimi", "MiniMax", "Xiaomi", "ByteDance", "Tencent", "01.AI", "StepFun", "OpenBMB"]);
  const METR_ESTIMATE_MIN_INDEX = 150;
  const METR_ESTIMATE_LIMIT = 12;
  const ECI_PROJECTION_MIN_INDEX = 150;
  const ECI_PROJECTION_LIMIT = 12;
  const BENCHMARK_TRANSFORMS = {
    "bullshitbench": "Score = green classifications / scored samples.",
    "benchmarks-bio-epibench": "Score = pass rate.",
    "benchmarks-bio-singlecell": "Score = accuracy.",
    "benchmarks-bio-txbench": "Score = pass rate.",
    "eqbench": "Score = 2 * P(win vs top Elo).",
    "gdpval-aa": "Score = 2 * P(win vs top Elo).",
    "halluhard": "Score = 1 - mean hallucination rate.",
    "mercor-apex": "Score = best allowed-harness pass@1.",
    "omniscience": "Score = accuracy.",
    "opus-magnum-bench": "Score = mean best human-normalized reward across 36 puzzles.",
    "posttrainbench": "Score = published aggregate; subtasks hidden.",
    "ppbench": "Score = max available success rate.",
    "programbench": "Score = raw pass rate.",
    "vending-bench-2": "Score = profit / $62.5k target profit, capped at 1.",
    "contextarena": "Score = 8-needle MRCRv2 AUC @1M.",
    "lisanbench": "Score = mean path length / current leaderboard maximum.",
    "game-arena": "Score = 2 * P(win vs top Bradley-Terry rating).",
    "zerobench": "Score = main-question pass@5.",
  };

  function labColor(lab) {
    if (LAB_COLORS[lab]) return LAB_COLORS[lab];
    let hash = 0;
    for (const ch of String(lab || "")) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
    const hue = hash % 360;
    return `hsl(${hue}, 55%, 48%)`;
  }
  const withAlpha = (hex, alpha) => {
    if (hex.startsWith("hsl")) return hex.replace(")", ` / ${alpha})`).replace("hsl(", "hsla(");
    const m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (!m) return hex;
    const n = parseInt(m[1], 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  };

  const uniqueSorted = (vs) => Array.from(new Set(vs.filter(Boolean))).sort((a, b) => a.localeCompare(b));

  const optionHtml = (label, values) => [`<option value="">${label}</option>`].concat(
    values.map((value) => `<option value="${value}">${value}</option>`)
  ).join("");

  function setStatus(text) {
    const el = $("[data-dashboard-status]");
    if (!el) return;
    el.textContent = text;
    el.hidden = !text;
  }

  function setUpdatedDate(generatedAt) {
    const generated = new Date(generatedAt);
    const text = Number.isNaN(generated.getTime()) ? "" : `Updated ${dateFull.format(generated)}`;
    document.querySelectorAll("[data-chart-updated]").forEach((el) => {
      el.textContent = text;
    });
  }

  function setEmpty(key, on) {
    const empty = document.querySelector(`[data-empty='${key}']`);
    const canvas = document.querySelector(`canvas[data-chart='${key}']`);
    if (empty) empty.hidden = !on;
    if (canvas) canvas.style.display = on ? "none" : "";
    if (on && state.charts[key]) {
      state.charts[key].destroy();
      state.charts[key] = null;
    }
  }

  function isMobileChart() {
    return window.innerWidth <= 700;
  }

  function chartPadding() {
    if (window.innerWidth <= 560) return { left: 0, right: 0, top: 6, bottom: 0 };
    if (window.innerWidth <= 900) return { left: 6, right: 6, top: 10, bottom: 2 };
    return { left: 60, right: 60, top: 16, bottom: 4 };
  }

  function axisTitle(text) {
    return { display: !isMobileChart(), text };
  }

  function yAxisTicks(options = {}) {
    return {
      maxTicksLimit: isMobileChart() ? 5 : 8,
      padding: isMobileChart() ? 1 : 3,
      font: { size: isMobileChart() ? 10 : 11 },
      ...options,
    };
  }

  function compactRange(values, fraction = 0.025, always = false) {
    if (!always && !isMobileChart()) return {};
    const finite = values.filter((value) => Number.isFinite(value));
    if (!finite.length) return {};
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    const span = max - min || Math.max(Math.abs(max), 1);
    const pad = span * fraction;
    return { min: min - pad, max: max + pad };
  }

  function rankingTickOptions() {
    if (window.innerWidth <= 560) {
      return { autoSkip: true, maxTicksLimit: 12, maxRotation: 65, minRotation: 45, font: { size: 10 } };
    }
    return { autoSkip: false, maxRotation: 70, minRotation: 60, font: { size: 11 } };
  }

  function initControls() {
    const data = state.data;
    const categoryOpts = uniqueSorted(data.categories || data.results.flatMap((r) => rowCategories(r)));
    const labOpts = uniqueSorted(data.results.map((r) => r.lab));

    const benchmarkFilter = $("[data-filter='benchmark']");
    const categoryFilter = $("[data-filter='category']");
    const labFilter = $("[data-filter='lab']");
    updateBenchmarkOptions();
    if (categoryFilter) categoryFilter.innerHTML = optionHtml("All categories", categoryOpts);
    if (labFilter) labFilter.innerHTML = optionHtml("All labs", labOpts);

    document.querySelectorAll("[data-filter]").forEach((el) => {
      el.addEventListener("input", () => {
        const key = el.dataset.filter;
        state.filters[key] = el.value;
        if (key === "category") {
          state.filters.benchmark = "";
          updateBenchmarkOptions();
        }
        render();
      });
    });

    document.querySelectorAll("[data-ranking-view]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.rankingView = btn.dataset.rankingView;
        document.querySelectorAll("[data-ranking-view]").forEach((b) => {
          b.classList.toggle("is-active", b.dataset.rankingView === state.rankingView);
        });
        applyRankingView();
      });
    });

    document.querySelectorAll("[data-frontier-delta-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.frontierDeltaMode = btn.dataset.frontierDeltaMode;
        document.querySelectorAll("[data-frontier-delta-mode]").forEach((b) => {
          b.classList.toggle("is-active", b.dataset.frontierDeltaMode === state.frontierDeltaMode);
        });
        renderFrontierDelta();
      });
    });

    applyRankingView();
    initStickyControls();
  }

  function initStickyControls() {
    const shell = document.querySelector("[data-dashboard-controls-shell]");
    const controls = shell?.querySelector(".dashboard-controls");
    const stopPanel = document.querySelector("[data-frontier-delta-panel]");
    if (!shell || !controls || !stopPanel) return;

    const update = () => {
      shell.classList.remove("is-sticky", "is-dismissed");
      const shellRect = shell.getBoundingClientRect();
      const stopTop = window.scrollY + stopPanel.getBoundingClientRect().top;
      const shellTop = window.scrollY + shellRect.top;
      const controlsHeight = controls.offsetHeight;

      shell.style.setProperty("--sticky-controls-left", `${shellRect.left}px`);
      shell.style.setProperty("--sticky-controls-width", `${shellRect.width}px`);
      shell.style.setProperty("--sticky-controls-height", `${controlsHeight}px`);

      if (window.scrollY >= stopTop) {
        shell.classList.add("is-dismissed");
      } else if (window.scrollY > shellTop) {
        shell.classList.add("is-sticky");
      }
    };

    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
  }

  function updateBenchmarkOptions() {
    const benchmarkFilter = $("[data-filter='benchmark']");
    if (!benchmarkFilter || !state.data) return;
    const category = state.filters.category;
    const benchmarks = uniqueSorted(
      state.data.results
        .filter((row) => !category || rowCategories(row).includes(category))
        .map((row) => row.benchmark_name)
    );
    benchmarkFilter.innerHTML = optionHtml("All benchmarks", benchmarks);
    benchmarkFilter.value = state.filters.benchmark;
  }

  function applyRankingView() {
    document.querySelectorAll("[data-view-pane]").forEach((pane) => {
      const isActive = pane.dataset.viewPane === state.rankingView;
      const isEmptyBanner = pane.hasAttribute("data-empty");
      // Charts: visible iff their view is active. Empty banners: never shown here; render() decides.
      if (isEmptyBanner) {
        if (!isActive) pane.hidden = true;
      } else {
        pane.hidden = !isActive;
      }
    });
    if (state.rankingView === "bar") renderRanking();
    else renderTime();
  }

  function passes(row) {
    const f = state.filters;
    if (f.benchmark && row.benchmark_name !== f.benchmark) return false;
    if (f.category && !rowCategories(row).includes(f.category)) return false;
    if (f.lab && row.lab !== f.lab) return false;
    if (f.openness === "open" && row.open_weights !== true) return false;
    if (f.openness === "closed" && row.open_weights !== false) return false;
    return true;
  }

  function rowCategories(row) {
    if (Array.isArray(row.categories)) return row.categories.filter(Boolean);
    return row.category ? [row.category] : [];
  }

  function filteredResults() {
    return state.data.results.filter(passes);
  }

  function modelReleaseDateMap() {
    const dates = new Map();
    const setEarliest = (key, value) => {
      const date = parseDate(value);
      if (!key || !date) return;
      const candidate = { value, date, precision: datePrecision(value) };
      const current = dates.get(key);
      if (!current || shouldUseReleaseDate(candidate, current)) dates.set(key, candidate);
    };

    (state.data.models || []).forEach((model) => {
      setEarliest(model.model_id, model.release_date);
    });

    (state.data.results || []).forEach((row) => {
      setEarliest(row.model_id, row.release_date);
      setEarliest(row.capability_model_id, row.release_date);
    });

    return new Map(Array.from(dates, ([key, item]) => [key, item.value]));
  }

  function datePrecision(value) {
    const text = String(value || "").trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return 3;
    if (/^\d{4}-\d{2}$/.test(text) || /^[A-Za-z]{3,9}\s+\d{4}$/.test(text)) return 2;
    if (/^\d{4}$/.test(text)) return 1;
    return 3;
  }

  function shouldUseReleaseDate(candidate, current) {
    if (candidate.date.getTime() < current.date.getTime() && !sameDateBucket(candidate, current)) return true;
    if (sameDateBucket(candidate, current) && candidate.precision > current.precision) return true;
    return false;
  }

  function sameDateBucket(a, b) {
    const precision = Math.min(a.precision, b.precision);
    if (precision <= 1) return a.date.getUTCFullYear() === b.date.getUTCFullYear();
    return a.date.getUTCFullYear() === b.date.getUTCFullYear()
      && a.date.getUTCMonth() === b.date.getUTCMonth();
  }

  function selectedCapabilityRows() {
    const cat = state.filters.category;
    if (cat) return state.data.capability_indices_by_category?.[cat] || [];
    return state.data.capability_index || [];
  }

  function capabilityIndexLabel() {
    return state.filters.category ? `${titleCase(state.filters.category)} Capabilities Index` : "Capabilities Index";
  }

  function jakeCapabilityIndexLabel() {
    return state.filters.category ? `Jake's ${titleCase(state.filters.category)} Capabilities Index` : "Jake's Capabilities Index";
  }

  function titleCase(value) {
    return String(value || "")
      .split(/[-_\s]+/)
      .filter(Boolean)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  function destroyChart(key) {
    if (state.charts[key]) {
      state.charts[key].destroy();
      state.charts[key] = null;
    }
  }

  function rankingEntries(rows) {
    if (state.filters.benchmark) {
      // Single-benchmark mode: best score per model.
      const best = new Map();
      rows.forEach((r) => {
        const k = r.capability_model_id || r.model_id;
        const cur = best.get(k);
        if (!cur || r.normalized_score > cur.normalized_score) best.set(k, r);
      });
      return Array.from(best.values())
        .sort((a, b) => b.normalized_score - a.normalized_score)
        .map((r) => ({
          label: r.capability_model || r.model,
          lab: r.lab,
          value: Number(r.normalized_score) * 100,
          valueText: pct(r.normalized_score),
          detail: `${r.lab} - ${r.benchmark_name}`,
          modelId: r.capability_model_id || r.model_id,
          releaseDate: r.release_date,
        }));
    }
    // Capability-index mode.
    const allowed = new Set(rows.map((r) => r.capability_model_id || r.model_id));
    return selectedCapabilityRows()
      .filter((r) => allowed.has(r.model_id))
      .map((r) => ({
        label: r.model,
        lab: r.lab,
        value: Number(r.index),
        valueText: r.index.toFixed(1),
        detail: `${r.lab} - ${r.benchmark_count} benchmarks`,
        modelId: r.model_id,
        releaseDate: r.release_date,
      }));
  }

  function rankingTitleText() {
    const isBenchmark = !!state.filters.benchmark;
    if (state.rankingView === "time") {
      return isBenchmark ? `${state.filters.benchmark} Over Time` : "Capabilities Index Over Time";
    }
    return isBenchmark ? `${state.filters.benchmark} Leaderboard` : "Capabilities Index";
  }

  function selectedBenchmark() {
    if (!state.filters.benchmark) return "";
    return (state.data.benchmarks || []).find((item) => item.name === state.filters.benchmark) || null;
  }

  function selectedBenchmarkId() {
    return selectedBenchmark()?.id || "";
  }

  function rankingSubtitleText() {
    const benchmarkId = selectedBenchmarkId();
    if (benchmarkId) return BENCHMARK_TRANSFORMS[benchmarkId] || "";
    return "GPT-4.1 = 100, GPT-5 = 150";
  }

  function selectedBenchmarkMetricRows() {
    const best = new Map();
    filteredResults().forEach((row) => {
      const score = Number(row.normalized_score);
      if (!Number.isFinite(score)) return;
      const modelId = row.capability_model_id || row.model_id;
      const current = best.get(modelId);
      if (!current || score > Number(current.normalized_score)) best.set(modelId, row);
    });
    return Array.from(best.values()).map((row) => ({
      model: row.capability_model || row.model,
      model_id: row.capability_model_id || row.model_id,
      lab: row.lab,
      value: Number(row.normalized_score) * 100,
      valueText: pct(row.normalized_score),
    }));
  }

  function selectedCapabilityMetricRows() {
    const allowed = new Set(filteredResults().map((row) => row.capability_model_id || row.model_id));
    return selectedCapabilityRows()
      .filter((row) => allowed.has(row.model_id))
      .map((row) => ({
        model: row.model,
        model_id: row.model_id,
        lab: row.lab,
        value: Number(row.index),
        valueText: row.index.toFixed(1),
      }));
  }

  function metrComparisonLabel() {
    if (state.filters.benchmark) return `${state.filters.benchmark} score`;
    return capabilityIndexLabel();
  }

  function metrCorrelationPairs() {
    const timeHorizons = new Map((state.data.metr_time_horizons || []).map((row) => [row.model_id, row]));
    const metricRows = state.filters.benchmark ? selectedBenchmarkMetricRows() : selectedCapabilityMetricRows();
    return metricRows
      .map((row) => {
        const horizon = timeHorizons.get(row.model_id);
        if (!horizon || !Number.isFinite(Number(horizon.log2_time_horizon)) || !Number.isFinite(Number(horizon.p50_minutes))) return null;
        return {
          model: row.model,
          model_id: row.model_id,
          lab: row.lab,
          xValue: Number(row.value),
          valueText: row.valueText,
          log2TimeHorizon: Number(horizon.log2_time_horizon),
          p50Minutes: Number(horizon.p50_minutes),
          estimated: false,
        };
      })
      .filter((pair) => pair && Number.isFinite(pair.xValue) && pair.p50Minutes > 0);
  }

  function pearsonCorrelation(pairs, yKey = "log2TimeHorizon") {
    const xMean = pairs.reduce((sum, pair) => sum + pair.xValue, 0) / pairs.length;
    const yMean = pairs.reduce((sum, pair) => sum + pair[yKey], 0) / pairs.length;
    const numerator = pairs.reduce((sum, pair) => sum + (pair.xValue - xMean) * (pair[yKey] - yMean), 0);
    const xVariance = pairs.reduce((sum, pair) => sum + (pair.xValue - xMean) ** 2, 0);
    const yVariance = pairs.reduce((sum, pair) => sum + (pair[yKey] - yMean) ** 2, 0);
    const denominator = Math.sqrt(xVariance * yVariance);
    return denominator ? numerator / denominator : 0;
  }

  function logYRegression(points) {
    const logPoints = points
      .filter((point) => point.y > 0)
      .map((point) => ({ x: point.x, y: Math.log2(point.y) }));
    if (logPoints.length < 2) return null;
    const xMean = logPoints.reduce((sum, point) => sum + point.x, 0) / logPoints.length;
    const yMean = logPoints.reduce((sum, point) => sum + point.y, 0) / logPoints.length;
    const denominator = logPoints.reduce((sum, point) => sum + (point.x - xMean) ** 2, 0);
    if (!denominator) return null;
    const slope = logPoints.reduce((sum, point) => sum + (point.x - xMean) * (point.y - yMean), 0) / denominator;
    return { slope, intercept: yMean - slope * xMean };
  }

  function metrEstimatedPairs(officialPairs) {
    if (state.filters.benchmark || officialPairs.length < 3) return [];
    const model = logYRegression(officialPairs.map((pair) => ({ x: pair.xValue, y: pair.p50Minutes })));
    if (!model) return [];
    const officialIds = new Set(officialPairs.map((pair) => pair.model_id));
    return selectedCapabilityMetricRows()
      .filter((row) => !officialIds.has(row.model_id))
      .filter((row) => Number(row.value) >= METR_ESTIMATE_MIN_INDEX)
      .sort((a, b) => b.value - a.value)
      .slice(0, METR_ESTIMATE_LIMIT)
      .map((row) => {
        const log2TimeHorizon = model.intercept + model.slope * row.value;
        const p50Minutes = 2 ** log2TimeHorizon;
        return {
          model: row.model,
          model_id: row.model_id,
          lab: row.lab,
          xValue: row.value,
          valueText: row.valueText,
          log2TimeHorizon,
          p50Minutes,
          estimated: true,
        };
      })
      .filter((pair) => Number.isFinite(pair.p50Minutes) && pair.p50Minutes > 0);
  }

  function renderMetrCorrelation() {
    const pairs = metrCorrelationPairs();
    const estimatedPairs = metrEstimatedPairs(pairs);
    const title = $("[data-metr-title]");
    const meta = $("[data-metr-meta]");
    const sourceLink = $("[data-metr-source-link]");
    if (title) title.textContent = `METR Time Horizon vs ${metrComparisonLabel()}`;
    if (sourceLink) sourceLink.href = state.data.metr_time_horizon_metadata?.source_url || "https://metr.org/time-horizons/";
    if (pairs.length < 3) {
      if (meta) meta.textContent = "";
      setEmpty("metr", true);
      return;
    }
    setEmpty("metr", false);

    const points = pairs.map((pair) => ({
      x: pair.xValue,
      y: pair.p50Minutes,
      label: pair.model,
      lab: pair.lab,
      valueText: pair.valueText,
      p50Minutes: pair.p50Minutes,
      log2TimeHorizon: pair.log2TimeHorizon,
    }));
    const estimatedPoints = estimatedPairs.map((pair) => ({
      x: pair.xValue,
      y: pair.p50Minutes,
      label: pair.model,
      lab: pair.lab,
      valueText: pair.valueText,
      p50Minutes: pair.p50Minutes,
      log2TimeHorizon: pair.log2TimeHorizon,
      estimated: true,
    }));
    const trend = logYTrend(points);
    if (meta) meta.textContent = `Pearson r = ${pearsonCorrelation(pairs).toFixed(2)} (n = ${pairs.length})`;

    const canvas = document.querySelector("canvas[data-chart='metr']");
    if (!canvas) return;
    destroyChart("metr");
    state.charts.metr = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Official METR",
            data: points,
            parsing: false,
            backgroundColor: points.map((point) => withAlpha(labColor(point.lab), 0.85)),
            borderColor: points.map((point) => labColor(point.lab)),
            pointRadius: 5,
            pointHoverRadius: 8,
          },
          {
            label: "Estimated",
            data: estimatedPoints,
            parsing: false,
            backgroundColor: "rgba(255, 255, 255, 0.9)",
            borderColor: estimatedPoints.map((point) => labColor(point.lab)),
            pointStyle: "triangle",
            pointRadius: 6,
            pointHoverRadius: 9,
            pointBorderWidth: 2,
          },
          {
            type: "line",
            label: "Fit",
            data: trend,
            parsing: false,
            borderColor: "#111827",
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: chartPadding() },
        plugins: {
          legend: {
            position: "top",
            labels: { boxWidth: 14, font: { size: 11 } },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                if (ctx.dataset.label === "Fit") return `Fit: ${formatMinutes(ctx.parsed.y)}`;
                const point = ctx.dataset.data[ctx.dataIndex];
                const source = point.estimated ? "estimated time horizon" : "time horizon";
                return `${point.label} (${point.lab}): ${metrComparisonLabel()} ${point.valueText}, ${source} ${formatMinutes(point.p50Minutes)}`;
              },
            },
          },
        },
        scales: {
          x: {
            ...compactRange(points.concat(estimatedPoints).map((point) => point.x)),
            title: axisTitle(metrComparisonLabel()),
            ticks: { maxTicksLimit: isMobileChart() ? 5 : 8 },
          },
          y: {
            type: "logarithmic",
            title: axisTitle("METR p50 time horizon"),
            ticks: yAxisTicks({ callback: (value) => formatMinutes(Number(value)) }),
          },
        },
      },
    });
  }

  function epochEciMap() {
    return new Map((state.data.epoch_eci_scores || []).map((row) => [row.model_id, row]));
  }

  function eciCorrelationPairs() {
    const epochScores = epochEciMap();
    const metricRows = state.filters.benchmark ? selectedBenchmarkMetricRows() : selectedCapabilityMetricRows();
    return metricRows
      .map((row) => {
        const epoch = epochScores.get(row.model_id);
        if (!epoch || !Number.isFinite(Number(epoch.eci))) return null;
        return {
          model: row.model,
          model_id: row.model_id,
          lab: row.lab,
          xValue: Number(row.value),
          yValue: Number(epoch.eci),
          valueText: row.valueText,
          epochText: Number(epoch.eci).toFixed(1),
        };
      })
      .filter((pair) => pair && Number.isFinite(pair.xValue) && Number.isFinite(pair.yValue));
  }

  function eciProjectedPairs(officialPairs) {
    if (state.filters.benchmark || officialPairs.length < 3) return [];
    const fit = linearRegression(officialPairs.map((pair) => ({ x: pair.xValue, y: pair.yValue })));
    if (!fit) return [];
    const officialIds = new Set(officialPairs.map((pair) => pair.model_id));
    return selectedCapabilityMetricRows()
      .filter((row) => !officialIds.has(row.model_id))
      .filter((row) => Number(row.value) >= ECI_PROJECTION_MIN_INDEX)
      .sort((a, b) => b.value - a.value)
      .slice(0, ECI_PROJECTION_LIMIT)
      .map((row) => {
        const projectedEci = fit.intercept + fit.slope * row.value;
        return {
          model: row.model,
          model_id: row.model_id,
          lab: row.lab,
          xValue: row.value,
          yValue: projectedEci,
          valueText: row.valueText,
          epochText: projectedEci.toFixed(1),
          projected: true,
        };
      })
      .filter((pair) => Number.isFinite(pair.yValue));
  }

  function renderEciCorrelation() {
    const title = $("[data-eci-title]");
    const meta = $("[data-eci-meta]");
    const sourceLink = $("[data-eci-source-link]");
    const comparisonLabel = state.filters.benchmark ? `${state.filters.benchmark} score` : jakeCapabilityIndexLabel();
    if (title) title.textContent = `${comparisonLabel} vs ECI`;
    if (sourceLink) sourceLink.href = state.data.epoch_eci_metadata?.source_url || "https://epoch.ai/eci";

    const pairs = eciCorrelationPairs();
    const projectedPairs = eciProjectedPairs(pairs);
    if (pairs.length < 3) {
      if (meta) meta.textContent = "";
      setEmpty("eci", true);
      return;
    }
    setEmpty("eci", false);

    const points = pairs.map((pair) => ({
      x: pair.xValue,
      y: pair.yValue,
      label: pair.model,
      lab: pair.lab,
      valueText: pair.valueText,
      epochText: pair.epochText,
    }));
    const projectedPoints = projectedPairs.map((pair) => ({
      x: pair.xValue,
      y: pair.yValue,
      label: pair.model,
      lab: pair.lab,
      valueText: pair.valueText,
      epochText: pair.epochText,
      projected: true,
    }));
    const trend = linearTrend(points);
    if (meta) meta.textContent = `Pearson r = ${pearsonCorrelation(pairs, "yValue").toFixed(2)} (n = ${pairs.length})`;

    const canvas = document.querySelector("canvas[data-chart='eci']");
    if (!canvas) return;
    destroyChart("eci");
    state.charts.eci = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Models",
            data: points,
            parsing: false,
            backgroundColor: points.map((point) => withAlpha(labColor(point.lab), 0.85)),
            borderColor: points.map((point) => labColor(point.lab)),
            pointRadius: 5,
            pointHoverRadius: 8,
          },
          {
            label: "Projected",
            data: projectedPoints,
            parsing: false,
            backgroundColor: "rgba(255, 255, 255, 0.9)",
            borderColor: projectedPoints.map((point) => labColor(point.lab)),
            pointStyle: "triangle",
            pointRadius: 6,
            pointHoverRadius: 9,
            pointBorderWidth: 2,
          },
          {
            type: "line",
            label: "Fit",
            data: trend,
            parsing: false,
            borderColor: "#111827",
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: chartPadding() },
        plugins: {
          legend: {
            position: "top",
            labels: { boxWidth: 14, font: { size: 11 } },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                if (ctx.dataset.label === "Fit") return `Fit: ECI ${ctx.parsed.y.toFixed(1)}`;
                const point = ctx.dataset.data[ctx.dataIndex];
                const eciLabel = point.projected ? "projected ECI" : "ECI";
                return `${point.label} (${point.lab}): ${comparisonLabel} ${point.valueText}, ${eciLabel} ${point.epochText}`;
              },
            },
          },
        },
        scales: {
          x: {
            ...compactRange(points.concat(projectedPoints).map((point) => point.x), 0.05, true),
            title: axisTitle(comparisonLabel),
            ticks: { maxTicksLimit: isMobileChart() ? 5 : 8 },
          },
          y: {
            ...compactRange(points.concat(projectedPoints).map((point) => point.y), 0.05, true),
            title: axisTitle("ECI"),
            ticks: yAxisTicks(),
          },
        },
      },
    });
  }

  function linearTrend(points) {
    const fit = linearRegression(points);
    if (!fit) return [];
    const xs = points.map((point) => point.x);
    const endpoints = [Math.min(...xs), Math.max(...xs)];
    return endpoints.map((x) => ({ x, y: fit.intercept + fit.slope * x }));
  }

  function linearRegression(points) {
    if (points.length < 2) return null;
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    const xMean = xs.reduce((sum, value) => sum + value, 0) / xs.length;
    const yMean = ys.reduce((sum, value) => sum + value, 0) / ys.length;
    const denominator = xs.reduce((sum, value) => sum + (value - xMean) ** 2, 0);
    if (!denominator) return null;
    const slope = xs.reduce((sum, value, index) => sum + (value - xMean) * (ys[index] - yMean), 0) / denominator;
    return { slope, intercept: yMean - slope * xMean };
  }

  function logYTrend(points) {
    if (points.length < 2) return [];
    const logPoints = points
      .filter((point) => point.y > 0)
      .map((point) => ({ x: point.x, y: Math.log2(point.y) }));
    return linearTrend(logPoints).map((point) => ({ x: point.x, y: 2 ** point.y }));
  }

  function formatMinutes(minutes) {
    if (!Number.isFinite(minutes)) return "unknown horizon";
    if (minutes < 60) return `${minutes.toFixed(1)} min`;
    return `${(minutes / 60).toFixed(1)} hr`;
  }

  function updateRankingHeader() {
    const title = $("[data-ranking-title]");
    const meta = $("[data-ranking-meta]");
    const sourceLink = $("[data-source-link]");
    if (title) title.textContent = rankingTitleText();
    if (meta) meta.textContent = rankingSubtitleText();
    if (sourceLink) {
      const benchmark = selectedBenchmark();
      sourceLink.hidden = !benchmark?.url;
      if (benchmark?.url) sourceLink.href = benchmark.url;
    }
  }

  function renderRanking() {
    const rows = filteredResults();
    const allEntries = rankingEntries(rows);
    const entries = allEntries.slice(0, 40);
    const isBenchmark = !!state.filters.benchmark;

    updateRankingHeader();

    if (!entries.length) {
      setEmpty("ranking", true);
      return;
    }
    setEmpty("ranking", false);

    const labels = entries.map((e) => e.label);
    const values = entries.map((e) => e.value);
    const colors = entries.map((e) => labColor(e.lab));
    const canvas = document.querySelector("canvas[data-chart='ranking']");
    if (!canvas) return;
    canvas.style.height = isBenchmark ? "420px" : "460px";

    destroyChart("ranking");
    state.charts.ranking = new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: isBenchmark ? "Normalized score" : "Capabilities index",
          data: values,
          backgroundColor: colors.map((c) => withAlpha(c, 0.85)),
          borderColor: colors,
          borderWidth: 1,
        }],
      },
      options: {
        indexAxis: "x",
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: chartPadding() },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const e = entries[ctx.dataIndex];
                return `${e.valueText} - ${e.detail}`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: rankingTickOptions(),
          },
          y: {
            beginAtZero: isBenchmark ? true : false,
            title: axisTitle(isBenchmark ? "Normalized score (%)" : "Capabilities index"),
            ticks: yAxisTicks(isBenchmark ? { callback: (v) => `${v}%` } : {}),
          },
        },
      },
    });
  }

  function timeEntries(rows) {
    const releaseDates = modelReleaseDateMap();
    const useBenchmark = !!state.filters.benchmark;
    if (useBenchmark) {
      const best = new Map();
      rows.forEach((r) => {
        const k = r.capability_model_id || r.model_id;
        const cur = best.get(k);
        if (!cur || r.normalized_score > cur.normalized_score) best.set(k, r);
      });
      return Array.from(best.values())
        .map((r) => {
          const k = r.capability_model_id || r.model_id;
          const date = r.release_date || releaseDates.get(k) || releaseDates.get(r.model_id);
          return {
            label: r.capability_model || r.model,
            lab: r.lab,
            date,
            value: Number(r.normalized_score) * 100,
            valueText: pct(r.normalized_score),
          };
        })
        .filter((e) => parseDate(e.date) != null);
    }
    const allowed = new Set(rows.map((r) => r.capability_model_id || r.model_id));
    return selectedCapabilityRows()
      .filter((r) => allowed.has(r.model_id))
      .map((r) => {
        const date = r.release_date || releaseDates.get(r.model_id);
        return {
          label: r.model,
          lab: r.lab,
          date,
          value: Number(r.index),
          valueText: r.index.toFixed(1),
        };
      })
      .filter((e) => parseDate(e.date) != null);
  }

  function parseDate(value) {
    if (!value) return null;
    const text = String(value).trim();
    let m = /^(\d{4})$/.exec(text);
    if (m) return new Date(Date.UTC(Number(m[1]), 0, 1));
    m = /^(\d{4})-(\d{2})$/.exec(text);
    if (m) return new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, 1));
    m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text);
    if (m) return new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
    const parsed = Date.parse(text);
    return Number.isFinite(parsed) ? new Date(parsed) : null;
  }

  function frontierLine(points) {
    const sorted = points.slice().sort((a, b) => a.x - b.x);
    const out = [];
    let best = -Infinity;
    for (const p of sorted) {
      if (p.y > best) {
        out.push({ x: p.x, y: p.y });
        best = p.y;
      }
    }
    return out;
  }

  function renderTime() {
    const rows = filteredResults();
    const entries = timeEntries(rows);
    const isBenchmark = !!state.filters.benchmark;

    if (!entries.length) {
      updateRankingHeader();
      setEmpty("time", true);
      return;
    }
    setEmpty("time", false);

    const points = entries.map((e) => ({
      x: parseDate(e.date).getTime(),
      y: e.value,
      label: e.label,
      lab: e.lab,
      valueText: e.valueText,
      date: e.date,
    }));

    const frontier = frontierLine(points.map((p) => ({ x: p.x, y: p.y })));

    updateRankingHeader();

    const canvas = document.querySelector("canvas[data-chart='time']");
    if (!canvas) return;
    destroyChart("time");
    state.charts.time = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Models",
            data: points,
            parsing: false,
            backgroundColor: points.map((p) => withAlpha(labColor(p.lab), 0.85)),
            borderColor: points.map((p) => labColor(p.lab)),
            pointRadius: 5,
            pointHoverRadius: 8,
          },
          {
            type: "line",
            label: "Frontier",
            data: frontier,
            parsing: false,
            borderColor: "rgba(220, 38, 38, 0.85)",
            borderWidth: 2,
            borderDash: [6, 4],
            fill: false,
            pointRadius: 0,
            datalabels: { display: false },
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: chartPadding() },
        plugins: {
          legend: {
            position: "top",
            labels: { boxWidth: 14, font: { size: 11 } },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                if (ctx.dataset.label === "Frontier") return `Frontier: ${ctx.parsed.y.toFixed(1)}`;
                const p = ctx.dataset.data[ctx.dataIndex];
                return `${p.label} (${p.lab}): ${p.valueText} on ${dateFull.format(new Date(p.x))}`;
              },
            },
          },
        },
        scales: {
          x: {
            ...compactRange(points.map((point) => point.x), 0.01),
            type: "time",
            bounds: "data",
            time: { unit: "month", tooltipFormat: "MMM yyyy" },
            title: axisTitle("Release date"),
            ticks: { maxTicksLimit: isMobileChart() ? 4 : 8 },
          },
          y: {
            title: axisTitle(isBenchmark ? "Normalized score (%)" : "Capabilities index"),
            ticks: yAxisTicks(isBenchmark ? { callback: (v) => `${v}%` } : {}),
          },
        },
      },
    });
  }

  function countryForLab(lab) {
    if (US_LABS.has(lab)) return "US";
    if (CHINA_LABS.has(lab)) return "China";
    return null;
  }

  function countryFrontierEntries(rows) {
    const releaseDates = modelReleaseDateMap();
    if (state.filters.benchmark) {
      const best = new Map();
      rows.forEach((row) => {
        const country = countryForLab(row.lab);
        if (!country || row.normalized_score == null) return;
        const modelId = row.capability_model_id || row.model_id;
        const key = `${country}::${modelId}`;
        const current = best.get(key);
        if (!current || row.normalized_score > current.normalized_score) best.set(key, row);
      });
      return Array.from(best.values()).map((row) => {
        const modelId = row.capability_model_id || row.model_id;
        const date = releaseDates.get(modelId) || releaseDates.get(row.model_id);
        return {
          country: countryForLab(row.lab),
          modelId,
          label: row.capability_model || row.model,
          lab: row.lab,
          date,
          value: Number(row.normalized_score) * 100,
          valueText: pct(row.normalized_score),
        };
      }).filter((entry) => entry.country && parseDate(entry.date));
    }

    const allowed = new Set(rows.map((row) => row.capability_model_id || row.model_id));
    return selectedCapabilityRows()
      .filter((row) => allowed.has(row.model_id))
      .map((row) => ({
        country: countryForLab(row.lab),
        modelId: row.model_id,
        label: row.model,
        lab: row.lab,
        date: row.release_date || releaseDates.get(row.model_id),
        value: Number(row.index),
        valueText: row.index.toFixed(1),
      }))
      .filter((entry) => entry.country && parseDate(entry.date));
  }

  function countryFrontierPoints(entries, country) {
    const sorted = entries
      .filter((entry) => entry.country === country)
      .map((entry) => ({ ...entry, x: parseDate(entry.date).getTime(), y: entry.value }))
      .sort((a, b) => a.x - b.x || b.y - a.y);
    const points = [];
    let best = -Infinity;
    for (const entry of sorted) {
      if (entry.y > best) {
        points.push(entry);
        best = entry.y;
      }
    }
    return points;
  }

  function currentCountryFrontiers() {
    const entries = countryFrontierEntries(filteredResults());
    return {
      entries,
      usPoints: countryFrontierPoints(entries, "US"),
      chinaPoints: countryFrontierPoints(entries, "China"),
    };
  }

  function renderFrontier() {
    const { usPoints, chinaPoints } = currentCountryFrontiers();
    const isBenchmark = !!state.filters.benchmark;

    if (!usPoints.length && !chinaPoints.length) {
      setEmpty("frontier", true);
      return;
    }
    setEmpty("frontier", false);

    const canvas = document.querySelector("canvas[data-chart='frontier']");
    if (!canvas) return;
    destroyChart("frontier");
    state.charts.frontier = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        datasets: [
          {
            label: "US frontier",
            data: usPoints,
            parsing: false,
            borderColor: "#2563eb",
            backgroundColor: withAlpha("#2563eb", 0.12),
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 7,
            tension: 0,
          },
          {
            label: "China frontier",
            data: chinaPoints,
            parsing: false,
            borderColor: "#dc2626",
            backgroundColor: withAlpha("#dc2626", 0.12),
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 7,
            tension: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: chartPadding() },
        plugins: {
          legend: {
            position: "top",
            labels: { boxWidth: 14, font: { size: 11 } },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const p = ctx.dataset.data[ctx.dataIndex];
                return `${p.label} (${p.lab}): ${p.valueText} on ${dateFull.format(new Date(p.x))}`;
              },
            },
          },
        },
        scales: {
          x: {
            ...compactRange(usPoints.concat(chinaPoints).map((point) => point.x), 0.01),
            type: "time",
            bounds: "data",
            time: { unit: "month", tooltipFormat: "MMM yyyy" },
            title: axisTitle("Release date"),
            ticks: { maxTicksLimit: isMobileChart() ? 4 : 8 },
          },
          y: {
            title: axisTitle(isBenchmark ? "Normalized score (%)" : "Capabilities index"),
            ticks: yAxisTicks(isBenchmark ? { callback: (v) => `${v}%` } : {}),
          },
        },
      },
    });
  }

  function frontierScoreDeltaPoints(usPoints, chinaPoints) {
    const eventDates = uniqueSorted(usPoints.concat(chinaPoints).map((point) => String(point.x)))
      .map((value) => Number(value))
      .sort((a, b) => a - b);
    const deltas = [];
    let usIndex = -1;
    let chinaIndex = -1;

    for (const date of eventDates) {
      while (usIndex + 1 < usPoints.length && usPoints[usIndex + 1].x <= date) usIndex += 1;
      while (chinaIndex + 1 < chinaPoints.length && chinaPoints[chinaIndex + 1].x <= date) chinaIndex += 1;
      if (usIndex < 0 || chinaIndex < 0) continue;

      const usPoint = usPoints[usIndex];
      const chinaPoint = chinaPoints[chinaIndex];
      deltas.push({
        x: date,
        y: usPoint.y - chinaPoint.y,
        usPoint,
        chinaPoint,
      });
    }

    return deltas;
  }

  function firstFrontierReachDate(points, targetScore) {
    const epsilon = 1e-9;
    const point = points.find((candidate) => candidate.y + epsilon >= targetScore);
    return point ? point.x : null;
  }

  function yearsBetween(start, end) {
    return (end - start) / (86400000 * 365.25);
  }

  function frontierTimeLagPoints(usPoints, chinaPoints) {
    return frontierScoreDeltaPoints(usPoints, chinaPoints)
      .map((point) => {
        const usLeads = point.usPoint.y >= point.chinaPoint.y;
        const referenceDate = usLeads
          ? firstFrontierReachDate(usPoints, point.chinaPoint.y)
          : firstFrontierReachDate(chinaPoints, point.usPoint.y);
        if (referenceDate == null) return null;
        return {
          ...point,
          y: yearsBetween(referenceDate, point.x) * (usLeads ? 1 : -1),
          scoreDelta: point.y,
          referenceDate,
          direction: usLeads ? "China lag" : "China lead",
        };
      })
      .filter(Boolean);
  }

  function frontierDeltaPoints(usPoints, chinaPoints) {
    if (state.frontierDeltaMode === "time") return frontierTimeLagPoints(usPoints, chinaPoints);
    return frontierScoreDeltaPoints(usPoints, chinaPoints);
  }

  function frontierDeltaValueText(value) {
    if (state.frontierDeltaMode === "time") {
      const abs = Math.abs(value);
      return abs < 0.1 ? "0.0 years" : `${value.toFixed(1)} years`;
    }
    return value.toFixed(1);
  }

  function frontierDeltaFillPoints(points, side) {
    return points.map((point) => ({
      ...point,
      y: side === "positive" ? Math.max(point.y, 0) : Math.min(point.y, 0),
    }));
  }

  function currentDateTime() {
    const now = new Date();
    return Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  }

  function durationText(milliseconds) {
    const days = Math.max(0, Math.ceil(milliseconds / 86400000));
    if (days < 60) return `${days} ${days === 1 ? "day" : "days"}`;
    const months = Math.floor(days / 30.4375);
    const remainingDays = Math.round(days - months * 30.4375);
    if (months < 24) {
      return `${months} ${months === 1 ? "month" : "months"}, ${remainingDays} ${remainingDays === 1 ? "day" : "days"}`;
    }
    const years = days / 365.25;
    return `${years.toFixed(1)} years`;
  }

  function linearFit(points) {
    if (points.length < 2) return null;
    const firstX = points[0].x;
    const xs = points.map((point) => (point.x - firstX) / 86400000);
    const ys = points.map((point) => point.y);
    const xMean = xs.reduce((sum, value) => sum + value, 0) / xs.length;
    const yMean = ys.reduce((sum, value) => sum + value, 0) / ys.length;
    const denominator = xs.reduce((sum, value) => sum + (value - xMean) ** 2, 0);
    if (!denominator) return null;

    const slope = xs.reduce((sum, value, index) => sum + (value - xMean) * (ys[index] - yMean), 0) / denominator;
    return {
      firstX,
      slope,
      intercept: yMean - slope * xMean,
    };
  }

  function updateFableChinaEstimate() {
    const meta = document.querySelector("[data-frontier-projection-meta]");
    if (!meta) return;

    const capabilityRows = state.data.capability_index || [];
    const fable = capabilityRows.find((row) => row.model_id === "claude-fable-5");
    const chinaEntries = capabilityRows
      .map((row) => ({
        country: countryForLab(row.lab),
        label: row.model,
        lab: row.lab,
        date: row.release_date,
        value: Number(row.index),
      }))
      .filter((entry) => entry.country === "China" && parseDate(entry.date));
    const chinaPoints = countryFrontierPoints(chinaEntries, "China");
    const fit = linearFit(chinaPoints);
    if (!fable || !fit || fit.slope <= 0) {
      meta.textContent = "";
      return;
    }

    const projectedDays = (Number(fable.index) - fit.intercept) / fit.slope;
    const projectedChinaReachDate = fit.firstX + projectedDays * 86400000;
    const timeRemaining = projectedChinaReachDate - currentDateTime();
    meta.textContent = `Time to Chinese model with Fable-level capabilities: ${durationText(timeRemaining)}`;
  }

  function frontierDeltaStepSeries(points, endX = currentDateTime()) {
    if (!points.length) return [];
    const series = [{ ...points[0] }];
    for (let index = 1; index < points.length; index += 1) {
      const previous = points[index - 1];
      const point = points[index];
      series.push({ ...previous, x: point.x, isStepHelper: true });
      series.push({ ...point });
    }
    const last = points[points.length - 1];
    if (Number.isFinite(endX) && endX > last.x) {
      series.push({ ...last, x: endX, isExtension: true });
    }
    return series;
  }

  function frontierDeltaColor(value) {
    return value >= 0 ? "#2563eb" : "#dc2626";
  }

  function trendLine(points, endX = null) {
    const fit = linearFit(points);
    if (!fit) return [];
    const last = points[points.length - 1];
    const endpoints = [points[0], { ...last, x: endX && endX > last.x ? endX : last.x }];
    return endpoints.map((point) => {
      const x = (point.x - fit.firstX) / 86400000;
      return { x: point.x, y: fit.intercept + fit.slope * x };
    });
  }

  function renderFrontierDelta() {
    const { usPoints, chinaPoints } = currentCountryFrontiers();
    const deltaPoints = frontierDeltaPoints(usPoints, chinaPoints);
    const isBenchmark = !!state.filters.benchmark;
    const isTimeMode = state.frontierDeltaMode === "time";
    updateFableChinaEstimate();

    if (!deltaPoints.length) {
      setEmpty("frontierDelta", true);
      return;
    }
    setEmpty("frontierDelta", false);

    const endX = currentDateTime();
    const deltaSeries = frontierDeltaStepSeries(deltaPoints, endX);
    const positiveSeries = frontierDeltaFillPoints(deltaSeries, "positive");
    const negativeSeries = frontierDeltaFillPoints(deltaSeries, "negative");
    const trend = trendLine(deltaPoints, endX);

    const canvas = document.querySelector("canvas[data-chart='frontierDelta']");
    if (!canvas) return;
    destroyChart("frontierDelta");
    state.charts.frontierDelta = new Chart(canvas.getContext("2d"), {
      data: {
        datasets: [
          {
            type: "line",
            label: "US lead",
            data: positiveSeries,
            parsing: false,
            borderColor: "transparent",
            borderWidth: 0,
            pointRadius: 0,
            fill: "origin",
            backgroundColor: withAlpha("#2563eb", 0.18),
            deltaRole: "area",
          },
          {
            type: "line",
            label: "China lead",
            data: negativeSeries,
            parsing: false,
            borderColor: "transparent",
            borderWidth: 0,
            pointRadius: 0,
            fill: "origin",
            backgroundColor: withAlpha("#dc2626", 0.18),
            deltaRole: "area",
          },
          {
            type: "line",
            label: "US lead over China",
            data: deltaSeries,
            parsing: false,
            borderColor: "#2563eb",
            borderWidth: 2,
            fill: false,
            pointRadius: (ctx) => ctx.raw?.isStepHelper || ctx.raw?.isExtension ? 0 : 4,
            pointHoverRadius: (ctx) => ctx.raw?.isStepHelper || ctx.raw?.isExtension ? 0 : 7,
            pointBackgroundColor: deltaSeries.map((point) => frontierDeltaColor(point.y)),
            pointBorderColor: "#ffffff",
            pointBorderWidth: 1.5,
            segment: {
              borderColor: (ctx) => frontierDeltaColor(ctx.p1.parsed.y),
            },
            deltaRole: "line",
          },
          {
            type: "line",
            label: "Trend",
            data: trend,
            parsing: false,
            borderColor: "#111827",
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: chartPadding() },
        plugins: {
          legend: {
            position: "top",
            labels: {
              boxWidth: 14,
              font: { size: 11 },
              filter: (item, data) => data.datasets[item.datasetIndex].deltaRole !== "line",
            },
          },
          tooltip: {
            filter: (ctx) => {
              const point = ctx.dataset.data[ctx.dataIndex];
              return ctx.dataset.deltaRole !== "area" && !point?.isStepHelper && !point?.isExtension;
            },
            callbacks: {
              label: (ctx) => {
                if (ctx.dataset.label === "Trend") return `Trend: ${frontierDeltaValueText(ctx.parsed.y)}`;
                const point = ctx.dataset.data[ctx.dataIndex];
                const lines = isTimeMode
                  ? [
                    `${point.direction}: ${frontierDeltaValueText(Math.abs(point.y))}`,
                    `Score gap: ${point.scoreDelta.toFixed(1)}`,
                  ]
                  : [`US lead over China: ${frontierDeltaValueText(point.y)}`];
                return lines.concat([
                  `US: ${point.usPoint.label} (${point.usPoint.valueText})`,
                  `China: ${point.chinaPoint.label} (${point.chinaPoint.valueText})`,
                ]);
              },
            },
          },
        },
        scales: {
          x: {
            ...compactRange(deltaSeries.map((point) => point.x), 0.01),
            type: "time",
            bounds: "data",
            time: { unit: "month", tooltipFormat: "MMM yyyy" },
            title: axisTitle("Frontier date"),
            ticks: { maxTicksLimit: isMobileChart() ? 4 : 8 },
          },
          y: {
            title: axisTitle(isTimeMode ? "US lead over China (years)" : (isBenchmark ? "US lead over China (pp)" : "US lead over China (score)")),
            ticks: isTimeMode
              ? yAxisTicks({ callback: (value) => `${value}y` })
              : yAxisTicks(isBenchmark ? { callback: (value) => `${value}pp` } : {}),
          },
        },
      },
    });
  }

  function render() {
    if (state.rankingView === "bar") renderRanking();
    else renderTime();
    renderMetrCorrelation();
    renderEciCorrelation();
    renderFrontier();
    renderFrontierDelta();
  }

  fetch("/data/benchmark-dashboard.json", { cache: "no-store" })
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((data) => {
      state.data = data;
      initControls();
      render();
      setUpdatedDate(data.generated_at);
      setStatus("");
    })
    .catch((err) => {
      setStatus(`Could not load benchmark dataset: ${err.message}`);
    });
}());
