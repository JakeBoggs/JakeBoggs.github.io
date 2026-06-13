(function () {
  const roots = Array.from(document.querySelectorAll("[data-openrouter-benchmark-analysis]"));
  if (!roots.length) return;

  const fmt = new Intl.NumberFormat("en-US");
  const pct = (value) => value == null ? "" : `${(value * 100).toFixed(1)}%`;
  const pctWhole = (value) => value == null ? "" : `${value.toFixed(1)}%`;
  const pctPrecise = (value) => {
    if (value == null || Number.isNaN(value)) return "";
    const percent = Number(value) * 100;
    const abs = Math.abs(percent);
    const digits = abs >= 10 ? 1 : abs >= 1 ? 2 : abs >= 0.1 ? 3 : abs >= 0.01 ? 4 : abs >= 0.001 ? 5 : 6;
    return `${percent.toFixed(digits)}%`;
  };
  const pp = (value) => value == null ? "" : `${(value * 100).toFixed(1)} pp`;
  const num = (value, digits = 2) => value == null || Number.isNaN(value) ? "" : Number(value).toFixed(digits);
  const money = (value) => {
    if (value == null || Number.isNaN(value)) return "";
    const n = Number(value);
    const abs = Math.abs(n);
    const digits = n === 0 ? 2 : abs < 0.01 ? 4 : abs < 1 ? 3 : abs < 10 ? 2 : 1;
    return `$${n.toFixed(digits)}/M`;
  };
  const axisMoney = (value) => {
    if (value == null || Number.isNaN(value)) return "";
    const n = Number(value);
    if (n < 1) return `$${n.toFixed(2)}`;
    if (n < 10) return `$${n.toFixed(1).replace(/\.0$/, "")}`;
    return `$${Math.round(n)}`;
  };
  const dollars = (value) => value == null || Number.isNaN(value) ? "" : `$${fmt.format(Math.round(value))}`;
  const tokens = (value) => {
    if (value == null) return "";
    const n = Number(value);
    if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    return fmt.format(Math.round(n));
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function metricName(row) {
    if (row.id === "capability-index" || row.name === "Jake Capability Index") return "Jake's Index";
    if (row.id === "epoch-eci" || row.name === "Epoch ECI") return "ECI";
    return row.benchmark_id ? row.name : row.name.replace(" Capability Index", " Index");
  }

  function colorFor(row) {
    if (row.kind === "aggregate") return "rgba(37, 99, 235, 0.78)";
    if (row.kind === "category_index") return "rgba(79, 70, 229, 0.72)";
    return "rgba(16, 163, 127, 0.72)";
  }

  function borderFor(row) {
    if (row.kind === "aggregate") return "#2563eb";
    if (row.kind === "category_index") return "#4f46e5";
    return "#0f8f70";
  }

  function setChartWidth(canvas, rowCount, base = 96, columnWidth = 28, min = 0, max = 1280) {
    const wrap = canvas.closest(".openrouter-chart-wrap");
    if (!wrap) return;
    wrap.style.minWidth = `${Math.max(min, Math.min(max, base + rowCount * columnWidth))}px`;
  }

  function renderVerticalChart(canvas, rows, options) {
    if (!canvas || !window.Chart) return;
    const limit = options.limit ?? rows.length;
    const chartRows = rows
      .filter((row) => options.value(row) != null && !Number.isNaN(options.value(row)))
      .slice(0, limit);
    setChartWidth(canvas, chartRows.length, options.baseWidth, options.columnWidth, options.minWidth, options.maxWidth);
    new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: chartRows.map((row) => options.label(row)),
        datasets: [{
          label: options.datasetLabel,
          data: chartRows.map((row) => options.value(row)),
          backgroundColor: chartRows.map((row) => options.backgroundColor ? options.backgroundColor(row) : "rgba(37, 99, 235, 0.72)"),
          borderColor: chartRows.map((row) => options.borderColor ? options.borderColor(row) : "#2563eb"),
          borderWidth: 1,
        }],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => options.tooltip(chartRows[ctx.dataIndex], ctx.parsed.y),
            },
          },
        },
        scales: {
          x: {
            ticks: {
              autoSkip: false,
              maxRotation: 60,
              minRotation: 45,
            },
          },
          y: {
            beginAtZero: true,
            max: options.max,
            ticks: { callback: options.tick },
          },
        },
      },
    });
  }

  function renderCharts(root, data) {
    const charts = root.querySelector("[data-openrouter-charts]");
    if (!charts) return;
    const price = data.price_usage_relationship || {};
    const rawRows = data.score_usage_examples?.top_raw_correlations || [];
    const adjustedRows = data.score_usage_examples?.top_incremental_r2_after_price || [];

    const priceCopy = root.querySelector("[data-price-baseline-copy]");
    if (priceCopy) {
      priceCopy.textContent = `Price-only R^2 is ${pct(price.price_only_r2)}; log-price/log-usage r is ${num(price.log_price_log_usage_pearson, 2)}; elasticity is ${num(price.price_elasticity, 2)}.`;
    }

    renderVerticalChart(root.querySelector("canvas[data-openrouter-chart='price-bands']"), price.price_bands || [], {
      datasetLabel: "Token share",
      label: (row) => row.label,
      value: (row) => (row.token_share || 0) * 100,
      tooltip: (row) => `${pct(row.token_share)} of tokens across ${fmt.format(row.model_count || 0)} models`,
      tick: (value) => `${value}%`,
      max: 60,
      minWidth: 420,
      columnWidth: 58,
      backgroundColor: () => "rgba(99, 102, 241, 0.72)",
      borderColor: () => "#4f46e5",
    });

    renderVerticalChart(root.querySelector("canvas[data-openrouter-chart='score-r2']"), rawRows, {
      datasetLabel: "Score-only R^2",
      label: metricName,
      value: (row) => (row.score_only_r2 || 0) * 100,
      tooltip: (row) => `R^2 ${pct(row.score_only_r2)}; raw r ${num(row.score_log_usage_pearson, 2)}; ${fmt.format(row.matched_model_count || 0)} models`,
      tick: (value) => `${value}%`,
      max: 100,
      columnWidth: 28,
      backgroundColor: colorFor,
      borderColor: borderFor,
    });

    renderVerticalChart(root.querySelector("canvas[data-openrouter-chart='incremental-r2']"), adjustedRows, {
      datasetLabel: "Incremental R^2 after price",
      label: metricName,
      value: (row) => (row.incremental_r2_after_price || 0) * 100,
      tooltip: (row) => `${pp(row.incremental_r2_after_price)}; partial r ${num(row.score_log_usage_price_adjusted_partial_pearson, 2)}`,
      tick: (value) => `${value} pp`,
      max: 100,
      columnWidth: 28,
      backgroundColor: colorFor,
      borderColor: borderFor,
    });

    charts.hidden = false;
  }

  function compactNumber(value, digits = 3) {
    if (value == null || Number.isNaN(value)) return "";
    const n = Number(value);
    if (n === 0) return "0";
    return n.toFixed(digits).replace(/\.?0+$/, "");
  }

  function priceInputValue(value) {
    if (value == null || Number.isNaN(value)) return "";
    const n = Number(value);
    const abs = Math.abs(n);
    const digits = abs < 0.01 ? 6 : abs < 1 ? 4 : abs < 10 ? 3 : 2;
    return compactNumber(n, digits);
  }

  function shareInputValue(value) {
    if (value == null || Number.isNaN(value)) return "";
    const percent = Number(value) * 100;
    const abs = Math.abs(percent);
    const digits = abs >= 10 ? 3 : abs >= 1 ? 4 : abs >= 0.1 ? 5 : 6;
    return compactNumber(percent, digits);
  }

  function parseInputNumber(input) {
    const raw = input.value.trim();
    if (!raw || input.validity?.badInput) return null;
    const value = Number(raw);
    if (!Number.isFinite(value)) return null;
    const min = input.getAttribute("min");
    const max = input.getAttribute("max");
    if (min !== null && min !== "" && value < Number(min)) return null;
    if (max !== null && max !== "" && value > Number(max)) return null;
    return value;
  }

  function rememberInputNumber(input, value) {
    if (value != null && Number.isFinite(Number(value))) {
      input.dataset.lastValidNumber = String(value);
    }
  }

  function setInputNumber(input, value, formatter = compactNumber, rememberedValue = value) {
    input.value = formatter(value);
    rememberInputNumber(input, rememberedValue);
  }

  function readNumber(input, fallback = 0) {
    const value = parseInputNumber(input);
    if (value != null) {
      rememberInputNumber(input, value);
      return value;
    }
    const lastValue = Number(input.dataset.lastValidNumber);
    return Number.isFinite(lastValue) ? lastValue : fallback;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function effectivePrice(inputPrice, outputPrice, inputOutputRatio) {
    const ratio = Math.max(0, inputOutputRatio || 0);
    if (ratio === 0) return outputPrice;
    return (inputPrice * ratio + outputPrice) / (ratio + 1);
  }

  function ratioForEffectivePrice(inputPrice, outputPrice, targetEffectivePrice) {
    const denominator = targetEffectivePrice - inputPrice;
    if (Math.abs(denominator) < 1e-12) return null;
    const ratio = (outputPrice - targetEffectivePrice) / denominator;
    return ratio > 0 && Number.isFinite(ratio) ? ratio : null;
  }

  function isLabeledLogTick(value) {
    if (value <= 0) return false;
    const exponent = Math.floor(Math.log10(value));
    const mantissa = value / Math.pow(10, exponent);
    return [1, 2, 5].some((target) => Math.abs(mantissa - target) < 1e-8);
  }

  function renderPricingTool(root, data) {
    const pricingTool = root.querySelector("[data-openrouter-pricing]");
    const demand = data.demand_model || {};
    const models = (data.pricing_models || []).filter((model) => (
      model.capability_index != null
      && model.prompt_price_per_million != null
      && model.completion_price_per_million != null
      && model.token_share > 0
      && model.effective_price_per_million_tokens > 0
    ));
    if (!pricingTool || !models.length) return;
    const totalTokens = data.price_usage_relationship?.total_tokens || models.reduce((sum, row) => sum + (row.total_tokens || 0), 0);
    const priceCoefficient = demand.price_coefficient ?? -1.17;
    const capabilityCoefficient = demand.capability_coefficient ?? 0.05;
    const modelSelect = pricingTool.querySelector("[data-pricing-model]");
    const inputPriceInput = pricingTool.querySelector("[data-input-price]");
    const outputPriceInput = pricingTool.querySelector("[data-output-price]");
    const tokenRatioInput = pricingTool.querySelector("[data-token-ratio]");
    const effectivePriceInput = pricingTool.querySelector("[data-effective-price]");
    const marketShareInput = pricingTool.querySelector("[data-market-share]");
    const capabilityInput = pricingTool.querySelector("[data-capability-index]");
    const demandCanvas = pricingTool.querySelector("canvas[data-openrouter-chart='demand-curve']");
    let base = null;
    let demandChart = null;

    modelSelect.innerHTML = models.map((model, index) => (
      `<option value="${index}">${escapeHtml(model.model)} (${pctPrecise(model.token_share)}, ${money(model.effective_price_per_million_tokens)}, index ${num(model.capability_index, 1)})</option>`
    )).join("");

    function selectedModel() {
      return models[Number(modelSelect.value) || 0];
    }

    function initialState(model) {
      const inputPrice = model.prompt_price_per_million || model.effective_price_per_million_tokens;
      const outputPrice = model.completion_price_per_million || inputPrice;
      const observedEffective = model.effective_price_per_million_tokens;
      const observedRatio = model.input_output_token_ratio;
      let ratio = ratioForEffectivePrice(inputPrice, outputPrice, observedEffective);
      if (ratio == null && observedRatio && observedRatio > 0) ratio = observedRatio;
      if (ratio == null) ratio = 1;
      const effective = effectivePrice(inputPrice, outputPrice, ratio);
      const share = model.token_share ?? (totalTokens > 0 ? (model.total_tokens || 0) / totalTokens : null);
      const capability = model.capability_index;
      const anchor = Math.log(Math.max(share || 0, 1e-12))
        - priceCoefficient * Math.log(Math.max(effective, 1e-12))
        - capabilityCoefficient * capability;
      return { inputPrice, outputPrice, ratio, effective, share, capability, anchor };
    }

    function shareFor(price, capability) {
      if (!base || price <= 0 || capability == null) return null;
      const logShare = base.anchor + priceCoefficient * Math.log(price) + capabilityCoefficient * capability;
      return clamp(Math.exp(logShare), 1e-12, 0.99999);
    }

    function priceForShare(share, capability) {
      if (!base || priceCoefficient === 0 || share <= 0 || share >= 1) return null;
      const logPrice = (Math.log(share) - base.anchor - capabilityCoefficient * capability) / priceCoefficient;
      const price = Math.exp(logPrice);
      return Number.isFinite(price) && price > 0 ? price : null;
    }

    function setEffectiveAndScalePrices(nextEffective, options = {}) {
      const writeEffective = options.writeEffective !== false;
      const ratio = Math.max(0.001, readNumber(tokenRatioInput, base.ratio));
      const currentInput = Math.max(0, readNumber(inputPriceInput, base.inputPrice));
      const currentOutput = Math.max(0, readNumber(outputPriceInput, base.outputPrice));
      if (currentInput > 0) {
        const outputInputPriceRatio = currentOutput / currentInput;
        const nextInput = nextEffective * (ratio + 1) / (ratio + outputInputPriceRatio);
        setInputNumber(inputPriceInput, nextInput, priceInputValue);
        setInputNumber(outputPriceInput, nextInput * outputInputPriceRatio, priceInputValue);
      } else {
        setInputNumber(inputPriceInput, nextEffective, priceInputValue);
        setInputNumber(outputPriceInput, nextEffective, priceInputValue);
      }
      rememberInputNumber(effectivePriceInput, nextEffective);
      if (writeEffective) {
        setInputNumber(effectivePriceInput, nextEffective, priceInputValue);
      }
    }

    function syncEffectiveFromPrices() {
      const inputPrice = Math.max(0, readNumber(inputPriceInput, base.inputPrice));
      const outputPrice = Math.max(0, readNumber(outputPriceInput, base.outputPrice));
      const ratio = Math.max(0.001, readNumber(tokenRatioInput, base.ratio));
      setInputNumber(effectivePriceInput, effectivePrice(inputPrice, outputPrice, ratio), priceInputValue);
    }

    function syncShareFromPrice() {
      const price = Math.max(1e-12, readNumber(effectivePriceInput, base.effective));
      const capability = readNumber(capabilityInput, base.capability);
      const share = shareFor(price, capability);
      setInputNumber(marketShareInput, share, shareInputValue, share == null ? share : share * 100);
    }

    function syncPriceFromShare() {
      const share = clamp(readNumber(marketShareInput, (base.share || 0) * 100) / 100, 1e-12, 0.99999);
      const capability = readNumber(capabilityInput, base.capability);
      const price = priceForShare(share, capability);
      if (price != null) {
        setEffectiveAndScalePrices(price);
      }
    }

    function writeInitialControls() {
      const model = selectedModel();
      base = initialState(model);
      setInputNumber(inputPriceInput, base.inputPrice, priceInputValue);
      setInputNumber(outputPriceInput, base.outputPrice, priceInputValue);
      setInputNumber(tokenRatioInput, base.ratio, (value) => compactNumber(value, value > 10 ? 2 : 3));
      setInputNumber(effectivePriceInput, base.effective, priceInputValue);
      setInputNumber(marketShareInput, base.share, shareInputValue, base.share == null ? base.share : base.share * 100);
      setInputNumber(capabilityInput, base.capability, (value) => compactNumber(value, 1));
    }

    function renderDemandChart(effective, capability, share) {
      if (!demandCanvas || !window.Chart || !base) return;
      const low = Math.max(0.000001, Math.min(effective, base.effective) / 6);
      const high = Math.max(low * 10, Math.max(effective, base.effective) * 6);
      const lowLog = Math.log(low);
      const highLog = Math.log(high);
      const points = Array.from({ length: 80 }, (_, index) => {
        const price = Math.exp(lowLog + (highLog - lowLog) * index / 79);
        return { x: price, y: (shareFor(price, capability) || 0) * 100 };
      });
      const currentPoint = { x: effective, y: share * 100 };
      const chartData = {
        datasets: [
          {
            type: "line",
            label: "Estimated paid-token share",
            data: points,
            borderColor: "#2563eb",
            backgroundColor: "rgba(37, 99, 235, 0.12)",
            pointRadius: 0,
            borderWidth: 2,
            tension: 0.2,
          },
          {
            type: "scatter",
            label: "Current setting",
            data: [currentPoint],
            borderColor: "#0f8f70",
            backgroundColor: "#0f8f70",
            pointRadius: 4,
          },
        ],
      };
      if (demandChart) {
        demandChart.data = chartData;
        demandChart.update();
        return;
      }
      demandChart = new Chart(demandCanvas.getContext("2d"), {
        data: chartData,
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => `${money(ctx.parsed.x)} effective price; ${num(ctx.parsed.y, 3)}% paid-token share`,
              },
            },
          },
          scales: {
            x: {
              type: "logarithmic",
              title: { display: true, text: "Effective price ($/M tokens)" },
              afterBuildTicks: (scale) => {
                scale.ticks = scale.ticks.filter((tick) => isLabeledLogTick(Number(tick.value)));
              },
              ticks: {
                maxTicksLimit: 8,
                maxRotation: 0,
                minRotation: 0,
                callback: (value) => isLabeledLogTick(Number(value)) ? axisMoney(Number(value)) : "",
              },
            },
            y: {
              beginAtZero: true,
              title: { display: true, text: "Estimated paid-token share" },
              ticks: { callback: (value) => `${value}%` },
            },
          },
        },
      });
    }

    function render() {
      const effective = Math.max(1e-12, readNumber(effectivePriceInput, base.effective));
      const capability = readNumber(capabilityInput, base.capability);
      const share = clamp(readNumber(marketShareInput, (base.share || 0) * 100) / 100, 1e-12, 0.99999);
      renderDemandChart(effective, capability, share);
    }

    modelSelect.addEventListener("change", () => {
      writeInitialControls();
      render();
    });

    [inputPriceInput, outputPriceInput, tokenRatioInput].forEach((element) => {
      element.addEventListener("input", () => {
        if (parseInputNumber(element) == null) return;
        syncEffectiveFromPrices();
        syncShareFromPrice();
        render();
      });
      element.addEventListener("change", () => {
        if (parseInputNumber(element) == null) return;
        syncEffectiveFromPrices();
        syncShareFromPrice();
        render();
      });
    });

    effectivePriceInput.addEventListener("input", () => {
      if (parseInputNumber(effectivePriceInput) == null) return;
      setEffectiveAndScalePrices(Math.max(1e-12, readNumber(effectivePriceInput, base.effective)), { writeEffective: false });
      syncShareFromPrice();
      render();
    });
    effectivePriceInput.addEventListener("change", () => {
      if (parseInputNumber(effectivePriceInput) == null) return;
      setEffectiveAndScalePrices(Math.max(1e-12, readNumber(effectivePriceInput, base.effective)));
      syncShareFromPrice();
      render();
    });

    marketShareInput.addEventListener("input", () => {
      if (parseInputNumber(marketShareInput) == null) return;
      syncPriceFromShare();
      render();
    });
    marketShareInput.addEventListener("change", () => {
      if (parseInputNumber(marketShareInput) == null) return;
      syncPriceFromShare();
      render();
    });

    capabilityInput.addEventListener("input", () => {
      if (parseInputNumber(capabilityInput) == null) return;
      syncShareFromPrice();
      render();
    });
    capabilityInput.addEventListener("change", () => {
      if (parseInputNumber(capabilityInput) == null) return;
      syncShareFromPrice();
      render();
    });
    writeInitialControls();
    render();
    pricingTool.hidden = false;
  }

  fetch("/data/openrouter-benchmark-analysis.json", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      roots.forEach((root) => {
        const status = root.querySelector("[data-openrouter-status]");
        renderPricingTool(root, data);
        renderCharts(root, data);
        if (status) status.hidden = true;
      });
    })
    .catch((error) => {
      roots.forEach((root) => {
        const status = root.querySelector("[data-openrouter-status]");
        if (status) status.textContent = `Could not load OpenRouter analysis: ${error.message}`;
      });
    });
}());
