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
    const digits = n < 0.01 ? 4 : n < 0.1 ? 3 : n < 10 ? 2 : n < 100 ? 1 : 0;
    return `$${n.toFixed(digits).replace(/\.?0+$/, "")}`;
  };
  const dollars = (value) => value == null || Number.isNaN(value) ? "" : `$${fmt.format(Math.round(value))}`;
  const compactDollars = (value) => {
    if (value == null || Number.isNaN(value)) return "";
    const n = Number(value);
    const abs = Math.abs(n);
    const sign = n < 0 ? "-" : "";
    if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
    if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(0)}K`;
    return `${sign}$${abs.toFixed(0)}`;
  };
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
    if (row.id === "capability-index") return "Jake's Index";
    if (row.id === "epoch-eci" || row.name === "Epoch ECI") return "ECI";
    return row.benchmark_id ? row.name : row.name.replace(" Capabilities Index", " Index");
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

  function lightColorFor(row) {
    if (row.kind === "aggregate") return "rgba(37, 99, 235, 0.18)";
    if (row.kind === "category_index") return "rgba(79, 70, 229, 0.18)";
    return "rgba(16, 163, 127, 0.18)";
  }

  function lightBorderFor(row) {
    if (row.kind === "aggregate") return "rgba(37, 99, 235, 0.38)";
    if (row.kind === "category_index") return "rgba(79, 70, 229, 0.38)";
    return "rgba(16, 163, 127, 0.38)";
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
      .filter((row) => {
        const primary = options.value(row);
        const secondary = options.secondaryValue ? options.secondaryValue(row) : null;
        return (
          (primary != null && !Number.isNaN(primary))
          || (secondary != null && !Number.isNaN(secondary))
        );
      })
      .slice(0, limit);
    const datasets = [{
      label: options.datasetLabel,
      data: chartRows.map((row) => options.value(row)),
      backgroundColor: chartRows.map((row) => options.backgroundColor ? options.backgroundColor(row) : "rgba(37, 99, 235, 0.72)"),
      borderColor: chartRows.map((row) => options.borderColor ? options.borderColor(row) : "#2563eb"),
      borderWidth: 1,
    }];
    if (options.secondaryValue) {
      datasets.push({
        label: options.secondaryDatasetLabel,
        data: chartRows.map((row) => options.secondaryValue(row)),
        backgroundColor: chartRows.map((row) => options.secondaryBackgroundColor ? options.secondaryBackgroundColor(row) : "rgba(217, 119, 6, 0.72)"),
        borderColor: chartRows.map((row) => options.secondaryBorderColor ? options.secondaryBorderColor(row) : "#b45309"),
        borderWidth: 1,
      });
    }
    setChartWidth(canvas, chartRows.length, options.baseWidth, options.columnWidth, options.minWidth, options.maxWidth);
    new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: chartRows.map((row) => options.label(row)),
        datasets,
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: datasets.length > 1 },
          tooltip: {
            callbacks: {
              label: (ctx) => (
                ctx.datasetIndex === 1
                  ? options.secondaryTooltip(chartRows[ctx.dataIndex], ctx.parsed.y)
                  : options.tooltip(chartRows[ctx.dataIndex], ctx.parsed.y)
              ),
            },
          },
        },
        scales: {
          x: {
            stacked: Boolean(options.stacked),
            ticks: {
              autoSkip: false,
              maxRotation: 60,
              minRotation: 45,
            },
          },
          y: {
            stacked: Boolean(options.stacked),
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
    const scoreRows = data.score_usage_examples?.top_score_correlations || [];
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

    renderVerticalChart(root.querySelector("canvas[data-openrouter-chart='score-r2']"), scoreRows, {
      datasetLabel: "Score-only R^2",
      label: metricName,
      value: (row) => (row.score_only_r2 || 0) * 100,
      tooltip: (row) => `R^2 ${pct(row.score_only_r2)}; r ${num(row.score_log_usage_pearson, 2)}; ${fmt.format(row.matched_model_count || 0)} models`,
      tick: (value) => `${value}%`,
      max: 100,
      columnWidth: 28,
      backgroundColor: colorFor,
      borderColor: borderFor,
    });

    renderVerticalChart(root.querySelector("canvas[data-openrouter-chart='incremental-r2']"), adjustedRows, {
      datasetLabel: "Incremental R^2 from benchmark",
      secondaryDatasetLabel: "Price-only R^2",
      label: metricName,
      value: (row) => (row.incremental_r2_after_price || 0) * 100,
      tooltip: (row) => `${pp(row.incremental_r2_after_price)} incremental; total R^2 ${pct(row.price_score_r2)} across ${fmt.format(row.matched_model_count || 0)} models`,
      secondaryValue: (row) => (row.price_only_r2_on_matched_models || 0) * 100,
      secondaryTooltip: (row) => `Price-only R^2 ${pct(row.price_only_r2_on_matched_models)}; total score + price R^2 ${pct(row.price_score_r2)}`,
      tick: (value) => `${value} pp`,
      max: 100,
      columnWidth: 28,
      backgroundColor: colorFor,
      borderColor: borderFor,
      secondaryBackgroundColor: lightColorFor,
      secondaryBorderColor: lightBorderFor,
      stacked: true,
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

  function adaptiveLogTicks(min, max, maxTicks = 7) {
    if (!(min > 0) || !(max > min)) return [];
    const span = Math.log10(max / min);
    const mantissas = span > 3.5 ? [1] : span > 1.8 ? [1, 5] : [1, 2, 5];
    const values = [];
    const firstExponent = Math.floor(Math.log10(min)) - 1;
    const lastExponent = Math.ceil(Math.log10(max)) + 1;
    for (let exponent = firstExponent; exponent <= lastExponent; exponent += 1) {
      for (const mantissa of mantissas) {
        const value = mantissa * Math.pow(10, exponent);
        if (value >= min && value <= max) values.push(value);
      }
    }
    if (!values.length) {
      return Array.from({ length: 5 }, (_, index) => (
        Math.exp(Math.log(min) + (Math.log(max) - Math.log(min)) * index / 4)
      ));
    }
    if (values.length <= maxTicks) return values;
    return Array.from({ length: maxTicks }, (_, index) => (
      values[Math.round(index * (values.length - 1) / (maxTicks - 1))]
    )).filter((value, index, array) => index === 0 || value !== array[index - 1]);
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
    const prefillCostInput = pricingTool.querySelector("[data-prefill-cost]");
    const outputCostInput = pricingTool.querySelector("[data-output-cost]");
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
      const prefillCost = inputPrice * 0.1;
      const outputCost = outputPrice * 0.1;
      const anchor = Math.log(Math.max(share || 0, 1e-12))
        - priceCoefficient * Math.log(Math.max(effective, 1e-12))
        - capabilityCoefficient * capability;
      return {
        inputPrice,
        outputPrice,
        prefillCost,
        outputCost,
        ratio,
        effective,
        share,
        capability,
        anchor,
      };
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
      setInputNumber(prefillCostInput, base.prefillCost, priceInputValue);
      setInputNumber(outputCostInput, base.outputCost, priceInputValue);
      setInputNumber(tokenRatioInput, base.ratio, (value) => compactNumber(value, value > 10 ? 2 : 3));
      setInputNumber(effectivePriceInput, base.effective, priceInputValue);
      setInputNumber(marketShareInput, base.share, shareInputValue, base.share == null ? base.share : base.share * 100);
      setInputNumber(capabilityInput, base.capability, (value) => compactNumber(value, 1));
    }

    function renderDemandChart(effective, capability, share) {
      if (!demandCanvas || !window.Chart || !base) return;
      const ratio = Math.max(0.001, readNumber(tokenRatioInput, base.ratio));
      const prefillCost = Math.max(0, readNumber(prefillCostInput, base.prefillCost));
      const outputCost = Math.max(0, readNumber(outputCostInput, base.outputCost));
      const effectiveServingCost = effectivePrice(prefillCost, outputCost, ratio);
      const modeledShare = clamp(shareFor(effective, capability) || share, 1e-8, 0.99999);
      const highShareTarget = clamp(modeledShare * 6, modeledShare, 0.85);
      const lowShareTarget = clamp(modeledShare / 8, 1e-8, modeledShare);
      const lowTargetPrice = priceForShare(highShareTarget, capability);
      const highTargetPrice = priceForShare(lowShareTarget, capability);
      let low = Math.max(
        0.000001,
        Math.min(effective / 3, lowTargetPrice && lowTargetPrice > 0 ? lowTargetPrice : effective / 3),
      );
      let high = Math.max(
        effective * 3,
        highTargetPrice && highTargetPrice > 0 ? highTargetPrice : effective * 3,
      );
      const padding = Math.max(0.04, (Math.log(high) - Math.log(low)) * 0.03);
      low = Math.max(0.000001, Math.exp(Math.log(low) - padding));
      high = Math.exp(Math.log(high) + padding);
      const xTicks = adaptiveLogTicks(low, high);
      const lowLog = Math.log(low);
      const highLog = Math.log(high);
      const points = Array.from({ length: 80 }, (_, index) => {
        const price = Math.exp(lowLog + (highLog - lowLog) * index / 79);
        return { x: price, y: (shareFor(price, capability) || 0) * 100 };
      });
      const profitPoints = points.map((point) => ({
        x: point.x,
        y: (point.y / 100) * totalTokens / 1e6 * (point.x - effectiveServingCost),
      }));
      const currentPoint = { x: effective, y: share * 100 };
      const currentProfitPoint = {
        x: effective,
        y: share * totalTokens / 1e6 * (effective - effectiveServingCost),
      };
      const chartData = {
        datasets: [
          {
            type: "line",
            label: "Estimated token share",
            data: points,
            borderColor: "#2563eb",
            backgroundColor: "rgba(37, 99, 235, 0.12)",
            pointRadius: 0,
            borderWidth: 2,
            tension: 0.2,
            yAxisID: "yShare",
          },
          {
            type: "line",
            label: "Estimated weekly profit",
            data: profitPoints,
            borderColor: "#b45309",
            backgroundColor: "rgba(180, 83, 9, 0.08)",
            pointRadius: 0,
            borderWidth: 2,
            tension: 0.2,
            yAxisID: "yProfit",
          },
          {
            type: "scatter",
            label: "Current share",
            data: [currentPoint],
            borderColor: "#0f8f70",
            backgroundColor: "#0f8f70",
            pointRadius: 4,
            yAxisID: "yShare",
          },
          {
            type: "scatter",
            label: "Current profit",
            data: [currentProfitPoint],
            borderColor: "#b45309",
            backgroundColor: "#b45309",
            pointRadius: 4,
            yAxisID: "yProfit",
          },
        ],
      };
      if (demandChart) {
        demandChart.data = chartData;
        demandChart.options.scales.x.min = low;
        demandChart.options.scales.x.max = high;
        demandChart.options.scales.x.afterBuildTicks = (scale) => {
          scale.ticks = xTicks.map((value) => ({ value }));
        };
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
            legend: {
              display: true,
              labels: {
                filter: (item) => !item.text.startsWith("Current "),
              },
            },
            tooltip: {
              callbacks: {
                label: (ctx) => (
                  ctx.dataset.yAxisID === "yProfit"
                    ? `${money(ctx.parsed.x)} effective price; ${dollars(ctx.parsed.y)} weekly profit`
                    : `${money(ctx.parsed.x)} effective price; ${num(ctx.parsed.y, 3)}% token share`
                ),
              },
            },
          },
          scales: {
            x: {
              type: "logarithmic",
              min: low,
              max: high,
              title: { display: true, text: "Effective price ($/M tokens)" },
              afterBuildTicks: (scale) => {
                scale.ticks = xTicks.map((value) => ({ value }));
              },
              ticks: {
                maxRotation: 0,
                minRotation: 0,
                callback: (value) => axisMoney(Number(value)),
              },
            },
            yShare: {
              position: "left",
              beginAtZero: true,
              title: { display: true, text: "Estimated token share" },
              ticks: { callback: (value) => `${value}%` },
            },
            yProfit: {
              position: "right",
              beginAtZero: true,
              title: { display: true, text: "Estimated weekly profit" },
              grid: { drawOnChartArea: false },
              ticks: { callback: (value) => compactDollars(value) },
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
    [prefillCostInput, outputCostInput].forEach((element) => {
      element.addEventListener("input", () => {
        if (parseInputNumber(element) == null) return;
        render();
      });
      element.addEventListener("change", () => {
        if (parseInputNumber(element) == null) return;
        render();
      });
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
