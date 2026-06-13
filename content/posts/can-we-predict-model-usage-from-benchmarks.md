---
title: "How Well Do Benchmarks Predict Which Models People Use?"
date: 2026-06-13
draft: false
summary: "Evaluating LLM benchmarks by analyzing how they correlate with OpenRouter market share."
---

The purpose of a benchmark is to help answer two questions: For users, **"which model is best for my use-case?"** and for labs, **"which checkpoint should we release and how should we price it?"**

One way to test whether a benchmark is useful is to ask whether it predicts revealed preference. When two models are available at roughly the same price, does the model with the higher benchmark score get used more? If the answer is usually yes, the benchmark is probably measuring something users value. If the answer is no, either the benchmark is missing the real capability, or other factors are dominating.

Measuring this perfectly would require private data from each of the labs, so I'm using [OpenRouter](https://openrouter.ai) (a model aggregator) as a proxy for the broader market. Current volume is approaching 50 trillion tokens / week and it serves models from dozens of providers. It is also the only source I am aware of that shares the required data publicly.

## Method

I pull input and output pricing from OpenRouter's [public model catalog](https://openrouter.ai/docs/api/api-reference/models/get-models) and I scrape token usage (over the past week) from their public rankings endpoints. This is combined with [my existing benchmark data](/posts/benchmarks), including my aggregate capability index, the ECI, category sub-indices, and individual benchmark scores.

The price variable is an **effective observed price**, not just the listed input-token price or output-token price. Models have very different input/output mixes, so I estimate dollars per million served tokens from the observed token mix in OpenRouter's ranking rows:

```text
estimated_cost =
  prompt_tokens * prompt_price
  + completion_tokens * completion_price
  + cached_tokens * cache_read_price

served_tokens =
  prompt_tokens + completion_tokens

effective_price_per_million =
  estimated_cost / served_tokens * 1,000,000
```

Reasoning tokens are billed at the output rate and are already counted inside `completion_tokens`, so they are not added separately. If OpenRouter does not expose a cache-read price, I fall back to the prompt-token price for cached tokens. This effective price is still approximate, but it is much closer to the user's actual economic tradeoff than comparing models by input price alone.

The analysis then runs in three passes.

First, I estimate the price baseline:

```text
log_usage = a + b * log_effective_price
```

This answers the simplest question: how much of model usage is explained by price before looking at benchmark scores at all?

Second, I estimate raw score-to-usage relationships for each benchmark:

```text
log_usage = a + b * benchmark_score
```

This flips the previous question and answers: how much of model usage is explained by the benchmark before looking at price?

Third, I ask whether scores explain usage after accounting for price:

```text
log_usage = a + b * log_effective_price + c * benchmark_score
```

The key statistic here is incremental R^2: how much more usage variance is explained when benchmark score is added to the price-only model. I also report a partial correlation, computed by residualizing both usage and score against price and then correlating the residuals:

```text
partial_r_after_price =
  corr(residual(log_usage ~ log_price), residual(score ~ log_price))
```

So the analysis starts with the obvious confounder, price; then asks whether scores correlate with usage at all; and finally asks whether scores add explanatory power beyond price.

{{< openrouter-benchmark-analysis charts >}}

## Results

These results were generated on June 13, 2026, from that week's OpenRouter usage.

Price matters, but it is far from the whole story. Across the current OpenRouter sample, `log(effective price)` and `log(usage)` have a correlation of about **-0.29**. The log-log price elasticity is about **-0.56**, meaning a 10% higher effective price is associated with roughly 5.6% lower usage. Price alone explains about **8%** of log usage variance.

The price bands are useful context. Models below $0.50 per million effective tokens account for about 70% of paid token volume, but the $2-$10 band still accounts for about one fifth because high-end models like Claude Sonnet and Opus get substantial use despite much higher prices. The very expensive $10+ band has less than 1% token share.

Raw benchmark correlations are stronger than the price baseline. The agentic capability index has a raw score-to-log-usage correlation of about **0.60**; the full capability index is about **0.49**. Among individual benchmarks, HiL-Bench, GSO-Bench, Terminal-Bench Hard, FrontierCode Main, and Humanity's Last Exam are near the top.

The more important result is the price-adjusted one. Adding the capability index to a price-only usage model adds about **53 percentage points** of R^2 on its matched model set. The agentic index adds about **56 percentage points**, which makes sense when you consider that OpenClaw deployments are one of OpenRouter's most popular use-cases. GSO-Bench and Terminal-Bench Hard also add large amounts of explanatory power after price. One caveat worth keeping in mind: each metric is scored on a different set of models, with different price baselines, so these increments are best read as "does score help here," not as a precise head-to-head ranking.

Not every benchmark looks predictive. SlopCodeBench, Kaggle Game Arena, SpatialBench, DeepSWE, and Blueprint-Bench 2 have raw score-to-usage correlations near zero and add essentially no explanatory power after price. That doesn't make them bad benchmarks—they may be measuring something real that OpenRouter's particular workloads simply don't reward—but they don't track the majority of usage here.

## Robustness and Out-of-Sample Tests

The headline numbers above come from a single week, which raises two fair objections: that the relationship is a one-week fluke, and that it is fit in-sample, so a search over benchmarks could be rewarding noise. To check both, I pulled each matched model's daily usage from OpenRouter's per-model activity endpoint and built a four-week panel. (The daily endpoint caps at about a month of history; the longer 52-week rankings chart only covers the few top models, too few of which carry benchmark scores to fit a cross-section, so four weeks is the honest ceiling.) Each week uses that week's own token mix to recompute effective price, so price is not held fixed.

The relationship barely moves week to week. Across the four most recent weeks (about 45 matched models each), the raw capability-to-log-usage correlation stays at **+0.47** (range +0.45 to +0.49), the partial correlation after price at **+0.69** (+0.68 to +0.72), the price elasticity at **-1.16** (-1.03 to -1.20), and the capability coefficient at **+0.044** (+0.041 to +0.045). Adding capability to the price-only model raises R^2 by about **42 percentage points** each week—a little below the single-week headline of 53, mostly because this panel corrects the cache-token accounting and uses a slightly different usage measure, but the same story. The stable **-1.16** elasticity is also reassuring: it matches the capability-controlled coefficient the calculator uses, so the gap from the simple **-0.56** is the difference between the frontier-model panel and the full sample, not week-to-week instability.

The out-of-sample tests matter more. In a walk-forward setup—fit the demand model on one week, then predict the *next* week's token shares for the models it was fit on—price alone gives an out-of-sample R^2 of about **0.10**, while price plus capability gives about **0.54**. A stricter test holds out *models* instead of weeks: in 5-fold cross-validation on the current week, price alone has an out-of-sample R^2 of roughly **0.00** (no better than guessing the mean), while price plus the capability index reaches about **0.47**, and price plus the agentic index about **0.42**. So for models the fit has never seen, price predicts essentially nothing, but capability predicts close to half the variance in log usage—the opposite of what overfitting to a noisy snapshot would produce.

These checks have their own limits: four weeks is not many, consecutive weeks are highly correlated (so the tight ranges overstate precision), and unit prices are current even though each week's token mix is historical. But the core result—capability tracks usage beyond price, stably and out-of-sample—survives them.

## Pricing Implications

This matters for both sides of the market. Users want to know which model is the best deal for their workload. Labs want to know whether a price cut would plausibly buy share, or whether a capability improvement would support a premium.

I made an interactive pricing calculator that uses the price-and-capability model directly. Select a model, then adjust its input price, output price, input/output token ratio, effective price, paid-token share, or capability index.

Input price, output price, and token mix define the effective price:

```text
effective_price =
  (input_price * input_output_ratio + output_price)
  / (input_output_ratio + 1)
```

Changing the input price, output price, or input/output ratio updates effective price. Changing effective price scales input and output prices together while preserving their current ratio.

Paid-token share and capability are linked through the capability-adjusted demand model:

```text
log(paid_token_share) =
  anchored_intercept
  + price_coefficient * log(effective_price)
  + capability_coefficient * capability_index
```

The important detail is the intercept. For each selected model, I recompute the intercept so the curve passes exactly through that model's observed paid OpenRouter token share, effective price, and capability index.

Changing paid-token share therefore solves for the effective price implied by the anchored curve. Changing capability keeps the current effective price fixed and recomputes the expected paid-token share. The chart shows the estimated paid-token-share curve over effective price at the current capability level.

{{< openrouter-benchmark-analysis pricing >}}

## Limitations

This is a correlational study on observational data, and there are several reasons to read it as suggestive rather than definitive.

**OpenRouter is a proxy, not the market.** The biggest consumers of LLM tokens—ChatGPT, Claude.ai, the Gemini app, and direct first-party API customers—mostly bypass OpenRouter. Its traffic skews toward open-weight models, coding agents, and price-sensitive developers. So "usage" here is the revealed preference of OpenRouter's particular userbase, not the whole industry.

**Usage is tokens, not decisions.** Token volume is dominated by a handful of high-throughput integrations. A coding tool defaulting to one model can produce billions of tokens, so the signal is throughput-weighted rather than a count of users or independent choices.

**Correlation is not preference.** Higher-scoring models are also newer, more heavily marketed, and more likely to be set as defaults in popular tools—all of which raise usage independent of any benchmark. The agentic result is partly circular: OpenRouter usage is dominated by coding agents, and the agentic benchmarks measure coding ability, so the two partly measure the same thing on the same popular models. My capability index is also an aggregate of the individual benchmarks it is ranked against, so it is not an independent competitor in that comparison.

**Small, uneven samples.** Each benchmark is matched on a different subset of models—from over 100 down to roughly 10—and the headline price statistics cover a larger population (a few hundred paid models) than the benchmark correlations (a few dozen frontier models). With so few models and a search over many benchmarks, some will look predictive by chance, and I report no confidence intervals or significance tests. Treat the rankings as directional.

**A short window.** The headline figures come from a single weekly snapshot (June 13, 2026); re-running the scrape on a later week would shift them, and the charts reflect whichever snapshot was last published rather than live data. The robustness panel spans only four weeks because the per-model daily history caps at about a month. Those four weeks are also highly correlated, so the stability they show is reassuring but not the same as four independent samples. A longer panel with model fixed effects would be the stronger design.

**The demand model is cross-sectional.** The pricing calculator fits a single cross-section of a few dozen models and lets you extrapolate, but "a price cut would buy this much share" is a causal claim this data can't support. Note also that the capability-controlled price coefficient (about -1.2) is much steeper than the simple price elasticity (-0.56), because cheap models also tend to be lower-capability; the calculator uses the former.
