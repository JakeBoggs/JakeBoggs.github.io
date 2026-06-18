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

I pull input and output pricing from OpenRouter's [public model catalog](https://openrouter.ai/docs/api/api-reference/models/get-models) and I scrape token usage (over the past week) from their public rankings endpoints. This is combined with [my existing benchmark data](/posts/benchmarks), including my own aggregate capability index, the ECI, Arena.ai rankings, and individual benchmark scores.

The price variable is an **effective observed price**, not just the listed input-token price or output-token price. Models have very different input/output mixes, so I estimate dollars per million served tokens from the observed token mix in OpenRouter's ranking rows:

```text
estimated_cost =
  (prompt_tokens - cached_tokens) * prompt_price
  + cached_tokens * cache_read_price
  + completion_tokens * completion_price

served_tokens =
  prompt_tokens + completion_tokens

effective_price_per_million =
  estimated_cost / served_tokens * 1,000,000
```

Cached tokens are a subset of the prompt tokens that were served from cache, so only the non-cached prompt is billed at the full prompt price and the cached portion at the cache-read rate. Reasoning tokens are billed at the output rate and are already counted inside `completion_tokens`, so they are not added separately. If OpenRouter does not expose a cache-read price, I fall back to the prompt-token price for cached tokens. This effective price is still approximate, but it is much closer to the user's actual economic tradeoff than comparing models by input price alone.

The analysis then runs in three passes.

First, I estimate the price baseline:

```text
log_usage = a + b * log_effective_price
```

This answers the simplest question: how much of model usage is explained by price before looking at benchmark scores at all?

{{< openrouter-benchmark-analysis price >}}

Second, I estimate raw score-to-usage relationships for each benchmark:

```text
log_usage = a + b * benchmark_score
```

This flips the previous question and answers: how much of model usage is explained by the benchmark before looking at price?

{{< openrouter-benchmark-analysis score >}}

Third, I ask whether scores explain usage after accounting for price:

```text
log_usage = a + b * log_effective_price + c * benchmark_score
```

The key statistic here is incremental R^2: how much more usage variance is explained when benchmark score is added to the price-only model.

{{< openrouter-benchmark-analysis adjusted >}}

## Results

These results were generated on June 18, 2026, from that week's OpenRouter usage.

Price matters, but it is far from the whole story. Across the current OpenRouter sample, `log(effective price)` and `log(usage)` have a correlation of about **-0.29**. The log-log price elasticity is about **-0.57**, meaning a 10% higher effective price is associated with roughly 5.7% lower usage. Price alone explains about **8%** of log usage variance.

The price bands are useful context. Models below $0.50 per million effective tokens account for about 68% of paid token volume, but the $2-$10 band still accounts for about one fifth because high-end models like Claude Sonnet and Opus get substantial use despite much higher prices. The very expensive $10+ band has less than 1% token share.

Raw benchmark correlations are stronger than the price baseline. My [capability index](/posts/benchmarks) has a raw score-to-log-usage correlation of about **0.51**. Among individual benchmarks, HiL-Bench, GSO-Bench, FrontierCode Main, scBench, and Terminal-Bench Hard are near the top.

The more important result is the price-adjusted one. Adding the capability index to a price-only usage model adds about **59 percentage points** of R^2 on its model set. GSO-Bench, Terminal-Bench Hard, Humanity's Last Exam, the Arena.ai Text Elo, Arena.ai Agent Net Improvement, and raw EQ-Bench Elo also add large amounts of explanatory power after price. The agent score adds about **48 percentage points** on its smaller overlap, with about **70%** pairwise accuracy.

Not every benchmark looks predictive. SlopCodeBench, Opus Magnum Bench, Kaggle Game Arena, FrontierSWE, and Blueprint-Bench 2 have raw score-to-usage correlations near zero or add almost no explanatory power after price. That doesn't necessarily make them bad benchmarks, but they don't explain the majority of usage here.

## Robustness and Out-of-Sample Tests

The numbers above come from a single week, which invites two objections. First, the relationship could be a one-week fluke, and second, because I check around thirty benchmarks and then report the ones that score highest, a few could look predictive just by luck.

To address both, I pulled each model's daily usage from OpenRouter's per-model activity endpoint and built a four-week panel. (The daily endpoint only goes back about a month, and the 52-week chart covers just the top few models, almost none of which have benchmark scores, so four weeks is as far as I can go.) Each week recomputes effective price from that week's own token mix, so price is not held fixed.

The relationship is stable across the four weeks, with about 45 models in each. The raw capability-to-log-usage correlation stays around **+0.47**, the price elasticity around **-1.16**, and the capability coefficient around **+0.044**. Adding capability to the price-only model raises R^2 by about **42 percentage points** each week, below the current single-week figure of 59 (this panel fixes the cache-token accounting and measures usage slightly differently). The steady **-1.16** elasticity matches the coefficient the calculator uses, so the gap from the simple **-0.57** is frontier models versus the full sample, not week-to-week instability.

The out-of-sample tests matter more. First, walk-forward: fit the model on one week, then predict the next week's token shares for those models. Price alone gives an out-of-sample R^2 around **0.10**; price plus capability gives about **0.54**. Second, a harder test that holds out models instead of weeks: in 5-fold cross-validation on the current week, price alone scores about **0.00**, no better than guessing the mean, while price plus capability reaches **0.47**. For models the fit has never seen, price on its own predicts almost nothing, but capability predicts close to half the variance.

These checks have limits. Four weeks is not much, the weeks overlap heavily so the tight ranges look more precise than they are, and prices are current even though each week's token mix is historical. Still, the main result holds: capability tracks usage beyond price, and it does so out-of-sample.

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

**Usage does not always imply preference.** Higher-scoring models are also newer, more heavily marketed, and more likely to be set as defaults in popular tools—all of which raise usage independent of any benchmark.

**A short window.** The figures come from a single weekly snapshot (June 18, 2026); re-running the scrape on a later week would shift them, and the charts reflect whichever snapshot was last published rather than live data. The robustness panel spans only four weeks because the per-model daily history caps at about a month. Those four weeks are also highly correlated, so the stability they show is reassuring but not the same as four independent samples.

**The demand model is cross-sectional.** The pricing calculator fits a single cross-section of a few dozen models and lets you extrapolate, but "a price cut would buy this much share" is a causal claim this data can't support. Note also that the capability-controlled price coefficient (about -1.2) is much steeper than the simple price elasticity (-0.56), because cheap models also tend to be lower-capability; the calculator uses the former.
