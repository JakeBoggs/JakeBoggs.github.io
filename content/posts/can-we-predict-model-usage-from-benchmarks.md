---
title: "Do Benchmarks Predict Which Models People Use?"
date: 2026-06-13
draft: false
summary: "Evaluating LLM benchmarks by analyzing how they correlate with OpenRouter market share."
---

The purpose of a benchmark is to help answer two questions: For users, **"which model is best for my use-case?"** and for labs, **"which checkpoint should we release and how should we price it?"**

One way to test whether a benchmark is useful is to ask whether it predicts revealed preference. When two models are available at roughly the same price, does the model with the higher benchmark score get used more? If the answer is usually yes, the benchmark is probably measuring something users value. If the answer is no, either the benchmark is missing the real capability, or other factors are dominating.

## Method

I pull input and output pricing from OpenRouter's [public model catalog](https://openrouter.ai/docs/api/api-reference/models/get-models) and I scrape token usage (over the past week) from their public rankings endpoints. This is combined with my existing benchmark data, including the aggregate capability index, Epoch ECI, category sub-indices, and individual benchmark scores.

The price variable is an **effective observed price**, not just the listed input-token price or output-token price. Models have very different input/output mixes, so I estimate dollars per million served tokens from the observed token mix in OpenRouter's ranking rows:

```text
estimated_cost =
  prompt_tokens * prompt_price
  + completion_tokens * completion_price
  + cached_tokens * cache_read_price

served_tokens =
  prompt_tokens + completion_tokens + reasoning_tokens

effective_price_per_million =
  estimated_cost / served_tokens * 1,000,000
```

If OpenRouter does not expose a cache-read price, I fall back to the prompt-token price for cached tokens. This effective price is still approximate, but it is much closer to the user's actual economic tradeoff than comparing models by input price alone.

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

Finally, for each benchmark, I also look at every pair of matched models whose effective prices are within 2x of each other. A benchmark gets a correct prediction when the model with the higher benchmark score also has higher OpenRouter token usage. This gives a simple price-controlled pairwise accuracy:

```text
pairwise_accuracy = correct same-price-ish model orderings / all same-price-ish model orderings
```

So the analysis starts with the obvious confounder, price; then asks whether scores correlate with usage at all; and finally asks whether scores add explanatory power beyond price.

{{< openrouter-benchmark-analysis charts >}}

## Results

Price matters, but it is far from the whole story. Across the current OpenRouter weekly sample, `log(effective price)` and `log(usage)` have a correlation of about **-0.29**. The log-log price elasticity is about **-0.56**, meaning a 10% higher effective price is associated with roughly 5.6% lower usage in this cross-sectional cut. Price alone explains about **8%** of log usage variance.

The price bands are useful context. Models below $0.50 per million effective tokens account for about 70% of paid weekly token volume, but the $2-$10 band still accounts for about one fifth because high-end models like Claude Sonnet and Opus get substantial use despite much higher prices. The very expensive $10+ band has less than 1% token share.

Raw benchmark correlations are stronger than the price baseline. The agentic capability index has a raw score-to-log-usage correlation of about **0.60**; the all-up capability index is about **0.49**. Among individual benchmarks, HiL-Bench, GSO-Bench, Terminal-Bench Hard, FrontierCode Main, and Humanity's Last Exam are near the top of the raw score-to-usage table in this scrape.

The more important result is the price-adjusted one. Adding the all-up capability index to a price-only usage model adds about **53 percentage points** of R^2 on its matched model set. The agentic index adds about **56 percentage points**. GSO-Bench and Terminal-Bench Hard also add large amounts of explanatory power after price. This is the signal I care about most: benchmark score is not just rediscovering that cheap models get used.

Pairwise accuracy is a different lens on the same question. In the weekly scrape, my all-up capability index gets about **75%** of price-matched model pairs right. The agentic sub-index does better, at about **79%**, which fits the current OpenRouter user base: a lot of usage appears to come from coding, automation, and agent-style workflows rather than generic chat.

Several individual benchmarks look surprisingly predictive by pairwise accuracy. FrontierSWE, GSO-Bench, EQ-Bench 3, ProgramBench Extended, Terminal-Bench Hard, CritPt, ProofBench, GDPval-AA, and PPBench all clear 74% pairwise accuracy in this scrape. I would not overinterpret the exact ordering because coverage differs a lot by benchmark, but the broad pattern is useful: benchmarks with open-ended or work-like tasks tend to predict usage better than narrow one-off tasks. This makes sense when you consider that OpenClaw deployments are one of OpenRouter's most popular use-cases.

The weak end is also informative. SlopCodeBench, FrontierCode Main, Blueprint-Bench 2, Kaggle Game Arena, DeepSWE, and HalluHard all come in near chance or below. That does not mean they are bad benchmarks in isolation. It means they are not currently explaining OpenRouter paid usage at a given price very well.

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
