---
title: "Benchmark Dashboard"
date: 2026-05-18
draft: false
summary: "My own version of the Epoch Capabilities Index for tracking niche benchmarks and my takes on how to identify ones worth paying attention to."
---

I found myself refreshing ~40 different leaderboards whenever a new model dropped, so I made a dashboard that scrapes my favorites and computes a composite score using the same [methodology](https://epoch.ai/data/eci-documentation/methodology) as Epoch AI's [ECI](https://epoch.ai/eci). For those who are unfamiliar, Epoch uses multiple benchmarks to estimate a single "capability" level per model (kind of like an IQ score). Their index is great, but it doesn't track a lot of the benchmarks I care about, especially those which are unbounded like [EQ-Bench](https://eqbench.com/index.html). My [follow-up analysis](/posts/can-we-predict-model-usage-from-benchmarks) also shows my index to be more predictive of actual usage.

{{< benchmark-dashboard-controls >}}

{{< benchmark-dashboard-ranking >}}

The index is fixed so that GPT-4.1 will always be at 100 and GPT-5 will always be at 150. The other values might drift as the benchmark composition changes. Epoch's model requires scores to range from 0-1, so I apply normalization to Elo-based evals such that the top model scores 100% and each other model x's score equals `2 * P(model_x wins against top_model)`. To reduce noise from models with low benchmark coverage, I require that each model be on at least eight benchmarks and at least 1/3rd of the benchmarks that have data on or before that model's release date. For sub-indices, I apply the same 1/3rd rule to the category but reduce the lower bound to four across the full dataset.

Expect that I will add / remove benchmarks and adjust the formulas over time. The main objective is to strongly differentiate between the latest models, and less to track long-term progress. As a result, scores for some older models are likely to be missing or unstable.

How did I decide what to include? I mainly look at three things: data quality, contamination, and real-world relevance. Data quality is perhaps the biggest peristent issue across benchmarks and can be broken down into sub-categories like incorrect answer keys, faulty automated graders, and impossible tasks. The best way to prevent quality issues is simple but boring: look at the data. Ideally you will have a human, or multiple, attempt each task in your benchmark. If this is impractical, you better have a good automated pipeline and do manual validation of that. When I evaluate a benchmark, I either look at the data myself, or I judge whether the authors or someone with domain knowledge have done this thoroughly.

Relevance is a close runner-up for the title of biggest benchmark issue. Many benchmarks make grand claims about measuring some ability, but then quickly saturate despite models still clearly lacking in that area. Sometimes this is due to the authors making overly broad generalizations with the underlying tasks still measuring something useful, while in other cases the tasks are poor imitations of real use-cases and success does not translate. Unbounded tasks (Elo or profit based as examples) or aggregation methods like Epoch's can resolve the saturation problem, while transferability requires authors to ask "what would this look like in production" and design their tasks around this. I generally filter for benchmarks that make specific, detailed claims and have sensible orderings. Newer models in a family are almost always better; occasionally there are outliers but if a benchmark consistently disagrees, it is suspicious. Labs all have extensive internal evals to ensure they are not regressing on capabilities and they rarely miss the mark.

Memorization was historically a large problem, since any public dataset quickly gets scraped and included in the pretraining mix. Labs attempt to filter out answers for benchmarks they care about, but this is imperfect and there's no way to know exactly what is and isn't contaminated. Fortunately, memorization is becoming less of an issue with newer benchmarks (though still something to consider), as we've evolved from simple question answer pairs to "complete this long-horizon workflow." Full trajectories are often not published and memorizing perfectly across millions of tokens is much more difficult. When I look at strictly knowledge-based benchmarks, I prefer those where the answers cannot be easily found through Google and require multiple reasoning hops to arrive at.

{{< benchmark-dashboard-metr >}}

For fun, I've calculated the correlation between the indices and the [METR Time Horizon](https://metr.org/time-horizons/), with the coding sub-index having a particularly strong r^2 of 0.88. METR is extremely thorough and produced a great benchmark, but I've excluded it from my index for two reasons. First, it has mostly been saturated at the time of writing and second, I wanted to see how well I could predict their results for new models.

{{< benchmark-dashboard-eci >}}

Of course we need to compare against the original ECI too. As expected, the correlation is high, with an r^2 of 0.92. An interesting pattern I notice is that my index consistently shows Anthropic's models doing better relative to Epoch's, while theirs shows Gemini outperforming. This aligns with my personal experience, where Gemini often does well on common benchmarks but underperforms for actual use-cases.

{{< benchmark-dashboard-frontier >}}

I've also made a tracker for the US-China gap, which I estimate currently stands at ~5 months. This is slightly longer than [Epoch's calculation](https://epoch.ai/data-insights/open-closed-eci-gap).

{{< benchmark-dashboard-frontier-delta >}}

Finally, you can compare benchmark results across models with this card generator:

{{< benchmark-dashboard-model-cards >}}
