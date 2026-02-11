---
title: "About me"
---
My name is Jake and I have a snake. Currently building environments and benchmarks full-time at [Endeavor](https://endeavorai.com/). Previously studied computer science and math at NC State, before leaving my senior year to move to San Francisco. I post here occassionally to share thoughts and projects.

### Research Interests
* **Training objectives & evals.** Specifically, designing metrics and datasets to measure the ability of models to achieve outcomes. To give a concrete example, suppose you are training an LSTM to trade a stock given its last 30 days of historical price data. A naive method would be to minimize the MSE of the next day's closing price and use this prediction to decide whether to go long or short. If you try this, you might get excited as you see the test loss drop sharply. However, attempting to trade with this model is unlikely to be successful due to your objective being poorly formulated. Stocks typically move little from day to day, so you can easily reduce your error to within a couple percent, but this isn't enough to gain a meaningful signal. You need to optimize something with a stronger correlation to your profitability. As another example, in my current role we encounter many documents that use handwritten diagrams or annotations to convey meaning (like a drawing of a pipe with dimensions on it). There are many correct ways you could interpret these diagrams as text ("10 ft steel pipe" and "pipe: 10 feet, material: steel" are semantically equivalent), so a good eval must go beyond metrics like word or character error rates.
* **Data consistency.** Large Language Models are great at modeling. This might seem obvious, as "Model" is in the name, but I think people forget what this means: LLMs construct internal representations that are consistent with their training data. If your data is inconsistent with reality, your model's representation of reality will be too. Garbage in, garbage out. Reality is important, but not the only aspect of consistency you should consider. In order to reason about new data points, models must make assumptions and they are extremely sensitive to subtle implications in their training set. Some of the classic examples are ["weird generalization"](https://arxiv.org/abs/2512.09742) and ["emergent misaligment"](https://arxiv.org/abs/2502.17424), where the authors find that fine-tuning on outdated bird names makes the models behave as though they are in the 19th century or fine-tuning on insecure code causes them to act maliciously in other contexts. While unintended, these behaviors are consistent with the data. A helpful assistant would not write vulnerable code, so the model becomes unaligned. If the assistant was the in 21st century, it would not use 19th century names, so it assumes it is in the 1800s. I think this shows how current architectures are already sufficently generalizable and also how careful we need to be to ensure that our data is consistent with the behaviors we desire.
* **My current pet project:** genetically engineering a real-life <a href="/about#lotus">Black Lotus</a>. This is really just a way for me to teach myself about genetics and how AI can be applied to the space. My initial approach combines a couple of genes to boost anthocyanin production in the petals and I'm paying a lab to carry out the experiment. I'll post more about this as results arrive over the summer.

### Gallery
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 1rem; align-items: flex-start;">
    <div style="width: 300px; min-width: 250px; max-width: 90%;">
        <img src="/images/jake.png" alt="Jake" style="width: 100%; margin: 0; display: block;">
        <div style="text-align: center; font-size: 0.8rem; color: #888; margin-top: 0.25rem;">Jake</div>
    </div>
    <div style="width: 300px; min-width: 250px; max-width: 90%;">
        <img src="/images/snake.jpg" alt="Snake" style="width: 100%; margin: 0; display: block;">
        <div style="text-align: center; font-size: 0.8rem; color: #888; margin-top: 0.25rem;">Snake</div>
    </div>
</div>
<span id="crocs"></span>

<div style="margin-top: 2rem;">
    If you see me in the wild, chances are that I'll be rocking these:
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 1rem; align-items: flex-start;">
        <img src="/images/crocs.jpg" alt="Crocs" style="max-width: min(500px, 90%); margin: 0; display: block;">
    </div>
</div>
<span id="lotus"></span>

<div style="margin-top: 2rem;">
    I'm also skilled at Magic: the Gathering 😏
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 1rem; align-items: flex-start;">
        <img src="/images/lotus.jpg" alt="Black Lotus" style="max-width: min(400px, 90%); margin: 0; display: block;">
    </div>
</div>