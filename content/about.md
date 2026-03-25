---
title: "About me"
---
My name is Jake and I have a snake. Previously at [Endeavor](https://endeavor.ai/), and before that I studied computer science and math at NC State, before leaving my senior year to move to San Francisco. I post here occassionally to share thoughts and projects.

### Research Interests
**Training objectives & evals.** Specifically, designing metrics and datasets that match an intended use-case as closely as possible. To give a concrete example, suppose you are training an LSTM to trade a stock given its last 30 days of price data. A naive method would be to minimize the MSE of the next day's closing price and use this prediction to decide whether to go long or short. If you try this, you might get excited as you see the validation loss drop sharply. However, trading with this model is unlikely to be successful due to your objective being poorly formulated. Stocks typically move little from day to day, so you can easily reduce your error to within a couple percent, but this isn't enough to gain a meaningful signal. You need to optimize for something with a stronger correlation to your profitability.

As another example, in my previous role we encountered many documents that used handwritten diagrams or annotations to convey meaning (like a drawing of a pipe with dimensions on it). There are many "correct" ways you could interpret these diagrams as text ("10 ft steel pipe" and "pipe: 10 feet, material: steel" are semantically equivalent), so a good benchmark must measure how well a model understands the intent, rather than the distance from a ground-truth.

**Harnesses.** Models are only useful if you can apply them. Frontier LLMs today have the knowledge and skills to perform many valuable tasks, but are constrained by their ability to interact with other systems. There's lots of low-hanging fruit here.

A few of my favorites:
* [RuneBench](https://maxbittker.github.io/runebench/). AI plays RuneScape
* [Remote Labor Index](https://www.remotelabor.ai/). Another benchmark, also has a harness for long-horizon digital work tasks. Still very unsaturated, no model gets above 5%
* [Kosmos](https://edisonscientific.com/articles/announcing-kosmos). This one has caught issues with some of my experiments before I ran them

**My current pet project:** engineering a real-life <a href="/about#lotus">Black Lotus</a>. There are two primary motivations here:
* First, it's a way for me to improve my understanding of genetics / how AI can be applied to the field (now I have an excuse to train some novel DNA sequence architectures)
* It's cool and I want one

My initial approach combines a couple of genes to boost anthocyanin production in the petals and I'm paying a lab to carry out the experiment. I'll post more about this as results arrive over the summer.

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