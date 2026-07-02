---
title: "HieroglyphBench: A Modern OCR Benchmark Using Ancient Text"
date: 2026-06-24
draft: false
summary: "Standard OCR benchmarks are mostly saturated, so I made something harder. HieroglyphBench tests how well VLMs can read ancient Egyptian hieroglyphs."
---

Today I'm releasing HieroglyphBench, a challenging benchmark that tests how well VLMs can transcribe ancient Egyptian hieroglyphs. You can check out the [dataset](https://huggingface.co/datasets/jakeboggs/HieroglyphBench) on Huggingface and the [source code](https://github.com/JakeBoggs/HieroglyphBench) on Github.

Given a column of hieroglyphs, models must output the signs they see as [Gardiner sign-list](https://en.wikipedia.org/wiki/Gardiner%27s_sign_list) codes, in reading order. A Gardiner code names a sign's type regardless of which way it faces: the owl is **`G17`**, the mouth is **`D21`**, and the seated man is **`A1`**. The results are scored using edit distance against ground-truth transcriptions.

<script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>

<div style="width: 90%; margin: 20px auto;">
    <canvas id="hieroLeaderboardChart"></canvas>
</div>

<script>
    document.addEventListener('DOMContentLoaded', () => {
        const canvasElement = document.getElementById('hieroLeaderboardChart');
        const ctx = canvasElement.getContext('2d');

        const labels = [
            'Gemini 3.5 Flash',           // 52.5
            'Gemini 3 Flash Preview',     // 48.8
            'Gemini 3.1 Pro',             // 39.7
            'Fable 5',                    // 22.9
            'GPT-5.5',                    // 22.8
            'Kimi K2.6',                  // 19.2
            'Gemini 2.5 Pro',             // 18.8
            'GPT-5.4 Mini',               // 15.7
            'Claude Opus 4.8',            // 14.0
            'MiniMax M3',                 // 11.1
            'Qwen 3.7 Plus',              // 10.1
            'GPT-4o',                     // 9.5
        ];
        const dataValues = [52.5, 48.8, 39.7, 22.9, 22.8, 19.2, 18.8, 15.7, 14.0, 11.1, 10.1, 9.5];
        const backgroundColors = [
            'rgba(0, 200, 150, 0.85)',  // Gemini 3.5 Flash
            'rgba(0, 150, 255, 0.85)',  // Gemini 3 Flash Preview
            'rgba(0, 220, 220, 0.85)',  // Gemini 3.1 Pro
            'rgba(235, 150, 110, 0.85)',// Fable 5
            'rgba(0, 114, 178, 0.85)',  // GPT-5.5
            'rgba(140, 190, 70, 0.85)', // Kimi K2.6
            'rgba(90, 175, 190, 0.85)', // Gemini 2.5 Pro
            'rgba(204, 121, 167, 0.85)',// GPT-5.4 Mini
            'rgba(210, 120, 80, 0.85)', // Claude Opus 4.8
            'rgba(240, 170, 50, 0.85)', // MiniMax M3
            'rgba(120, 90, 60, 0.85)',  // Qwen 3.7 Plus
            'rgba(153, 102, 255, 0.8)', // GPT-4o
        ];
        const borderColors = backgroundColors.map(c => c.replace('0.8', '1').replace('0.85', '1'));

        canvasElement.style.height = '480px';

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Sign accuracy',
                    data: dataValues,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'x',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Leaderboard', font: { size: 18 } },
                    tooltip: {
                        callbacks: {
                            label: (c) => `Sign accuracy: ${c.parsed.y}%`
                        }
                    }
                },
                scales: {
                    y: { beginAtZero: true, max: 100, title: { display: true, text: 'Sign accuracy (%)' } },
                    x: { ticks: { autoSkip: false, maxRotation: 90, minRotation: 45 } }
                }
            }
        });
    });
</script>

Every reasoning-capable model is run at medium effort. The three open-weight models, Kimi K2.6, MiniMax M3, and Qwen 3.7 Plus, use their official providers to avoid third-party hosting issues.

Many visual understanding benchmarks are nearing saturation and struggle to differentiate between the latest frontier models, despite substantial differences in their capabilities. HieroglyphBench exposes this gap, with the best models (all of which are from the Gemini family) barely scoring 50%. Models from the other labs struggle even more and most do not reach 20%.

## Examples

<style>
.hiero-ex { border:1px solid #e2ddd2; border-radius:12px; padding:16px; margin:22px 0; background:#fcfbf8; }
.hiero-grid { display:flex; gap:18px; align-items:flex-start; }
.hiero-img { flex:0 0 auto; max-height:360px; width:auto; border-radius:8px; border:1px solid #d8d2c4; background:#fff; }
.hiero-readings { flex:1 1 auto; min-width:0; }
.hiero-readings .rd { margin-bottom:14px; }
.hiero-readings .lbl { display:block; font-size:13px; font-weight:600; color:#5b5446; margin-bottom:5px; }
.hiero-readings .signs { display:block; max-width:100%; height:auto; margin:2px 0 5px; }
.hiero-readings .codes { font-family:ui-monospace,Consolas,"SFMono-Regular",monospace; font-size:11.5px; line-height:1.5; color:#9a9384; word-break:break-word; }
.hiero-readings .acc { font-family:ui-monospace,monospace; font-size:12px; padding:1px 7px; border-radius:999px; color:#fff; vertical-align:middle; }
.hiero-readings .acc.good { background:#2e9e44; } .hiero-readings .acc.mid { background:#d08a1e; } .hiero-readings .acc.bad { background:#c8432f; }
.hiero-ex figcaption { margin-top:10px; font-size:13.5px; color:#6b6457; font-style:italic; }
@media (max-width:640px){ .hiero-grid{ flex-direction:column; } .hiero-img{ max-height:300px; align-self:center; } }
</style>

<figure class="hiero-ex">
  <div class="hiero-grid">
    <img class="hiero-img" src="/images/hierobench/unas-p20-c3-2.png" alt="Photographed column of hieroglyphs, Pyramid of Unas plate 20 (col 3)">
    <div class="hiero-readings">
      <div class="rd">
        <span class="lbl">Ground truth</span>
        <img class="signs" src="/images/hierobench/unas-p20-c3-2_gt.png" alt="Ground-truth signs rendered as hieroglyphs">
        <div class="codes">D21 G17 O4 D21 G43 N5 Q3 N35 G17 M17 D4 E23</div>
      </div>
      <div class="rd">
        <span class="lbl">Gemini 3.5 Flash <span class="acc good">83%</span></span>
        <img class="signs" src="/images/hierobench/unas-p20-c3-2_g35flash.png" alt="Gemini 3.5 Flash predicted signs as hieroglyphs">
        <div class="codes">D21 G17 O4 D21 G43 N5 N35 G14 M17 D4 E23</div>
      </div>
      <div class="rd">
        <span class="lbl">GPT-5.5 <span class="acc mid">42%</span></span>
        <img class="signs" src="/images/hierobench/unas-p20-c3-2_gpt55.png" alt="GPT-5.5 predicted signs as hieroglyphs">
        <div class="codes">D21 G1 O4 D21 G1 N5 U6 N35 G1 S29 D21 E34</div>
      </div>
      <div class="rd">
        <span class="lbl">Claude Opus 4.8 <span class="acc bad">17%</span></span>
        <img class="signs" src="/images/hierobench/unas-p20-c3-2_opus48.png" alt="Claude Opus 4.8 predicted signs as hieroglyphs">
        <div class="codes">N1 G17 O4 N37 G17 N5 X1 N35 G29 S29 D58 G43 N35 T22</div>
      </div>
    </div>
  </div>
</figure>

<figure class="hiero-ex">
  <div class="hiero-grid">
    <img class="hiero-img" src="/images/hierobench/unas-p21-c11-1.png" alt="Photographed column of hieroglyphs, Pyramid of Unas plate 21 (col 11)">
    <div class="hiero-readings">
      <div class="rd">
        <span class="lbl">Ground truth</span>
        <img class="signs" src="/images/hierobench/unas-p21-c11-1_gt.png" alt="Ground-truth signs rendered as hieroglyphs">
        <div class="codes">N31 N35 V28 D21 X1 N31 E34 N35 M17 S29 Q3 M17 D4 G43 F13 Q3 X1 N35 N37 N35</div>
      </div>
      <div class="rd">
        <span class="lbl">Gemini 3.1 Pro <span class="acc good">60%</span></span>
        <img class="signs" src="/images/hierobench/unas-p21-c11-1_g31pro.png" alt="Gemini 3.1 Pro predicted signs as hieroglyphs">
        <div class="codes">U33 N35 S34 D21 X1 U33 I3 E34 N35 M17 M17 M17 Q3 D4 G43 F31 Q3 X1 N35 N37 N35</div>
      </div>
      <div class="rd">
        <span class="lbl">Kimi K2.6 <span class="acc bad">20%</span></span>
        <img class="signs" src="/images/hierobench/unas-p21-c11-1_kimi.png" alt="Kimi K2.6 predicted signs as hieroglyphs">
        <div class="codes">O36 F40 D21 X1 I9 G5 N35 Z1 Z1 M17 O29 D21 G39 I9 D36 X8 N35 O29</div>
      </div>
      <div class="rd">
        <span class="lbl">Qwen 3.7 Plus <span class="acc bad">20%</span></span>
        <img class="signs" src="/images/hierobench/unas-p21-c11-1_qwen.png" alt="Qwen 3.7 Plus predicted signs as hieroglyphs">
        <div class="codes">O29 Z1 D21 Q3 M17 L2 D21 N5 G1 D54 F21 G17 N35 Z1 N35</div>
      </div>
    </div>
  </div>
</figure>

<figure class="hiero-ex">
  <div class="hiero-grid">
    <img class="hiero-img" src="/images/hierobench/unas-p7-c12-2.png" alt="Photographed column of hieroglyphs, Pyramid of Unas plate 7 (col 12)">
    <div class="hiero-readings">
      <div class="rd">
        <span class="lbl">Ground truth</span>
        <img class="signs" src="/images/hierobench/unas-p7-c12-2_gt.png" alt="Ground-truth signs rendered as hieroglyphs">
        <div class="codes">D46 V4 N14 S29 N35 V13 G43 G17 D21 N35 V31 G43 Q3 N35</div>
      </div>
      <div class="rd">
        <span class="lbl">Gemini 3 Flash Preview <span class="acc good">64%</span></span>
        <img class="signs" src="/images/hierobench/unas-p7-c12-2_g3flashprev.png" alt="Gemini 3 Flash Preview predicted signs as hieroglyphs">
        <div class="codes">D21 D46 V1 N14 S29 N35 V31 G43 G17 D21 N35 V30 G43 N35</div>
      </div>
      <div class="rd">
        <span class="lbl">GPT-5.4 Mini <span class="acc bad">21%</span></span>
        <img class="signs" src="/images/hierobench/unas-p7-c12-2_gpt54mini.png" alt="GPT-5.4 Mini predicted signs as hieroglyphs">
        <div class="codes">O1 Z9 Z1 N35 G43 G43</div>
      </div>
      <div class="rd">
        <span class="lbl">MiniMax M3 <span class="acc bad">7%</span></span>
        <img class="signs" src="/images/hierobench/unas-p7-c12-2_minimax.png" alt="MiniMax M3 predicted signs as hieroglyphs">
        <div class="codes">Aa2 N14 Z4 V31 T8 G5 Z4 G17 O11</div>
      </div>
    </div>
  </div>
</figure>

The top models track the column sign by sign and slip mostly on signs that look alike: one bird for another, a reed for a forearm. Lower-scoring models catch an occasional sign but lose the order or fill the gaps with noise.

## Dataset construction

The source is the **Pyramid of Unas dataset** ([Morris Franken's GlyphDataset](https://github.com/morrisfranken/glyphreader)), built from Alexandre Piankoff's 1955 photographic plates of the Pyramid Texts in the tomb of the pharaoh Unas during the late 5th Dynasty, around 2350 BCE.

From this, I build column-level inscription items by detecting and splitting columns into chunks. Each chunk is cropped from the original plate photograph using the bounding box of its signs plus padding, and the ground truth is the ordered list of Gardiner codes in that crop. The result was ~200 images, which I then skimmed through manually to find 30 examples for the final dataset. This is enough to keep the noise low (re-running doesn't move the scores more than ~1%) while also staying cheap enough for me to update the leaderboard as new models are released.

Gathering the data was by far the most time-consuming of this project. I tried scraping a bunch of different sources, but they were all too low quality. I was hoping for some more diversity from different monuments, but I think the current version is good enough for now. If you're reading this and you know of another high-quality source I could add, please reach out and maybe I'll be able to add it into a v2!

## Scoring

The model returns a JSON array of Gardiner codes, enforced with a strict JSON-schema structured output. From this, I compute two metrics:

- **Sign error rate**: the Levenshtein distance between the predicted sequence and the ground-truth, divided by the number of ground-truth signs.
- **Sign accuracy**: `1 − sign error rate`, floored at 0, so a prediction with more errors than there are signs scores 0 rather than going negative.

Codes are canonicalized before comparison (`G017` -> `G17`, `aa1` -> `Aa1`) so formatting differences don't count as errors. Both metrics are computed per inscription and then averaged across the whole eval set, weighting each inscription equally. Random guessing scores ~0, since there are over a thousand glyphs.

## Thoughts
Last December, after Gemini 3 was released, I had some fun using it to [translate medieval manuscripts](/posts/translating-manuscripts-with-gemini/). This made me wonder how far the models could be pushed, but I couldn't find any evaluations online, so it just stayed as a nagging idea in the back of my head. I recently had a bit of time to do more testing and this benchmark is the result. None of the current models are reliable enough yet to be used for serious research, but they're improving quickly and it seems plausible that they'll saturate this task within a generation or two. This is very interesting to me because while it is possible labs are training specifically for this task, it seems unlikely and I suspect this is an emergent capability. I plan to track this as new models are released and will update the leaderboard.
