# -*- coding: utf-8 -*-
"""Build the English illustrated PDF report (text + real curves + tables + image analysis)."""
import os, html
from weasyprint import HTML

BASE = os.path.dirname(os.path.abspath(__file__)) + "/"   # reports/  (figures/ live here)
def fig(path, cap):
    return f'<figure><img src="{path}"/><figcaption>{cap}</figcaption></figure>'
def table(headers, rows):
    h = ''.join(f'<th>{c}</th>' for c in headers)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm;
  @bottom-center { content: "Krea 2 -> Scene-Linear HDR . Implementation & Validation"; font-size: 8pt; color:#9aa0ab; }
  @bottom-right { content: counter(page) " / " counter(pages); font-size: 8pt; color:#9aa0ab; } }
* { box-sizing: border-box; }
html { font-family:'DejaVu Sans', sans-serif; color:#1b1d24; font-size:9.6pt; line-height:1.5; }
h1 { font-family:'DejaVu Serif', serif; font-size:21pt; margin:0 0 2mm; color:#141621; line-height:1.15; }
h2 { font-family:'DejaVu Serif', serif; font-size:13.5pt; margin:7mm 0 2mm; color:#8a5406; border-bottom:1.5pt solid #e6cfa8; padding-bottom:1mm; }
h3 { font-size:10.5pt; margin:4mm 0 1.5mm; color:#141621; }
p { margin:1.6mm 0; } .mono, code { font-family:'DejaVu Sans Mono', monospace; font-size:8.2pt; }
.lead { color:#4a4f5b; font-size:10pt; } .accent { color:#b06f0c; font-weight:bold; }
.good { color:#0f8a79; font-weight:bold; } .warn { color:#cf4638; font-weight:bold; }
figure { margin:3mm 0 2mm; page-break-inside:avoid; text-align:center; }
figure img { max-width:100%; border:1pt solid #e5e2db; border-radius:2pt; }
figcaption { font-size:8pt; color:#6b7280; margin-top:1mm; text-align:left; }
table { width:100%; border-collapse:collapse; margin:2.5mm 0; font-size:8.6pt; page-break-inside:avoid; }
th { background:#f4ede1; color:#5a4413; text-align:left; padding:1.4mm 2mm; border-bottom:1.2pt solid #e6cfa8; }
td { padding:1.2mm 2mm; border-bottom:.5pt solid #eee7db; vertical-align:top; }
tbody tr:nth-child(even){ background:#faf7f1; }
ul { margin:1.5mm 0; padding-left:5mm; } li { margin:.8mm 0; }
.cover { border-bottom:2.5pt solid #c1770f; padding-bottom:5mm; margin-bottom:4mm; }
.pill { display:inline-block; font-size:8pt; background:#f4ede1; color:#8a5406; padding:.6mm 2mm; border-radius:8pt; margin-right:2mm; }
.callout { background:#f7f3ea; border-left:2.5pt solid #c1770f; padding:2mm 3mm; margin:3mm 0; font-size:9pt; page-break-inside:avoid; }
.two { display:flex; gap:4mm; } .two > div { flex:1; } small { color:#6b7280; }
"""

F = 'figures/'
BODY = f"""
<div class="cover">
  <div><span class="pill">Implementation & Validation</span><span class="pill">2026-07</span><span class="pill">RTX PRO 6000 . ComfyUI v0.27</span></div>
  <h1>Krea 2 &rarr; Scene-Linear HDR<br/>true HDR assets from a small LoRA</h1>
  <p class="lead">Making Krea 2 (12.9B, Qwen Image VAE) generate true scene-linear HDR (EXR, values &gt;&gt; 1) via a small LoRA, validated end-to-end into ComfyUI. This report records what was built and measured, with real curves, tables, and image analysis.</p>
</div>

<div class="callout"><b>Five core conclusions (all measured):</b>
<ul>
<li><b>The route works</b> &mdash; "LogC4 encoding + frozen VAE + a DiT LoRA on RAW": Krea 2 learns to generate extended dynamic range.</li>
<li><b>Learned, not fabricated</b> &mdash; same prompt+seed, the base clips highlights flat (fake), the LoRA spreads them (real): pinned-at-ceiling 6.20% &rarr; 0.66%.</li>
<li><b>Generalizes</b> &mdash; 7/8 out-of-distribution prompts (fire/neon/portrait/cathedral/candle) produce real HDR with the bright region on the light source.</li>
<li><b>Runs in ComfyUI</b> &mdash; LoRA needs a key remap (0&rarr;264) + two nodes; e2e yields a real HDR EXR (459&times; / 8.7 stops).</li>
<li><b>Honest ceiling</b> &mdash; single-plane LogC4 caps peak radiance ~470&times;; sun magnitude needs a multi-plane scheme.</li>
</ul></div>

<h2>1. Method &amp; pipeline</h2>
<p>The VAE is untouched (keeping the official RAW&rarr;Turbo transfer). Scene-linear radiance is compressed into [0,1] with <span class="accent">ARRI LogC4</span> and fed to the frozen Qwen Image VAE; the model learns to generate in LogC4 space; <b>inverse-LogC4</b> recovers linear light for EXR. VAEDecode's [0,1] clamp is harmless (the output is a [0,1] LogC4 code; &gt;1 appears only after inverse-LogC4).</p>
<p class="mono">train: EXR &rarr;(median&rarr;0.18)&rarr; LogC4[0,1] &rarr; 8-bit PNG &rarr; VAE latent &rarr; DiT LoRA(RAW)<br/>infer: prompt &rarr; Krea2+LoRA &rarr; LogC4[0,1] &rarr; inverse-LogC4 &rarr; scene-linear(&gt;&gt;1) &rarr; EXR / HDR AVIF</p>
{fig(F+'figA_curves.png', '<b>Fig 1.</b> Five encoding curves mapping scene-linear to [0,1]. Camera logs (LogC4/Log3G10) place 18% gray near 0.28-0.33, close to sRGB 0.46 (on-manifold); each curve has a different ceiling (linear peak at code 1).')}

<h2>2. CPU validation (no GPU)</h2>
<p>Before any training code, the linear&harr;encoding mapping and the EXR chain were validated numerically. <b>All 8 curves invert to ~1e-13</b> (float64). Mid-gray placement: LogC4 0.278, Log3G10 0.333, PU21 0.359, PQ 0.348 (near sRGB 0.46); mu-law 0.015 and Log-Gamma 0.809 are badly placed &mdash; confirmed poor.</p>
{fig(F+'figB_precision.png', '<b>Fig 2.</b> Reconstruction error vs exposure under an 8-bit VAE proxy. Log curves stay ~2% flat until their ceiling (~17 codes/stop, no banding wall) &mdash; correcting the earlier "~85 codes -> 4-6 stops" oversimplification. The real limiter is the VAE&#39;s non-uniform precision + the sun overflowing the ceiling.')}
<div class="two"><div>
{fig(F+'figC_vae.png', '<b>Fig 3.</b> Real Qwen VAE round-trip code-PSNR on a real HDRI: LogC4 (38.2 dB) best, best bright band. X2HDR premise confirmed: perceptual encoding ~2% band error, while naive scaled-linear fed to the VAE gives <b>7211%</b> highlight error.')}
</div><div>
{fig(F+'vae_roundtrip_kloofendal_LogC4.png', '<b>Fig 4.</b> LogC4 -> real VAE -> inverse on a real HDRI: tonemapped original and reconstruction are near-identical (top/mid); error concentrates in dark textured foreground (~22%); the sun clips. Clean window is mid+high ~2%.')}
</div></div>
<p>First-hand: the Qwen VAE decoder has <b>no tanh/sigmoid</b> &mdash; its only bound is <code>torch.clamp(-1,1)</code>, so the bottleneck is the encoder, not the decoder. Real data: Poly Haven <code>kloofendal</code> max 111,253&times;, 31.6 stops, median 0.178; S2R-HDR sample peaks at only 21&times; with no sun (a hygiene source, not an IBL/sun backbone). EXR chain: 32-bit ZIP preserves 4000&times; exactly; half overflows above 65504 -&gt; +Inf; clamp-&gt;[0,1] and implicit sRGB destroy HDR (4000 -&gt; 1.0 / 33.4).</p>
{fig(F+'figF_hlg_vs_logc4.png', '<b>Fig 5.</b> HLG vs LogC4. HLG cannot place mid-gray on-manifold AND cover 11 stops at once: at a usable mid-gray its ceiling is only 12x (bright highlights clip); at LogC4&#39;s 470x ceiling it crushes mid-gray to ~0.03 (off-manifold). LogC4 uniquely does both.')}

<h2>3. Dataset</h2>
<p>40 CC0 Poly Haven HDRIs re-projected into <b>normal perspective scene images</b> (not panoramas); 320 crops, 229 HDR-rich; LogC4-encoded 8-bit PNG + caption + manifest (reproducible float crops).</p>
{fig(F+'hdr_content_proof.png', '<b>Fig 6.</b> Per-image HDR-content check: same PNG read naively as SDR -&gt; peak 1.0 (flat/grey); read correctly via inverse-LogC4 -&gt; peak 470x (sun and gradation present). 93% of training images decode to real HDR.')}
<div class="callout"><b>A real dataset flaw:</b> 50% of "with visible sun" captions sit on crops with no sun in frame; parking-lot content and the sun ended up in separate crops, so the model can't compose "parking lot + sun". Fix: honest per-crop captioning (only write "sun" when the crop's measured peak is high) + wider FOV so sun and scene co-occur.</div>

<h2>4. Training &amp; pipeline audit</h2>
<p>musubi-tuner <code>krea2_train_network.py</code>: RAW + <code>lora_krea2</code> dim/alpha 32 (264 modules), lr 1e-4, adamw8bit, 10 epochs / 2290 steps. <b>Training VRAM ~34 GB</b> (bf16 DiT + adamw8bit + gradient checkpointing + batch 2 @ 1024&sup2;); fits 24GB with fp8 + block swap.</p>
{table(['Framework','Image-&gt;VAE normalization','Color aug','HDR-safe?'],
  [['<b>musubi</b>','<span class="mono">/127.5-1</span> (validated [0,1]-&gt;[-1,1])','none','<span class="good">safe (recommended)</span>'],
  ['ai-toolkit','<span class="mono">image*2-1</span> correct','ships ColorJitter','<span class="warn">disable color aug</span> (else corrupts LogC4)']])}
<p><small>Rule: never apply a color op to LogC4 pixels. Under this route the trainer needs no HDR-awareness &mdash; the HDR lives in the encoding.</small></p>

<h2>5. Results</h2>
<h3>5.1 Real vs fabricated (core evidence)</h3>
<p>">1 alone isn't proof &mdash; the decode curve inflates any image. The test is highlight clipping: the base clips highlights flat at the ceiling (fake HDR); the LoRA spreads them into graduated values (real HDR).</p>
{fig(F+'figD_basevslora.png', '<b>Fig 7.</b> Same prompt+seed, base vs LoRA. Both metrics (pinned-at-ceiling / code saturation) are far lower for the LoRA -&gt; highlights keep real gradation.')}
{fig(F+'base_vs_lora_window.png', '<b>Fig 8.</b> Window close-up: the base blows the window to a flat ceiling slab (6.20% pinned, a histogram spike at the ceiling); the LoRA holds graduation (0.66%, a smooth highlight distribution). Direct evidence the backbone learned extra information.')}

<h3>5.2 Generalization (epoch 10, out of distribution)</h3>
<p>With only the HDR trigger phrase, on community prompts (portrait/fireplace/neon/dragon/cathedral/candle). <b>7/8 produce real HDR</b> (&le;0.01% pinned, light on the source). The candle that failed at epoch 8 is fixed at 10 (flame 400x). The lone exception (spaceship) is not a black frame &mdash; a dim deep-space starfield with no bright source (content, but no extended range).</p>
{fig(F+'figE_ood.png', '<b>Fig 9.</b> Decoded peak radiance per OOD scene (log axis). Green = has a bright source -&gt; real HDR (7); grey = dark scene, no source (spaceship starfield).')}
{fig(F+'ood_grid.png', '<b>Fig 10.</b> OOD generation (per-image auto-exposure so every scene is visible): fire/neon/stained-glass/lighthouse highlights land on the correct source; the candle is a lit candle (400x); the spaceship is a dim starfield.')}

<h3>5.3 End-to-end in ComfyUI</h3>
{fig(F+'comfyui_e2e_proof.png', '<b>Fig 11.</b> Full ComfyUI graph output: sunset over ocean, sun at the horizon as the bright region, peak 459x / 8.7 stops &mdash; quality on par with musubi.')}

<h3>5.4 Color check &amp; display fix</h3>
<div class="two"><div>
{fig(F+'color_distribution.png', '<b>Fig 12.</b> Highlights are neutral, not blue-tinted (lighthouse lamp R:G:B = 0.96:1.00:0.74).')}
</div><div>
{fig(F+'tonemap_fix.png', '<b>Fig 13.</b> The "blue overflow" was display-side: per-channel hard clip (left) -&gt; hue-preserving roll-off + highlight desaturation (right). The EXR is unaffected.')}
</div></div>

<h2>6. ComfyUI landing (end-to-end verified)</h2>
{table(['Question','Answer'],
  [['musubi LoRA loads directly?','<span class="warn">No</span> &mdash; ComfyUI&#39;s own loader matches 0/264 keys (naming differs)'],
  ['After remap?','<span class="good">Yes</span> &mdash; remap_lora_for_comfyui.py maps 264/264; ComfyUI attaches 263 patches'],
  ['Built-in SaveImage for EXR?','<span class="warn">No</span> &mdash; 8-bit + clamp, HDR lost'],
  ['Fix','Two nodes: <b>LogC4-&gt;Linear</b> + <b>Save EXR Scene-Linear</b> (also a LogC4 option contributed to built-in SaveImageAdvanced)'],
  ['End-to-end','<span class="good">Yes</span> &mdash; real EXR peak 459x / 8.7 stops'],
  ['Gotcha','<span class="warn">Do not add ModelSamplingSD3</span> &mdash; Krea2 already shifts; a second one washes the image out']])}

<h2>7. Conclusion, limits, next steps</h2>
<p><b>Conclusion:</b> the first-step route holds; the full chain is proven and lands in ComfyUI.</p>
<p><b>Limits:</b> single-plane ~470x cap; quality limited by the small dataset (40 HDRIs); caption/crop flaws; RAW-&gt;Turbo transfer untested.</p>
<p><b>Next (by payoff):</b> (1) data &mdash; honest captioning + wider FOV so sun and scene co-occur; scale to 978 Poly Haven + Fairchild + procedural. (2) sun magnitude &mdash; multi-plane / gain map or exposure stack. (3) sky &mdash; a parametric sun+sky model. (4) attack fabrication at the representation level &mdash; train the VAE encoder (16-&gt;32), at the cost of the free Turbo transfer.</p>

<h2>Appendix: key numbers</h2>
{table(['Item','Value'],
  [['Encoding round-trip error','~1e-13 (float64, 8 curves)'],
  ['Real-VAE code-PSNR','LogC4 38.2 dB (best); naive linear highlight error 7211%'],
  ['Real HDRI','kloofendal 111,253x / 31.6 stops / median 0.178; S2R-HDR 21x (no sun)'],
  ['Dataset','320 imgs / 229 HDR-rich; 93% decode to real HDR; 50% "sun" mislabeled'],
  ['Training','264 LoRA modules, dim/alpha 32, 2290 steps, 10 epochs, ~34 GB VRAM'],
  ['Real-vs-fabricated (window)','pinned-at-ceiling 6.20% -&gt; 0.66%; saturation 7.88% -&gt; 1.09%'],
  ['Generalization','7/8 OOD produce real HDR'],
  ['ComfyUI','LoRA match 0-&gt;264 (after remap), 263 patches; e2e EXR 459x / 8.7 stops'],
  ['Single-plane ceiling','~470x (LogC4)']])}
<p><small>Companion: research-findings.md (route survey), tools/ (validation suite), comfyui/ (nodes + remap + workflow). LoRA and dataset are public on Hugging Face. 2026-07.</small></p>
"""

HTML(string='<style>'+CSS+'</style>'+BODY, base_url=BASE).write_pdf(
    BASE + 'Krea2-HDR-Implementation-Report-EN.pdf')
print('PDF:', round(os.path.getsize(BASE + 'Krea2-HDR-Implementation-Report-EN.pdf')/1e6, 2), 'MB')
