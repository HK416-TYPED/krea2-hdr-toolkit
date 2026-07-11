# Krea 2 → Scene-Linear HDR: Implementation & Validation Report

*English · [中文](implementation-report.md)*

> Companion docs: `research-findings.md` (route survey, 6 exploration areas), this report
> (what was actually built/measured), and the illustrated PDF `Krea2-HDR-Implementation-Report-EN.pdf`.
> Code: the `tools/` and `comfyui/` directories. Date 2026-07 · Env: RTX PRO 6000 96GB / torch 2.9 / diffusers 0.39 / ComfyUI v0.27.0.

## 0. Summary

We made **Krea 2** (12.9B, Qwen Image VAE) generate **true scene-linear HDR** (EXR, values >> 1)
via a small LoRA, validated end-to-end into ComfyUI.

**Core conclusions (all backed by measurements):**

1. **The route works.** "LogC4 perceptual encoding + frozen VAE + a DiT LoRA on RAW" makes Krea 2 generate extended dynamic range.
2. **Learned, not fabricated.** Same prompt + seed, the base model clips highlights flat against the decode ceiling (fake HDR); the LoRA spreads them into graduated values (real HDR) — pinned-at-ceiling 6.20% → 0.66%.
3. **Generalizes.** For content entirely absent from training (fire / neon / portrait / cathedral / candle), 7/8 prompts produce real HDR with the brightest region on the actual light source.
4. **Runs in ComfyUI.** The musubi LoRA needs a key remap to load (0→264) plus two nodes; end-to-end it yields a real HDR EXR (peak 459× / 8.7 stops).
5. **Honest ceiling.** Single-plane LogC4 caps peak radiance at ~470× — not literal sun magnitude (10⁵⁺); that needs a multi-plane scheme. Quality is limited by the epoch-10 small dataset (40 HDRIs), but **HDR-realness and image quality are independent axes** — this project proves the former.

## 1. Method & pipeline

The VAE is not modified (preserving the official RAW→Turbo LoRA transfer). Scene-linear
radiance is compressed into `[0,1]` with **ARRI LogC4**, fed to the frozen Qwen Image VAE; the
model learns to generate in LogC4 space; **inverse-LogC4** recovers linear light for EXR.

```
train:  scene-linear EXR → (median→0.18) → LogC4[0,1] → 8-bit PNG → VAE latent → DiT LoRA (RAW)
infer:  prompt → Krea2 + LoRA → LogC4[0,1] → inverse-LogC4 → scene-linear (>>1) → EXR / HDR AVIF
```

**Why LogC4:** on a real-Qwen-VAE round-trip it had the best code-domain PSNR and best bright-band
handling (§2.4). **Key property:** VAEDecode's built-in `[0,1]` clamp is harmless here — the model
outputs a `[0,1]` LogC4 code; values >1 only appear *after* inverse-LogC4.

**Why 8-bit is fine:** bit depth ≠ dynamic range. A log curve spreads 0–470× across 256 codes
(~17 codes/stop); code 255 decodes to 470×, "display white" (1.0) sits at code 109. The frozen VAE
is ~8-bit effective anyway, so 8-bit LogC4 loses no range — only fineness.

## 2. CPU validation (no GPU)

Before writing training code, the linear↔encoding mapping and the EXR chain were validated numerically.

- **2.1 Round-trip invertibility.** All 8 curves invert to ~1e-13 (float64). Mapping and inverse are exact.
- **2.2 Mid-gray placement (on-manifold).** code@0.18: Log3G10 0.333, LogC4 0.278, PU21 0.359, PQ 0.348 (near sRGB's ~0.46); mu-law 0.015 and Log-Gamma 0.809 are badly placed → confirmed poor.
- **2.3 Precision ceiling (a correction to the research report).** Under a uniform 8-bit VAE proxy, log curves give ~17 codes/stop with no banding wall below the curve ceiling — so the "~85 codes → 4–6 stops" arithmetic claim was oversimplified. The real limiter is the VAE's *non-uniform* precision plus the sun overflowing the curve ceiling.
- **2.4 Real Qwen VAE round-trip.** Confirmed first-hand the decoder has **no tanh/sigmoid** — its only bound is `torch.clamp(-1,1)`, so the bottleneck is the encoder, not the decoder. **X2HDR's premise confirmed:** perceptually-encoded HDR round-trips at ~35–38 dB code-PSNR / ~2% mid+high band error, while naive scaled-linear fed to the VAE gives **7211%** error in the 1–50 band. **Curve ranking (code-PSNR on a real HDRI):** LogC4 38.2 > PU21 37.7 > PQ 37.1 > Log3G10 36.9 > asinh 36.1 > LogC3 35.4. The real VAE degrades **both ends** (deep shadow ~22–24%, sun clips); the clean window is mid+high (0.05–50) at ~2%.
- **2.5 EXR chain.** 32-bit float ZIP/PIZ preserves a 4000× sun exactly; half overflows above 65504 → +Inf; the two silent killers reproduced: clamp→[0,1] turns 4000→1.0, implicit sRGB turns 4000→33.4.
- **2.6 Real data.** Poly Haven `kloofendal`: median 0.178, max 111,253×, 31.6 stops, real sun. S2R-HDR sample: peak only 21×, no sun → a scene-linear hygiene source, not an IBL/sun backbone.
- **2.7 HLG vs LogC4.** LogC4 wins. HLG can't put mid-gray on-manifold **and** cover 11 stops: at a usable mid-gray its ceiling is 12× (bright band clips, 88% error); at LogC4's 470× ceiling it crushes mid-gray to 0.034 (off-manifold) — its higher code-PSNR (50.9 dB) is a mirage of a dark image being trivially reconstructable, while its actual mid-band linear error (8%) is worse than LogC4's (2.4%).

## 3. Dataset

40 CC0 Poly Haven HDRIs re-projected into **normal perspective scene images** (not panoramas);
320 crops, 229 HDR-rich; LogC4-encoded 8-bit PNG + caption + manifest (records
encoding/anchor/exposure_scale/yaw/pitch/fov, so float crops are reproducible).

- **3.1 HDR content verified.** 93% of images decode (inverse-LogC4) to real HDR (peak > 1.0). Same PNG: read naively as SDR → peak 1.0 (flat/grey); read correctly → peak 470× (sun/highlights present). Against source EXR: below-ceiling error 0.8%; the sun caps at ~470×.
- **3.2 A real dataset flaw.** 50% of "with visible sun" captions sit on crops with no sun in frame; parking-lot content and the sun ended up in different crops → the model can't compose "parking lot + sun". Fix: honest per-crop captioning (only write "sun" when hdr_peak is high) + wider FOV so sun and scene co-occur.

## 4. Training & pipeline audit

musubi-tuner `krea2_train_network.py`: RAW + `lora_krea2`, dim/alpha 32 (264 modules), lr 1e-4,
adamw8bit, 10 epochs / 2290 steps. **Training VRAM measured ~34 GB** (bf16 DiT + adamw8bit +
gradient checkpointing + batch 2 @ 1024²); fits a 24GB card with `--fp8_scaled` + `--blocks_to_swap`.

**Pipeline safety (audited):** musubi is safe out of the box (`/127.5-1` normalization, no color
augmentation). ai-toolkit is fine **only if color augmentation is disabled** (its ColorJitter would
scramble the LogC4 pixels). Rule: never apply any color op to LogC4 pixels; under this route the
trainer needs no HDR-awareness — the HDR lives in the encoding.

## 5. Results

- **5.1 End-to-end loop (proven at epoch 2).** prompt → Krea2+LoRA → LogC4 → inverse-LogC4 → scene-linear EXR: peak 65×, 30% of pixels > 1, sun spatially correct.
- **5.2 Real vs fabricated (core evidence).** Same prompt + seed, base vs LoRA, judged by highlight clipping:

  | scene | metric | base | HDR-LoRA |
  |---|---|---|---|
  | indoor window | code saturation (>0.98) | 7.88% | 1.09% |
  | indoor window | pinned at ceiling | 6.20% | 0.66% |
  | night lamp | code saturation | 1.15% | 0.01% |

  The base clips the window to a flat white slab (fake HDR); the LoRA holds it as graduated values (real HDR) — direct evidence the backbone learned extra information. (">1 alone isn't proof — the decode curve inflates any image; the proof is reduced clipping + a highlight histogram that spreads instead of spiking at the ceiling.)
- **5.3 Generalization (epoch 10, out of distribution).** 7/8 community prompts (portrait/fireplace/neon/dragon/cathedral/candle) produce real HDR (≤0.01% pinned, light on the source). The candle that failed at epoch 8 is fixed at 10 (flame 400×). The 8th (spaceship cockpit) is **not a black frame** — the model rendered a dim deep-space starfield (content visible when brightened) with no bright source, so it carries no extended range and reads dark once decoded — a "dark scene, no source" outcome, not a mechanism failure.
- **5.4 Color.** Highlights are neutral, not blue-tinted (lighthouse lamp R:G:B = 0.96:1.00:0.74). An observed "blue overflow" was a display-side issue (per-channel hard clip, no highlight desaturation), fixed in `to_display_hdr.py`; the EXR is unaffected.

## 6. Display delivery

- **Primary = scene-linear EXR** (render / IBL; texture set to linear / non-sRGB).
- **Secondary = HDR AVIF** (HDR screens): scene-linear → hue-preserving roll-off + highlight desaturation → PQ/BT.2020 → 10-bit HDR10 AVIF via `avifenc`. Master EXR: 32-bit float + ZIP/PIZ lossless (half overflows; DWAA degrades highlights).

## 7. ComfyUI — end-to-end verified

- **LoRA loading:** the musubi LoRA does **not** load as-is (ComfyUI's own loader matches 0/264 keys — musubi uses native layer names, ComfyUI expects diffusers-derived ones). `remap_lora_for_comfyui.py` maps 264/264, and ComfyUI attaches 263 patches at load.
- **EXR save:** the built-in SaveImage is 8-bit + clamps → HDR lost. The gap (ComfyUI has no inverse-LogC4) is filled by the node pack: **LogC4→Linear** + **Save EXR Scene-Linear**. (Also contributed upstream: a `LogC4` colorspace option for the built-in `SaveImageAdvanced`.)
- **End-to-end:** native graph `UNETLoader → CLIPLoader(krea2) → VAELoader → LoraLoader(remapped) → KSampler → VAEDecode → LogC4ToLinear → SaveEXRSceneLinear` → real scene-linear HDR EXR (peak 459×, 8.7 stops). **Gotcha:** do not add ModelSamplingSD3 — Krea2 already applies its resolution-aware shift; a second one double-shifts and washes the image out.

## 8. Conclusion · limits · next steps

**Conclusion:** the first-step route holds; the full chain is proven and lands in ComfyUI.

**Limits:** single-plane ~470× cap; quality limited by the small dataset; caption/crop flaws; RAW→Turbo transfer untested.

**Next (by payoff):** (1) data — honest captioning + wider FOV so sun and scene co-occur; scale to all 978 Poly Haven + Fairchild + procedural Blender/Nishita with an explicit sun lamp. (2) sun magnitude — multi-plane / gain map (HDR = SDR·2^gain) or an exposure-stack. (3) sky IBL — a parametric sun+sky model. (4) RAW→Turbo transfer. (5) attack fabrication at the representation level — train the VAE encoder (16→32, Sumit-style), at the cost of the free Turbo transfer.

## Appendix: key numbers

- Encoding round-trip error ~1e-13 (float64, 8 curves)
- Real-VAE code-PSNR: LogC4 38.2 dB (best); naive linear highlight error 7211%
- Real HDRI: kloofendal 111,253× / 31.6 stops / median 0.178; S2R-HDR 21× (no sun)
- Dataset: 320 imgs / 229 HDR-rich; 93% decode to real HDR; 50% "sun" mislabeled
- Training: 264 LoRA modules, dim/alpha 32, 2290 steps, 10 epochs, ~34 GB VRAM
- Real-vs-fabricated (window): pinned-at-ceiling 6.20% → 0.66%; saturation 7.88% → 1.09%
- Generalization: 7/8 OOD produce real HDR
- ComfyUI: LoRA match 0→264 (after remap), 263 patches; e2e EXR 459× / 8.7 stops
- Single-plane ceiling: ~470× (LogC4)
