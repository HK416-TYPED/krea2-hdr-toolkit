# Krea2 scene-linear HDR LoRA dataset (v1)

Normal-scene HDR training images for a Krea 2 LoRA that learns to generate
**scene-linear HDR** (recoverable to EXR), built from CC0 Poly Haven HDRIs.

## What each sample is

- `images/<slug>__v<k>.png` — a **normal perspective scene image** (re-projected from an
  equirect HDRI), **LogC4-encoded** into [0,1], saved 8-bit RGB. **This is what the VAE
  sees.** It looks slightly flat/grey — that is *correct* for log-encoded footage; the
  HDR is carried in the encoding (see "HDR content" below).
- `images/<slug>__v<k>.txt` — caption (metadata-driven; trigger phrase "scene-linear HDR").
- `preview/<slug>__v<k>.jpg` — 8-bit tonemapped preview for humans only (NOT training).
- `raw_exr/<slug>.exr` — the source 2k equirect HDRI (kept so exact float crops are
  reproducible from manifest params).
- `manifest.jsonl` — one row per crop with everything needed to reproduce & inverse:
  `encoding=LogC4`, `anchor=0.18`, `exposure_scale`, `yaw/pitch/fov`, `sun_targeted`,
  and decoded HDR stats `hdr_peak / hdr_stops / frac_gt1`.
- `manifest_hdr_rich.jsonl` — **filtered training set** (crops with source peak>2 and
  >=3 stops): the HDR-rich subset that actually exercises the >1.0 range. **Train on this.**

## Pipeline (how the HDR is carried and recovered)

```
scene-linear EXR ──median→0.18 normalize (scale recorded)──► linear crop
   ──LogC4 forward──► [0,1] code ──*255──► 8-bit PNG          (training image, VAE input)
INFERENCE: model outputs LogC4 code [0,1] ──inverse-LogC4──► scene-linear ──► EXR (32-bit ZIP)
```
LogC4 chosen as the encoding after a real-Qwen-VAE round-trip test ranked it best
(code-PSNR 38.2, best bright-band handling). Constants & inverse are in
`../hdr_cpu_validation/hdr_encodings.py`; the EXR write chain is validated in
`../hdr_cpu_validation/test_exr_roundtrip.py`.

## HDR content — VERIFIED (this dataset is real HDR, not washed-out SDR)

`../hdr_cpu_validation/verify_hdr_content.py` (run on this build):
- **297/320 crops (93%)** decode (inverse-LogC4) to real HDR with peak > 1.0.
- decoded peak: median ~5.5×, up to the LogC4 ceiling **470×**; median ~4.7 stops.
- **Same PNG, two readings** (proof `out/hdr_content_proof.png`): naive SDR read →
  peak 1.0 (flat); correct inverse-LogC4 decode → peak 470× (sun preserved). The HDR
  info is in the file.
- Validated vs source EXR: below-ceiling band error **0.8%**. The sun disk caps at ~470×
  (single-plane limit — expected; true sun magnitude needs a multi-plane scheme).
- ~22 crops are genuinely low-DR (overcast/indoor); excluded by `manifest_hdr_rich.jsonl`.

## Current scale & how to grow

- v1: 40 CC0 HDRIs × 8 crops = 320 (229 HDR-rich). Enough for a first small LoRA (Path A).
- Scale to all 978 Poly Haven HDRIs: `python3 build_dataset.py --limit 978 --persp 8`.
- Add captions via VLM later (tonemap preview -> Qwen-VL) for richer text.
- Licensing: all source HDRIs are **Poly Haven CC0**. Add Fairchild (public domain) or
  licensed Laval for more real-sun coverage (see research report Area 3).
