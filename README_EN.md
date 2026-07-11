# Krea 2 HDR Toolkit

*English · [中文](README.md)*

A standalone set of **tools and documentation** for making **Krea 2** generate **true
scene-linear HDR** (float, values >> 1, exportable as EXR / usable as IBL) via a small LoRA:
encoding curves + CPU validation, dataset builder & verifier, EXR / HDR-AVIF conversion,
a ComfyUI node pack & workflow, and the full report (with an illustrated PDF).

> **In one line:** "**LogC4 perceptual encoding + a frozen VAE + a DiT LoRA trained on
> RAW**" makes Krea 2 produce **genuinely learned** (not curve-fabricated) extended dynamic
> range that generalizes to out-of-distribution prompts. End-to-end in ComfyUI it yields a
> real HDR EXR (peak 459× / 8.7 stops). Single-plane ceiling is ~470× (sun magnitude needs a
> multi-plane scheme). Details in `reports/`.

**Open resources:** LoRA → [🤗 LAXMAYDAY/krea2-scene-linear-hdr-lora](https://huggingface.co/LAXMAYDAY/krea2-scene-linear-hdr-lora) · dataset → [🤗 datasets/LAXMAYDAY/krea2-scene-linear-hdr-dataset](https://huggingface.co/datasets/LAXMAYDAY/krea2-scene-linear-hdr-dataset)

Model weights are **not** in this repo (see [Weights & data](#weights--data)). This is **tools + reports**.

---

## Install

```bash
pip install -r requirements.txt
# HDR-AVIF export also needs system tools:  apt-get install -y ffmpeg libavif-bin
# optional GPU step (real VAE round-trip):  pip install "torch>=2.4" "diffusers>=0.39"
```
> **Dependency gotcha:** install `opencv-python-headless`, **not** `opencv-python` (which needs `libGL.so.1`).

## Layout

```
krea2-hdr-toolkit/
├── tools/            all CPU tools (flat; imports interop; run from this dir)
├── comfyui/          ComfyUI node pack + workflow + LoRA remap
├── dataset/          dataset card + manifests (images rebuildable via build_dataset.py)
├── reports/          full report (PDF + md) + figure package (figures/)
└── requirements.txt
```

## Core concept (30 s)

Scene-linear radiance (0 to a few hundred ×) is compressed into `[0,1]` with the **ARRI
LogC4** curve, stored as an **8-bit PNG** (HDR is carried by the *encoding*, not bit depth:
a log curve spreads 0–470× across 256 codes, so code 255 decodes back to 470×), fed to the
**frozen** Qwen Image VAE, and a DiT LoRA is trained on RAW. At inference the model outputs
LogC4 codes; **inverse-LogC4** recovers scene-linear light for EXR.

```
train:  EXR → (median → 0.18 normalize) → LogC4[0,1] → 8-bit PNG → VAE latent → DiT LoRA (RAW)
infer:  prompt → Krea2+LoRA → LogC4[0,1] → inverse-LogC4 → scene-linear (>>1) → EXR / HDR AVIF
```

---

## Usage

Run `tools/` scripts from inside `tools/` (they use relative `data/`, `out/`).

### 1) CPU validation suite (no GPU)
```bash
cd tools
python3 run_all.py                 # encoding round-trip + precision + EXR chain + real-data analysis
# or individually:
python3 test_precision.py          # 8 curves invert (~1e-13) + 8-bit precision/banding + plot
python3 test_exr_roundtrip.py      # 32-bit float preserves >1; reproduces clamp/sRGB/half HDR-killers
python3 test_vae_roundtrip.py      # (GPU) real Qwen VAE round-trip: per-curve code-PSNR + per-band error
python3 test_hlg_vs_logc4.py       # HLG vs LogC4 (conclusion: LogC4 wins — see below)
```

### 2) Encoding curves (`hdr_encodings.py`)
Forward/inverse of every transfer function + a registry: `LogC4 / Log3G10 / LogC3 / PU21 / PQ / asinh / HLG / …`
```bash
python3 hdr_encodings.py           # print each curve's code@0.18 and ceiling
```
- Get `(fwd, inv)` by name from `CURVES`; `curve_ceiling(name)` gives the linear peak at code 1.0.
- **HLG vs LogC4 (measured):** LogC4 wins. HLG cannot place mid-gray on-manifold *and* cover
  11 stops at once — at a usable mid-gray its ceiling is only 12× (bright highlights clip), and
  at LogC4's 470× ceiling it crushes mid-gray to ~0.03 (off-manifold). LogC4 uniquely does both.

### 3) Dataset builder & verifier
```bash
cd tools
python3 build_dataset.py --limit 40 --persp 8 --res 1024 --out ../dataset   # Poly Haven CC0 → perspective crops → LogC4
#   --limit 978 for the whole Poly Haven set; outputs PNG + caption + manifest (reproducible crops)
python3 verify_hdr_content.py       # per-image check: images carry decodable HDR (~93% peak > 1)
```
- Train on `../dataset/manifest_hdr_rich.jsonl` (crops with source peak > 2× and ≥ 3 stops).
- Normalization: mid-gray → 0.18 fixed anchor (scale recorded in the manifest).

### 4) EXR / HDR-AVIF conversion (inference post-processing)
```bash
cd tools
python3 decode_to_exr.py  gen.png  out.exr      # generated LogC4 PNG → scene-linear EXR (render / IBL)
python3 to_display_hdr.py out.exr  out.avif     # EXR → HDR AVIF (PQ/BT.2020 for HDR screens; + SDR preview)
```
- `decode_to_exr`: inverse-LogC4 → 32-bit float EXR (no clamp, no sRGB, Rec.709 primaries). Also reads LogC4 PNG.
- `to_display_hdr`: scene-linear → hue-preserving roll-off + highlight desaturation → PQ/BT.2020 → 10-bit HDR10 AVIF via `avifenc`.
- **EXR is the primary deliverable** (Blender/Nuke/Unreal, set the texture to linear / non-sRGB). Use 32-bit float + ZIP/PIZ lossless (half overflows above 65504; DWAA degrades highlights).

### 5) ComfyUI (`comfyui/`)
```bash
# ① Remap the LoRA (musubi key names → ComfyUI; without this 0/264 keys match and the LoRA does nothing)
cd comfyui/comfyui_krea_hdr
python3 remap_lora_for_comfyui.py in.safetensors out_COMFYUI.safetensors /path/to/ComfyUI /path/to/raw.safetensors
# ② Put comfyui_krea_hdr/ in ComfyUI/custom_nodes/; put the LoRA in models/loras/
# ③ Use comfyui/workflow_krea2_hdr_api.json, or wire it per comfyui_krea_hdr/README.md
```
Two nodes: **LogC4ToLinear** (after VAEDecode) + **Save EXR Scene-Linear** (32-bit, no clamp).
> **Key gotcha:** do **not** add a ModelSamplingSD3/shift node — Krea2 already applies its shift; a second one double-shifts and washes the image out. Never use the built-in SaveImage for HDR (8-bit + clamp).
>
> **Upstream (in progress):** a `LogC4` input-colorspace option was added to ComfyUI's built-in `SaveImageAdvanced` (inverse-LogC4 → scene-linear EXR); once merged it removes the need for a separate save node.

### 6) Reports & figures (`reports/`)
- `Krea2-HDR-Implementation-Report-EN.pdf` — illustrated report (real curves + tables + image analysis).
- `implementation-report-EN.md` / `research-findings.md` — implementation & validation / route survey.
- `figures/` — the **figure package** (figA–F curves + analysis images; `FigNN` maps to the report figure numbers).

---

## Key findings (all measured)

1. The route works: LogC4 encoding + frozen VAE + RAW LoRA — Krea 2 learns to generate extended range.
2. Learned, not fabricated: same prompt+seed, the base clips highlights flat (fake), the LoRA spreads them (real); pinned-at-ceiling 6.20% → 0.66%.
3. Generalizes: 7/8 out-of-distribution prompts produce real HDR with the bright region on the actual light source.
4. Runs in ComfyUI: LoRA needs a key remap (0→264) + the nodes; e2e yields a 459× / 8.7-stop EXR.
5. Honest ceiling: single-plane ~470×; sun magnitude needs a multi-plane / gain-map / parametric-sun scheme.

## Weights & data

- **LoRA (public):** [huggingface.co/LAXMAYDAY/krea2-scene-linear-hdr-lora](https://huggingface.co/LAXMAYDAY/krea2-scene-linear-hdr-lora) — musubi/diffusers build + ComfyUI-remapped build + model card.
- **Training dataset (public, CC0):** [huggingface.co/datasets/LAXMAYDAY/krea2-scene-linear-hdr-dataset](https://huggingface.co/datasets/LAXMAYDAY/krea2-scene-linear-hdr-dataset) — 320 LogC4 scene-linear HDR training images + captions + manifests (inverse-LogC4 recovers scene-linear). Also rebuildable via `tools/build_dataset.py`.
- **Base models:** [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw), [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL), [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) (Qwen Image VAE).

## License

Code is MIT (see `LICENSE`). Source HDRIs are Poly Haven CC0. Reports and figures ship with the repo.
