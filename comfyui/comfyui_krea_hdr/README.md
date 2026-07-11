# comfyui_krea_hdr — run the Krea2 scene-linear HDR LoRA in ComfyUI

Verified end-to-end in ComfyUI v0.27.0: Krea2 + this LoRA → native graph → real
scene-linear HDR EXR (peak 459×, 8.7 stops).

## Install
1. Copy this folder to `ComfyUI/custom_nodes/comfyui_krea_hdr`, restart ComfyUI.
   (needs `OpenEXR` importable in ComfyUI's Python; v3.x is fine.)
2. Put the model files in ComfyUI's folders:
   - `models/diffusion_models/krea2_raw.safetensors`  (Krea-2 RAW DiT)
   - `models/text_encoders/qwen3vl_4b_bf16.safetensors`
   - `models/vae/qwen_image_vae.safetensors`

## Step 0 — REMAP the LoRA (required, one-time)
A musubi-trained Krea2 LoRA does **NOT** load in ComfyUI as-is (0/264 keys match —
musubi names modules `lora_unet_blocks_0_attn_wq`, ComfyUI expects diffusers-derived
keys). Remap it:

```bash
python3 remap_lora_for_comfyui.py  in_musubi_lora.safetensors  out_comfyui_lora.safetensors  /path/to/ComfyUI  /path/to/krea2_raw.safetensors
```
Verified: 264/264 modules remap, and ComfyUI attaches 263 patches at load. Put the
output in `models/loras/`.

## Nodes
- **LogC4 to Linear (HDR decode)** — `[0,1]` LogC4 code (as generated) → scene-linear
  radiance (values >> 1). Insert AFTER VAEDecode. `exposure` multiplies the result.
- **Save EXR Scene-Linear (32-bit, no clamp)** — writes 32-bit/half float EXR, no clamp,
  no sRGB, Rec.709 primaries tagged scene-linear. Feed it the LogC4-to-Linear output.

## The graph (order matters)
```
UNETLoader(krea2_raw) ─┐
CLIPLoader(type=krea2) ─┼─ LoraLoader(remapped) ─ CLIPTextEncode(+/-) ─┐
VAELoader(qwen vae) ────┘                          EmptySD3LatentImage ─┤
                                                                        KSampler
   (cfg 5.5, euler, simple, 28 steps)  ← NO ModelSamplingSD3 node! ─────┘
     └─ VAEDecode ─ LogC4 to Linear ─ Save EXR Scene-Linear
```
`workflow_krea2_hdr_api.json` is the exact working /prompt-format graph.

## Gotchas (verified)
- **Do NOT add ModelSamplingSD3/Flux shift.** Krea2 already applies its resolution-aware
  shift; an extra shift node double-shifts and washes the image out (median code 0.80,
  2.8 stops). Without it: median 0.36, 8.7 stops, correct.
- **VAEDecode's [0,1] clamp is harmless here** — the model outputs a [0,1] LogC4 code;
  values >1 appear only after "LogC4 to Linear". No VAEDecode patch needed.
- **Never use the built-in SaveImage** (8-bit, clamps → HDR lost). Use this pack's
  SaveEXR, or ComfyUI-HQ-Image-Save with tonemap="linear", fed by LogC4-to-Linear.
- To view the EXR on an HDR screen, convert to PQ/BT.2020 AVIF (see
  `../hdr_cpu_validation/to_display_hdr.py`). The EXR itself is for render/IBL (Blender,
  Nuke, Unreal — set the texture to linear / non-sRGB).
```
