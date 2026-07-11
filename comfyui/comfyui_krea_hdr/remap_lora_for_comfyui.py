"""
Remap a musubi-tuner Krea2 LoRA (keys like `lora_unet_blocks_0_attn_wq`) to the key
form ComfyUI's Krea2 loader accepts. musubi named modules after the NATIVE layer paths;
ComfyUI's destination model keys are those same native paths
(`diffusion_model.blocks.0.attn.wq.weight`), so the remap is deterministic.

Usage:
  python3 remap_lora_for_comfyui.py <in_lora.safetensors> <out_lora.safetensors> [/path/to/ComfyUI] [/path/to/raw.safetensors]

Output loads directly in ComfyUI's standard LoraLoader.
"""
import sys, torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

def build_native_to_diffusers(comfy_path, dit_path):
    sys.path.insert(0, comfy_path)
    import comfy.utils, comfy.model_detection
    f = safe_open(dit_path, 'pt'); sd = {}
    for k in f.keys():
        sd[k] = torch.empty(f.get_slice(k).get_shape(), device='meta', dtype=torch.float16)
    uc = comfy.model_detection.detect_unet_config(sd, '')
    dk = comfy.utils.krea2_to_diffusers(uc, output_prefix='diffusion_model.')
    # native flattened stem (as musubi names it) -> recognized diffusers source stem
    native_to_src = {}
    for src, dst in dk.items():
        if not src.endswith('.weight'):
            continue
        native = 'lora_unet_' + dst[len('diffusion_model.'):-len('.weight')].replace('.', '_')
        native_to_src[native] = src[:-len('.weight')]      # bare diffusers form ComfyUI registers
    return native_to_src

def remap(in_path, out_path, comfy_path, dit_path):
    native_to_src = build_native_to_diffusers(comfy_path, dit_path)
    lora = load_file(in_path)
    out, matched, missed = {}, set(), set()
    for k, v in lora.items():
        for suf in ('.lora_down.weight', '.lora_up.weight', '.alpha', '.dora_scale'):
            if k.endswith(suf):
                stem = k[:-len(suf)]
                src = native_to_src.get(stem)
                if src is None:
                    missed.add(stem);
                else:
                    # ComfyUI accepts the bare diffusers form; use it (also 'diffusion_model.'+src works)
                    out[src + suf] = v; matched.add(stem)
                break
        else:
            out[k] = v                                     # pass through (metadata-ish)
    save_file(out, out_path)
    print(f"remapped {len(matched)} modules; {len(missed)} unmapped"
          + (f" -> {sorted(missed)[:6]}" if missed else ""))
    return out_path

if __name__ == '__main__':
    a = sys.argv
    comfy_path = a[3] if len(a) > 3 else '/workspace/ComfyUI'
    dit_path = a[4] if len(a) > 4 else '/workspace/krea-hdr/training/models/Krea-2-Raw/raw.safetensors'
    remap(a[1], a[2], comfy_path, dit_path)
