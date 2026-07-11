"""
ComfyUI custom nodes for the Krea2 scene-linear HDR LoRA pipeline.

Drop this folder into  ComfyUI/custom_nodes/  and restart.

Why these nodes exist
---------------------
The LoRA is trained to generate images in **LogC4-encoded** space (values live in
[0,1], which is exactly what the VAE and every ComfyUI node expect). The extended
dynamic range only appears AFTER you invert LogC4. So the ComfyUI graph is:

  Load Krea2 -> LoraLoader(this LoRA) -> KSampler -> VAEDecode
      -> [LogC4 to Linear]  (this pack)   # [0,1] LogC4 code -> scene-linear (>>1)
      -> [Save EXR Scene-Linear] (this pack, OR HQ-Image-Save tonemap=linear)

Key facts:
* VAEDecode's built-in clamp to [0,1] is HARMLESS here, because the model's output
  IS a [0,1] LogC4 code. Values >1 are produced by "LogC4 to Linear", after decode.
* ComfyUI's built-in SaveImage is 8-bit and clamps -> never use it for HDR.
* "Save EXR Scene-Linear" writes 32-bit (or half) float, NO clamp, NO sRGB, tagged
  scene-linear. If OpenEXR isn't importable in your ComfyUI env, use
  ComfyUI-HQ-Image-Save's SaveEXR with tonemap="linear" instead (fed by LogC4 to Linear).
"""
import numpy as np
import torch
import os
import folder_paths  # provided by ComfyUI

# ---------- ARRI LogC4 inverse (2022 spec) -- matches training encode ----------
_A = (2.0 ** 18 - 16.0) / 117.45
_B = (1023.0 - 95.0) / 1023.0
_C = 95.0 / 1023.0
_S = (7.0 * np.log(2.0) * 2.0 ** (7.0 - 14.0 * _C / _B)) / (_A * _B)
_T = (2.0 ** (14.0 * (-_C / _B) + 6.0) - 64.0) / _A

def logc4_to_linear(V):
    V = np.asarray(V, np.float64)
    hi = (np.power(2.0, 14.0 * (V - _C) / _B + 6.0) - 64.0) / _A
    lo = V * _S + _T
    return np.where(V >= _C, hi, lo)


class LogC4ToLinear:
    """[0,1] LogC4 code (as generated) -> scene-linear radiance (values can be >> 1)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "exposure": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scene_linear",)
    FUNCTION = "decode"
    CATEGORY = "HDR/Krea2"

    def decode(self, image, exposure):
        x = image.detach().cpu().float().numpy()          # (B,H,W,C) in [0,1]
        lin = logc4_to_linear(np.clip(x, 0.0, 1.0)) * float(exposure)
        return (torch.from_numpy(np.ascontiguousarray(lin.astype(np.float32))),)


class SaveEXRSceneLinear:
    """Write scene-linear IMAGE to 32-bit (or half) float EXR. No clamp, no sRGB."""
    _COMP = None
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "hdr/krea"}),
            "pixel_type": (["float32", "half"],),
            "compression": (["zip", "piz", "zips", "none"],),
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "HDR/Krea2"

    def save(self, images, filename_prefix, pixel_type, compression):
        import OpenEXR
        comp = {"zip": OpenEXR.ZIP_COMPRESSION, "piz": OpenEXR.PIZ_COMPRESSION,
                "zips": OpenEXR.ZIPS_COMPRESSION, "none": OpenEXR.NO_COMPRESSION}[compression]
        out_dir = folder_paths.get_output_directory()
        full, _ = os.path.split(os.path.join(out_dir, filename_prefix))
        os.makedirs(full, exist_ok=True)
        arr = images.detach().cpu().float().numpy()       # (B,H,W,C) raw floats, NO clamp
        dtype = np.float32 if pixel_type == "float32" else np.float16
        saved = []
        for b in range(arr.shape[0]):
            px = np.ascontiguousarray(arr[b, ..., :3].astype(dtype))
            path = os.path.join(out_dir, f"{filename_prefix}_{b:05d}.exr")
            header = {"compression": comp, "type": OpenEXR.scanlineimage,
                      # scene-linear, Rec.709 primaries, D65 (Blender/Nuke default if absent)
                      "chromaticities": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)}
            try:
                OpenEXR.File(header, {"RGB": px}).write(path)
            except TypeError:
                # older/newer signature: header without chromaticities
                OpenEXR.File({"compression": comp, "type": OpenEXR.scanlineimage},
                             {"RGB": px}).write(path)
            saved.append(path)
            print(f"[SaveEXRSceneLinear] wrote {path}  max={float(px.max()):.2f} ({pixel_type})")
        return {"ui": {"text": saved}}


NODE_CLASS_MAPPINGS = {
    "LogC4ToLinear": LogC4ToLinear,
    "SaveEXRSceneLinear": SaveEXRSceneLinear,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LogC4ToLinear": "LogC4 to Linear (HDR decode)",
    "SaveEXRSceneLinear": "Save EXR Scene-Linear (32-bit, no clamp)",
}
