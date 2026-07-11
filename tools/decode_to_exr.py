"""
Inference post-processing #1: generated LogC4 PNG -> scene-linear EXR (for render/IBL).

The Krea2-LoRA (trained on LogC4-encoded images) generates LogC4 code in [0,1], which
the generate script saves as an ordinary PNG. This recovers the HDR:
    png/255 -> inverse-LogC4 -> scene-linear radiance -> 32-bit EXR (ZIP, no clamp).

Usage: python3 decode_to_exr.py <in.png> [out.exr]
"""
import sys, numpy as np, imageio.v2 as imageio, OpenEXR
from hdr_encodings import logc4_inv

def decode_to_exr(png_path, exr_path):
    code = imageio.imread(png_path).astype(np.float64) / 255.0
    if code.ndim == 2:
        code = np.stack([code] * 3, -1)
    lin = logc4_inv(code[..., :3]).astype(np.float32)          # scene-linear, >>1 preserved
    hdr = {'compression': OpenEXR.ZIP_COMPRESSION, 'type': OpenEXR.scanlineimage}
    OpenEXR.File(hdr, {'RGB': np.ascontiguousarray(lin)}).write(exr_path)
    L = lin @ np.array([0.2126, 0.7152, 0.0722])
    print(f"{png_path} -> {exr_path}")
    print(f"  scene-linear peak={L.max():.1f}x  median={np.median(L[L>0]):.3f}  "
          f"stops(p99.99/med)={np.log2(np.percentile(L[L>0],99.99)/np.median(L[L>0])):.1f}  "
          f"frac>1.0={100*(L>1).mean():.1f}%")
    return lin

if __name__ == '__main__':
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.rsplit('.', 1)[0] + '.exr'
    decode_to_exr(inp, out)
