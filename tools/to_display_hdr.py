"""
Inference post-processing #2: scene-linear EXR -> DISPLAY HDR (PQ/BT.2020) AVIF that an
HDR screen (HDR10-capable monitor / phone / Chrome / Safari) shows with real high dynamic
range, PLUS an SDR tonemapped preview for comparison.

This is the OPTIONAL second delivery line (the primary target is the scene-linear EXR for
render/IBL). It converts scene-referred radiance -> display-referred nits -> PQ (ST.2084)
-> HDR10-tagged AVIF via ffmpeg/libaom.

Mapping: ref white 1.0 linear = REF_NITS nits; peak clipped to PEAK_NITS (typical HDR
monitor). Rec.709 primaries -> Rec.2020 for correct HDR container.

Usage: python3 to_display_hdr.py <in.exr|in.png(LogC4)> [out.avif]
"""
import sys, os, subprocess, numpy as np, imageio.v2 as imageio, OpenEXR
from hdr_encodings import logc4_inv

REF_NITS = 100.0      # 1.0 scene-linear (diffuse white) -> 100 nits
PEAK_NITS = 1000.0    # clip display peak (typical HDR monitor)

# Rec.709 -> Rec.2020 primaries matrix
M_709_2020 = np.array([[0.6274, 0.3293, 0.0433],
                       [0.0691, 0.9195, 0.0114],
                       [0.0164, 0.0880, 0.8956]])
PQ = dict(m1=0.1593017578125, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875)

def load_linear(path):
    if path.lower().endswith('.exr'):
        from dataset_prep import load_exr_rgb
        return load_exr_rgb(path).astype(np.float64)
    code = imageio.imread(path).astype(np.float64) / 255.0     # LogC4 PNG
    return logc4_inv(code[..., :3])

W709 = np.array([0.2126, 0.7152, 0.0722])

def pq_encode_nits(nits):
    Y = np.clip(nits / 10000.0, 0, 1)
    Ym = np.power(Y, PQ['m1'])
    return np.power((PQ['c1'] + PQ['c2'] * Ym) / (1 + PQ['c3'] * Ym), PQ['m2'])

def tonemap_display(lin, ref=REF_NITS, peak=PEAK_NITS, desat_start=0.55):
    """scene-linear -> display nits, HUE-PRESERVING soft rolloff to `peak` + highlight
    desaturation (path-to-white). Avoids per-channel-clip hue shift and saturated-color
    'overflow' on HDR displays."""
    lin = np.clip(lin, 0, None)
    Y = (lin @ W709) * ref                                      # luminance in nits
    Yt = Y / (1.0 + Y / peak)                                   # soft rolloff, asymptote=peak
    scale = Yt / np.maximum(Y, 1e-6)
    rgb = lin * ref * scale[..., None]                          # preserve hue: lum -> Yt
    f = np.clip((Yt / peak - desat_start) / (1 - desat_start), 0, 1) ** 1.5
    rgb = rgb * (1 - f[..., None]) + Yt[..., None] * f[..., None]  # blend toward neutral
    return np.clip(rgb, 0, peak)

def to_display_hdr(inp, out):
    lin709 = np.clip(load_linear(inp), 0, None)
    nits709 = tonemap_display(lin709)                           # hue-preserve + desat, in nits
    nits = np.clip(nits709 @ M_709_2020.T, 0, PEAK_NITS)        # 709 -> 2020 primaries
    pq16 = np.clip(pq_encode_nits(nits) * 65535 + 0.5, 0, 65535).astype(np.uint16)
    tmp = out + '.pq16.png'
    import cv2
    cv2.imwrite(tmp, cv2.cvtColor(pq16, cv2.COLOR_RGB2BGR))     # 16-bit PQ PNG, BT.2020
    # wrap into HDR10-tagged AVIF
    # avifenc: tag as HDR via CICP 9/16/9 = BT.2020 primaries / PQ(ST.2084) / BT.2020-NCL
    cmd = ['avifenc', '--cicp', '9/16/9', '-d', '10', '-y', '420',
           '--range', 'full', '-s', '6', tmp, out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(out)
    # SDR preview (Reinhard) for side-by-side
    sdr = np.clip(np.power(lin709 / (1 + lin709), 1 / 2.2) * 255, 0, 255).astype(np.uint8)
    imageio.imwrite(out.rsplit('.', 1)[0] + '_sdr_preview.jpg', sdr)
    L = lin709 @ np.array([0.2126, 0.7152, 0.0722])
    print(f"{inp} -> {out}  ({'OK' if ok else 'AVIF FAILED: ' + r.stderr[-200:]})")
    print(f"  scene peak {L.max():.0f}x -> display {min(L.max()*REF_NITS, PEAK_NITS):.0f} nits "
          f"(ref white {REF_NITS:.0f}, peak clip {PEAK_NITS:.0f})")
    print(f"  highlights reach {100*(L*REF_NITS>REF_NITS).mean():.1f}% of pixels above SDR white "
          f"-> visible HDR on an HDR display")
    print(f"  kept PQ16 PNG: {tmp} ; SDR preview: {out.rsplit('.',1)[0]}_sdr_preview.jpg")
    if not ok:
        print("  NOTE: view {tmp} in an HDR-aware tool, or run the ffmpeg cmd where libaom works.")
    return ok

if __name__ == '__main__':
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.rsplit('.', 1)[0] + '_hdr.avif'
    to_display_hdr(inp, out)
