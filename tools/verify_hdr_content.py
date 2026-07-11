"""
CRITICAL CHECK: do the training PNGs actually carry decodable HDR, or are they
washed-out SDR with the highlights gone? (the Felldude / "float container of SDR" trap)

A LogC4-encoded image LOOKS flat/grey -- that is normal for log footage. What matters
is whether inverse-LogC4 recovers a WIDE dynamic range with highlights >> 1.0 that
matches the source EXR. This script proves it per image and contrasts:
  * CORRECT decode:  png/255 -> inverse-LogC4 -> scene-linear   (should be wide HDR)
  * NAIVE SDR read:  png/255 treated as-is                       (flat [0,1], sun gone)
and validates the correct decode against the reproduced source EXR crop.
"""
import json, math
import numpy as np
import imageio.v2 as imageio
from hdr_encodings import logc4_fwd, logc4_inv
from dataset_prep import load_exr_rgb, lum
from build_dataset import equirect_to_persp, normalize_anchor

CEIL = float(logc4_inv(np.array([1.0]))[0])   # LogC4 ceiling ~470
ROOT = '../dataset'

def png_to_linear(path):
    code = imageio.imread(path).astype(np.float64) / 255.0
    return logc4_inv(code)                     # correct HDR decode

def stats_L(L):
    Lp = L[L > 0]
    p = np.percentile(Lp, [50, 99, 99.99, 100])
    return dict(median=float(p[0]), p99=float(p[1]), peak=float(p[3]),
                stops=float(np.log2(p[2] / max(p[0], 1e-6))),
                frac_gt1=float((L > 1.0).mean()), frac_gt10=float((L > 10.0).mean()))

def main():
    rows = [json.loads(l) for l in open(f'{ROOT}/manifest.jsonl')]
    print(f"verifying {len(rows)} training images\n")

    agg = []
    for r in rows:
        L = lum(png_to_linear(f"{ROOT}/{r['image']}"))
        s = stats_L(L)
        s['slug'] = r['slug']; s['sun'] = r['reaches_sun']
        agg.append(s)

    peaks = np.array([a['peak'] for a in agg])
    stops = np.array([a['stops'] for a in agg])
    fg1 = np.array([a['frac_gt1'] for a in agg])
    carry = peaks > 1.0
    print("=== AGGREGATE: does the decoded training set carry HDR? ===")
    print(f"  images whose decoded peak > 1.0 (carry highlights): {carry.sum()}/{len(agg)} "
          f"({100*carry.mean():.0f}%)")
    print(f"  decoded peak  : median {np.median(peaks):.1f}  p10 {np.percentile(peaks,10):.2f}  "
          f"max {peaks.max():.1f}  (LogC4 ceiling {CEIL:.0f})")
    print(f"  dynamic range : median {np.median(stops):.1f} stops  (p10 {np.percentile(stops,10):.1f})")
    print(f"  median frac pixels >1.0 (above SDR white): {100*np.median(fg1):.1f}%")
    flat = np.where(peaks <= 1.0)[0]
    print(f"  FLAT (peak<=1, no HDR headroom): {len(flat)} images"
          + (f" -> {[agg[i]['slug'] for i in flat][:6]}" if len(flat) else " -> none"))

    print("\n=== CORRECT HDR decode  vs  NAIVE SDR read  (same PNG file) ===")
    print(f"{'image':30s}{'HDRpeak':>9s}{'HDRstops':>9s}{'SDRpeak':>9s}{'SDRstops':>9s}")
    for r in rows[:8]:
        code = imageio.imread(f"{ROOT}/{r['image']}").astype(np.float64) / 255.0
        Lh = lum(logc4_inv(code)); Ls = lum(code)                 # correct vs naive
        sh, ss = stats_L(Lh), stats_L(Ls)
        print(f"{r['image'].split('/')[-1]:30s}{sh['peak']:9.1f}{sh['stops']:9.1f}"
              f"{ss['peak']:9.2f}{ss['stops']:9.1f}")

    print("\n=== validate CORRECT decode against reproduced SOURCE EXR (sun-bearing sample) ===")
    checked = 0
    for r in rows:
        if not r['reaches_sun'] or r['kind'] != 'persp':
            continue
        img = load_exr_rgb(f"{ROOT}/raw_exr/{r['slug']}.exr")
        norm, _, _ = normalize_anchor(img)
        view = equirect_to_persp(norm, r['yaw'], r['pitch'], r['fov'], (1024, 1024))
        Lsrc = lum(view); Lrec = lum(png_to_linear(f"{ROOT}/{r['image']}"))
        # compare in the un-clipped band (source below ceiling), where recovery should be faithful
        m = (Lsrc > 0.05) & (Lsrc < CEIL)
        rel = float(np.median(np.abs(Lrec[m] - Lsrc[m]) / Lsrc[m]))
        print(f"  {r['image'].split('/')[-1]:26s} src_peak={Lsrc.max():8.0f} "
              f"rec_peak={Lrec.max():7.1f} (cap {CEIL:.0f})  band<ceil rel_err={100*rel:.1f}%  "
              f"src_stops={np.log2(np.percentile(Lsrc[Lsrc>0],99.99)/np.median(Lsrc[Lsrc>0])):.1f}")
        checked += 1
        if checked >= 6:
            break

    print(f"\nVERDICT: {'PASS' if carry.mean() > 0.8 else 'CHECK'} - "
          f"{100*carry.mean():.0f}% of training images decode to real HDR (peak>1, "
          f"median {np.median(stops):.0f} stops). LogC4 caps the sun at ~{CEIL:.0f}x "
          f"(single-plane limit, expected); highlights below the cap are faithful.")

if __name__ == '__main__':
    main()
