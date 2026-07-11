"""
Dataset preparation + real-data validation for the Krea2 scene-linear HDR pipeline.

Downloads a real CC0 Poly Haven HDRI (scene-linear EXR, outdoor with sun), inspects
its radiance distribution vs the claims in the research report, prepares a tonemapped
LDR proxy for VLM captioning, and runs the encode->8bit-proxy->inverse restoration on
REAL radiance data (not synthetic) per luminance band.

Usage:  python3 dataset_prep.py [slug]
"""
import sys, os, json
import numpy as np
import requests
import OpenEXR
from hdr_encodings import log3g10_fwd, log3g10_inv, logc4_fwd, logc4_inv, pq_fwd, pq_inv

os.makedirs('data', exist_ok=True); os.makedirs('out', exist_ok=True)
REC709 = np.array([0.2126, 0.7152, 0.0722])

def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  [cached] {path}"); return path
    print(f"  downloading {url}")
    r = requests.get(url, timeout=120); r.raise_for_status()
    with open(path, 'wb') as f: f.write(r.content)
    print(f"  saved {path} ({len(r.content)/1e6:.1f} MB)"); return path

def load_exr_rgb(path):
    f = OpenEXR.File(path); ch = f.channels()
    for key in ('RGBA', 'RGB'):
        if key in ch:
            a = np.asarray(ch[key].pixels, np.float32)
            return a[..., :3]
    def get(*names):
        for n in names:
            if n in ch: return np.asarray(ch[n].pixels, np.float32)
        raise KeyError(names)
    return np.stack([get('R'), get('G'), get('B')], -1)

def lum(img): return img @ REC709

def analyze(img, name):
    L = lum(img); Lp = L[L > 0]
    pcts = {p: float(np.percentile(Lp, p)) for p in (1, 50, 99, 99.9, 99.99, 100)}
    med = pcts[50]
    dr_log10 = float(np.log10(Lp.max() / max(Lp.min(), 1e-8)))
    peak_stops_over_med = float(np.log2(Lp.max() / max(med, 1e-8)))
    # sun test: how concentrated is the top energy?
    thr = pcts[99.99]
    hot = L > max(thr, 50.0)
    frac_hot = float(hot.mean())
    out = dict(
        name=name, shape=list(img.shape), min=float(Lp.min()), max=float(Lp.max()),
        median=med, percentiles=pcts, dynamic_range_log10=round(dr_log10, 2),
        dynamic_range_stops=round(dr_log10 / np.log10(2), 1),
        peak_stops_over_median=round(peak_stops_over_med, 1),
        frac_pixels_over_50=round(frac_hot, 6),
        has_sun_magnitude=bool(Lp.max() > 1000.0),
    )
    return out, L

def tonemap_agx_ish(img):
    """Cheap filmic tonemap (Reinhard-extended + gamma) for a captioning LDR proxy."""
    x = np.clip(img, 0, None)
    x = x / (1.0 + x)                     # Reinhard
    return np.clip(np.power(x, 1/2.2), 0, 1)

def band_restore(img, fwd, inv, name):
    """Real-data restoration error per luminance band under 8-bit VAE proxy."""
    V = np.clip(fwd(img), 0, 1)
    rec = inv(np.round(V * 255) / 255)
    L = lum(img); Lr = lum(rec)
    bands = {'shadow(<0.05)': L < 0.05, 'mid(0.05-1)': (L >= 0.05) & (L < 1),
             'high(1-50)': (L >= 1) & (L < 50), 'bright(50-500)': (L >= 50) & (L < 500),
             'sun(>500)': L >= 500}
    res = {}
    for b, m in bands.items():
        if m.sum() == 0: res[b] = None; continue
        res[b] = round(float(np.median(np.abs(Lr[m] - L[m]) / np.maximum(L[m], 1e-6))), 4)
    ceil = float(inv(np.array([1.0]))[0])
    clipped = float((L > ceil).mean())
    return dict(curve=name, ceiling=round(ceil, 1), frac_clipped_by_ceiling=round(clipped, 6),
                band_relerr=res)

def analyze_s2r():
    """Download one S2R-HDR frame and contrast its dynamic range with Poly Haven."""
    url = ('https://huggingface.co/datasets/iimmortall/S2R-HDR/resolve/main/'
           'scene_0_FM/img/0000.exr')
    path = download(url, 'data/s2r_scene0_0000.exr')
    img = load_exr_rgb(path)
    info, _ = analyze(img, 'S2R-HDR/scene_0_FM/0000')
    print("\n== S2R-HDR sample (iimmortall/S2R-HDR, CC-BY-4.0) ==")
    print(f"  {info['shape']}  median={info['median']:.4g}  max={info['max']:.4g}  "
          f"DR={info['dynamic_range_stops']} stops  peak-over-median={info['peak_stops_over_median']} stops")
    print(f"  reaches sun-magnitude (>1000): {info['has_sun_magnitude']}   "
          f"-> {'IBL/sun backbone' if info['has_sun_magnitude'] else 'scene-linear hygiene ONLY (no sun)'}")
    return info

def main():
    slug = sys.argv[1] if len(sys.argv) > 1 else 'kloofendal_43d_clear'
    print(f"== Poly Haven HDRI: {slug} (CC0) ==")
    files = requests.get(f'https://api.polyhaven.com/files/{slug}', timeout=30).json()
    url = files['hdri']['2k']['exr']['url']
    path = download(url, f'data/{slug}_2k.exr')
    img = load_exr_rgb(path)
    print(f"  loaded {img.shape} dtype={img.dtype}\n")

    info, L = analyze(img, slug)
    print("-- radiance analysis (real scene-linear data) --")
    print(f"  min={info['min']:.4g}  median={info['median']:.4g}  max={info['max']:.4g}")
    print(f"  dynamic range: {info['dynamic_range_log10']} log10  = {info['dynamic_range_stops']} stops")
    print(f"  peak over median: {info['peak_stops_over_median']} stops")
    print(f"  percentiles(L): " + "  ".join(f"p{p}={v:.3g}" for p, v in info['percentiles'].items()))
    print(f"  frac pixels >50 (sun/specular): {info['frac_pixels_over_50']}")
    print(f"  reaches sun-magnitude (>1000): {info['has_sun_magnitude']}\n")

    # captioning proxy
    tm = (tonemap_agx_ish(img) * 255).astype(np.uint8)
    try:
        from PIL import Image; Image.fromarray(tm).save('out/caption_proxy.png')
        print("-- captioning: saved out/caption_proxy.png (tonemapped LDR for VLM) --")
        print(f"   suggested caption scaffold: 'scene-linear HDRI, {slug.replace('_',' ')}, "
              f"outdoor, clear sky with sun, peak {info['max']:.0f}x, DR {info['dynamic_range_stops']} stops'\n")
    except Exception as e:
        print("PIL save skipped:", e)

    # real-data restoration under each recommended curve
    print("-- real-data restoration (encode -> 8-bit VAE proxy -> inverse), per luminance band --")
    rows = []
    for nm, (fw, iv) in [('Log3G10', (log3g10_fwd, log3g10_inv)),
                         ('LogC4', (logc4_fwd, logc4_inv)),
                         ('PQ', (pq_fwd, pq_inv))]:
        r = band_restore(img, fw, iv, nm); rows.append(r)
        br = r['band_relerr']
        fmt = lambda k: ('%.2f%%' % (100*br[k])) if br.get(k) is not None else '   .'
        print(f"  {nm:8s} ceil={r['ceiling']:6.0f} clipped={r['frac_clipped_by_ceiling']:.4%}  "
              f"shadow={fmt('shadow(<0.05)')} mid={fmt('mid(0.05-1)')} high={fmt('high(1-50)')} "
              f"bright={fmt('bright(50-500)')} sun={fmt('sun(>500)')}")

    s2r = None
    try:
        s2r = analyze_s2r()
    except Exception as e:
        print("\n[S2R-HDR sample skipped]", e)

    report = dict(dataset=slug, license='CC0', source='Poly Haven', analysis=info,
                  restoration=rows, s2r_hdr_sample=s2r)
    with open('out/dataset_report.json', 'w') as f: json.dump(report, f, indent=2)
    print("\n[saved] out/dataset_report.json")
    return info

if __name__ == '__main__':
    main()
