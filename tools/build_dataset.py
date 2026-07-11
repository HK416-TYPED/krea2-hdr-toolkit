"""
Build a Krea2-LoRA-ready scene-linear HDR dataset from Poly Haven (CC0).

For each selected HDRI:
  * load scene-linear EXR
  * fixed-anchor exposure normalize (median -> 0.18; scale recorded for exact inverse)
  * LogC4-encode to [0,1]  (the validated winner on the real Qwen VAE)
  * emit (a) equirect panos [for IBL/HDRI LoRA] and (b) random perspective crops
    [for general HDR], each as 16-bit PNG (LogC4 code) + matching .txt caption
  * write an 8-bit tonemapped preview and a manifest row (with inverse metadata)

Captions are metadata-driven (Poly Haven categories/tags + trigger phrase) -- no GPU,
VLM captioning is an optional later pass.

At INFERENCE the model outputs LogC4 code -> inverse-LogC4 -> scene-linear -> EXR
(the chain validated in test_exr_roundtrip.py / test_vae_roundtrip.py).

Usage: python3 build_dataset.py --limit 30 --persp 4 --res 1024 --out ../dataset
"""
import os, io, json, argparse, math
import numpy as np
import requests
import imageio.v2 as imageio
from hdr_encodings import logc4_fwd, logc4_inv
from dataset_prep import load_exr_rgb, lum, tonemap_agx_ish

API = 'https://api.polyhaven.com'
ANCHOR = 0.18
REC709 = np.array([0.2126, 0.7152, 0.0722])

def select_assets(limit):
    """Diverse selection: prioritize sunny outdoor (IBL) + skies + indoor + urban."""
    assets = requests.get(f'{API}/assets?type=hdris', timeout=30).json()
    def score(k, v):
        cats = set(v.get('categories', []))
        s = 0
        if 'outdoor' in cats: s += 2
        if cats & {'clear', 'skies', 'midday', 'sunrise-sunset'}: s += 2  # sun present
        if 'indoor' in cats: s += 1
        if 'urban' in cats: s += 1
        s += min(v.get('download_count', 0), 50000) / 50000  # popularity tiebreak
        return s
    ranked = sorted(assets.items(), key=lambda kv: -score(*kv))
    # take a diverse spread: interleave to avoid all-sunny
    sunny = [k for k, v in ranked if 'clear' in set(v.get('categories', []))]
    indoor = [k for k, v in ranked if 'indoor' in set(v.get('categories', []))]
    other = [k for k, v in ranked if k not in set(sunny) | set(indoor)]
    out, seen, pools = [], set(), [sunny, other, indoor]
    i = 0
    while len(out) < limit and any(pools):
        p = pools[i % 3]
        if p:
            k = p.pop(0)
            if k not in seen:
                seen.add(k); out.append(k)
        i += 1
    return out[:limit], assets

def download_exr(slug, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    files = requests.get(f'{API}/files/{slug}', timeout=30).json()
    url = files['hdri']['2k']['exr']['url']
    r = requests.get(url, timeout=180); r.raise_for_status()
    open(path, 'wb').write(r.content)
    return path

def normalize_anchor(img):
    L = lum(img); med = float(np.median(L[L > 0]))
    scale = ANCHOR / max(med, 1e-6)
    return (img * scale).astype(np.float32), scale, med

def equirect_to_persp(img, yaw, pitch, fov_deg, out_hw):
    """Sample a pinhole perspective view from an equirect (H,W,3) panorama."""
    H, W = img.shape[:2]; oh, ow = out_hw
    f = 0.5 * ow / math.tan(math.radians(fov_deg) / 2)
    j, i = np.meshgrid(np.arange(ow), np.arange(oh))
    x = (j - ow / 2); y = (i - oh / 2); z = np.full_like(x, f)
    d = np.stack([x, y, z], -1).astype(np.float64)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    cy, sy = math.cos(yaw), math.sin(yaw); cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    dv = d @ (Ry @ Rx).T
    lon = np.arctan2(dv[..., 0], dv[..., 2]); lat = np.arcsin(np.clip(dv[..., 1], -1, 1))
    u = (lon / (2 * math.pi) + 0.5) * W; v = (lat / math.pi + 0.5) * H
    u0 = np.clip(u.astype(int), 0, W - 1); v0 = np.clip(v.astype(int), 0, H - 1)
    return img[v0, u0]

def sun_direction(norm):
    """yaw,pitch pointing at the brightest region of the equirect panorama."""
    H, W = norm.shape[:2]
    L = lum(norm)
    # blur a little (block-max) so we aim at the sun region, not a single hot texel
    bh, bw = H // 64, W // 64
    small = L[:bh * 64, :bw * 64].reshape(bh, 64, bw, 64).max(axis=(1, 3))
    v, u = np.unravel_index(int(small.argmax()), small.shape)
    lon = (u / bw - 0.5) * 2 * math.pi
    lat = (v / bh - 0.5) * math.pi
    return -lon, -lat            # yaw, pitch to look toward it

def hdr_stats(view_linear):
    L = lum(view_linear); Lp = L[L > 0]
    if Lp.size == 0:
        return dict(hdr_peak=0.0, hdr_stops=0.0, frac_gt1=0.0)
    p = np.percentile(Lp, [50, 99.99])
    return dict(hdr_peak=round(float(Lp.max()), 2),
                hdr_stops=round(float(np.log2(max(p[1], 1e-6) / max(p[0], 1e-6))), 2),
                frac_gt1=round(float((L > 1.0).mean()), 5))

def caption(meta, kind, slug):
    cats = [c for c in meta.get('categories', []) if ':' not in c]
    tags = meta.get('tags', [])
    sun = 'with visible sun' if ({'clear', 'midday'} & set(cats) or 'sun' in tags) else ''
    where = 'outdoor' if 'outdoor' in cats else ('indoor' if 'indoor' in cats else '')
    desc = meta.get('name', slug.replace('_', ' '))
    lead = 'scene-linear HDR equirectangular panorama HDRI' if kind == 'pano' else 'scene-linear HDR photo'
    parts = [lead, desc, where, sun, ', '.join(tags[:5]),
             ', '.join(c for c in cats if c not in (where,))[:80],
             'high dynamic range, physically based lighting']
    return ', '.join(p for p in parts if p).strip(', ')

def save_sample(base, code01, cap, meta_row):
    # 8-bit LogC4 code -> standard RGB PNG (trainer-ready; VAE is ~8-bit effective).
    # Exact float LogC4 crops are reproducible from raw_exr + manifest params if needed.
    png8 = np.clip(code01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    imageio.imwrite(base + '.png', png8)
    open(base + '.txt', 'w').write(cap)
    return base + '.png'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=30)
    ap.add_argument('--persp', type=int, default=8)
    ap.add_argument('--res', type=int, default=1024)
    ap.add_argument('--panos', type=int, default=0, help='equirect panos per HDRI (default 0: normal scene images only)')
    ap.add_argument('--pano_w', type=int, default=1536)
    ap.add_argument('--out', default='../dataset')
    args = ap.parse_args()

    root = args.out
    for d in ('images', 'preview', 'raw_exr'):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    slugs, assets = select_assets(args.limit)
    print(f"selected {len(slugs)} HDRIs; emitting {args.persp} persp"
          + (f" + {args.panos} pano" if args.panos else "") + " each")

    manifest = []
    rng = np.random.default_rng(0)
    for n, slug in enumerate(slugs):
        try:
            exr = download_exr(slug, os.path.join(root, 'raw_exr', f'{slug}.exr'))
            img = load_exr_rgb(exr)
            norm, scale, med = normalize_anchor(img)
            meta = assets[slug]; Lmax = float(lum(img).max())
            H, W = norm.shape[:2]
            # optional equirect panos (default OFF: we want normal scene images)
            for pidx in range(args.panos):
                pw = args.pano_w; ph = pw // 2
                pano = norm[np.clip((np.arange(ph)[:, None] * H // ph), 0, H - 1),
                            np.clip((np.arange(pw)[None, :] * W // pw), 0, W - 1)]
                code = np.clip(logc4_fwd(pano), 0, 1).astype(np.float32)
                base = os.path.join(root, 'images', f'{slug}__pano{pidx}')
                cap = caption(meta, 'pano', slug)
                save_sample(base, code, cap, None)
                imageio.imwrite(os.path.join(root, 'preview', f'{slug}__pano{pidx}.jpg'),
                                (tonemap_agx_ish(pano) * 255).astype(np.uint8))
                manifest.append(dict(image=f'images/{slug}__pano{pidx}.png', caption=cap, kind='pano',
                                     source='PolyHaven', slug=slug, license='CC0', encoding='LogC4',
                                     anchor=ANCHOR, exposure_scale=scale, orig_median=med,
                                     orig_max_linear=Lmax, reaches_sun=bool(Lmax > 1000)))
            # perspective crops (normal scene images) -- the main output.
            # Views: some random, plus sun-targeted ones for bright HDRIs so the crops
            # actually exercise the >1.0 range (fixes the "flat SDR crop" problem).
            views = []
            sun_yaw, sun_pitch = sun_direction(norm)
            n_sun = 3 if Lmax > 20 else 0                       # aim at the bright region
            for c in range(n_sun):
                views.append((sun_yaw + float(rng.uniform(-0.4, 0.4)),
                              sun_pitch + float(rng.uniform(-0.2, 0.2)),
                              float(rng.uniform(55, 80))))
            for c in range(args.persp - n_sun):
                views.append((float(rng.uniform(-math.pi, math.pi)),
                              float(rng.uniform(-0.35, 0.15)),
                              float(rng.uniform(55, 85))))
            for c, (yaw, pitch, fov) in enumerate(views):
                view = equirect_to_persp(norm, yaw, pitch, fov, (args.res, args.res))
                code = np.clip(logc4_fwd(view), 0, 1).astype(np.float32)
                base = os.path.join(root, 'images', f'{slug}__v{c}')
                cap = caption(meta, 'persp', slug)
                save_sample(base, code, cap, None)
                imageio.imwrite(os.path.join(root, 'preview', f'{slug}__v{c}.jpg'),
                                (tonemap_agx_ish(view) * 255).astype(np.uint8))
                row = dict(image=f'images/{slug}__v{c}.png', caption=cap, kind='persp',
                           source='PolyHaven', slug=slug, license='CC0', encoding='LogC4',
                           anchor=ANCHOR, exposure_scale=scale, yaw=yaw, pitch=pitch,
                           fov=fov, sun_targeted=(c < n_sun),
                           orig_max_linear=Lmax, reaches_sun=bool(Lmax > 1000))
                row.update(hdr_stats(view))
                manifest.append(row)
            nrich = sum(1 for r in manifest[-len(views):] if r['hdr_peak'] > 2.0)
            print(f"  [{n+1}/{len(slugs)}] {slug:32s} med={med:.3g} sun={Lmax>1000} "
                  f"hdr-rich {nrich}/{len(views)}")
        except Exception as e:
            print(f"  [skip {slug}] {type(e).__name__}: {e}")

    with open(os.path.join(root, 'manifest.jsonl'), 'w') as f:
        for r in manifest: f.write(json.dumps(r) + '\n')
    # HDR-rich filtered training set: keep crops that actually exercise >1.0
    rich = [r for r in manifest if r.get('hdr_peak', 0) > 2.0 and r.get('hdr_stops', 0) >= 3.0]
    with open(os.path.join(root, 'manifest_hdr_rich.jsonl'), 'w') as f:
        for r in rich: f.write(json.dumps(r) + '\n')
    peaks = sorted(r.get('hdr_peak', 0) for r in manifest)
    print(f"\nDONE: {len(manifest)} crops, {sum(r['hdr_peak']>1 for r in manifest)} carry HDR (peak>1). "
          f"HDR-rich (peak>2 & >=3 stops): {len(rich)} -> manifest_hdr_rich.jsonl")
    print(f"      decoded-peak distribution: p10={peaks[len(peaks)//10]:.1f} "
          f"median={peaks[len(peaks)//2]:.1f} p90={peaks[9*len(peaks)//10]:.1f}")

if __name__ == '__main__':
    main()
