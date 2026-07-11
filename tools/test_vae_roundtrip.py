"""
GPU experiment: REAL frozen Qwen Image VAE encode->decode round-trip on real HDR,
replacing the CPU quantization proxy. This is the "决定性往返实验" from the report.

For each encoding curve:
  scene-linear L --curve--> E in [0,1] --(*2-1)--> Qwen VAE encode -> decode
  --> ((.)+1)/2 = E_hat --curve^-1--> L_hat.
Measures:
  * code-domain PSNR (E vs E_hat)  == the X2HDR "does the SDR VAE round-trip the
    encoded HDR at LDR fidelity?" metric.
  * per-luminance-band relative error on scene-linear L_hat vs L (real dynamic range).
Baselines: naive-linear-clip and naive-linear-scaled (feeding linear >1 straight in)
  reproduce X2HDR's "linear RGB causes severe degradation".

Run: python3 test_vae_roundtrip.py
"""
import sys, json, time
import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage
from hdr_encodings import (log3g10_fwd, log3g10_inv, logc3_fwd, logc3_inv,
                           logc4_fwd, logc4_inv, pu21_fwd, pu21_inv, pq_fwd, pq_inv,
                           asinh_fwd, asinh_inv)
from dataset_prep import load_exr_rgb, lum

DEV = 'cuda'
REC709 = np.array([0.2126, 0.7152, 0.0722])
CURVES = {
    'Log3G10': (log3g10_fwd, log3g10_inv), 'LogC3': (logc3_fwd, logc3_inv),
    'LogC4': (logc4_fwd, logc4_inv), 'PU21': (pu21_fwd, pu21_inv),
    'PQ': (pq_fwd, pq_inv), 'asinh': (asinh_fwd, asinh_inv),
}

def crop16(img):
    h, w = img.shape[:2]
    return img[:h // 16 * 16, :w // 16 * 16]

def vae_roundtrip(vae, E):
    """E: HxWx3 in [0,1] -> VAE round-trip -> HxWx3 in [0,1]."""
    x = torch.from_numpy(E.transpose(2, 0, 1)[None, :, None]).float().to(DEV) * 2 - 1
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.mode()
        rec = vae.decode(lat).sample
    r = rec[0, :, 0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip((r + 1) / 2, 0.0, 1.0), int(np.prod(lat.shape[1:]))

def psnr01(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)

def bands(L):
    return {'shadow(<0.05)': L < 0.05, 'mid(0.05-1)': (L >= 0.05) & (L < 1),
            'high(1-50)': (L >= 1) & (L < 50), 'bright(50-500)': (L >= 50) & (L < 500),
            'sun(>500)': L >= 500}

def band_relerr(L, Lr):
    out = {}
    for b, m in bands(L).items():
        out[b] = None if m.sum() == 0 else round(float(np.median(
            np.abs(Lr[m] - L[m]) / np.maximum(L[m], 1e-6))), 4)
    return out

def run_image(vae, img, tag):
    img = np.clip(crop16(img), 0, None).astype(np.float32)
    L = lum(img)
    print(f"\n===== {tag}  shape={img.shape}  median={np.median(L[L>0]):.4g}  max={L.max():.4g} =====")
    print(f"{'curve':9s}{'codePSNR':>10s}{'ceil':>7s}{'clip%':>7s}  "
          f"{'shadow':>8s}{'mid':>8s}{'high':>8s}{'bright':>8s}{'sun':>8s}")
    results = {}
    for name, (fw, iv) in CURVES.items():
        E = np.clip(fw(img), 0.0, 1.0).astype(np.float32)
        Ehat, latdim = vae_roundtrip(vae, E)
        Lr = lum(iv(Ehat))
        cp = psnr01(E, Ehat)
        be = band_relerr(L, Lr)
        ceil = float(iv(np.array([1.0]))[0])
        clip = float((L > ceil).mean())
        fmt = lambda k: ('%.1f%%' % (100 * be[k])) if be.get(k) is not None else '  .'
        print(f"{name:9s}{cp:10.2f}{ceil:7.0f}{100*clip:7.3f}  "
              f"{fmt('shadow(<0.05)'):>8s}{fmt('mid(0.05-1)'):>8s}{fmt('high(1-50)'):>8s}"
              f"{fmt('bright(50-500)'):>8s}{fmt('sun(>500)'):>8s}")
        results[name] = dict(code_psnr=round(cp, 2), ceiling=round(ceil, 1),
                             frac_clipped=round(clip, 6), band_relerr=be)

    # baseline: naive linear (X2HDR's failure case)
    print("-- baselines (feed linear straight into SDR VAE) --")
    for bl, E in [('lin-clip[0,1]', np.clip(img, 0, 1)),
                  ('lin/scaled', np.clip(img / max(L.max(), 1e-6), 0, 1))]:
        Ehat, _ = vae_roundtrip(vae, E.astype(np.float32))
        cp = psnr01(E.astype(np.float32), Ehat)
        # for lin-clip the >1 info is already gone; for lin/scaled recover by *max
        if bl == 'lin/scaled':
            Lr = lum(Ehat * L.max()); be = band_relerr(L, Lr)
            fmt = lambda k: ('%.1f%%' % (100 * be[k])) if be.get(k) is not None else '  .'
            print(f"  {bl:14s} codePSNR={cp:6.2f}  high={fmt('high(1-50)')} "
                  f"bright={fmt('bright(50-500)')} sun={fmt('sun(>500)')}")
        else:
            print(f"  {bl:14s} codePSNR={cp:6.2f}  (>1 radiance discarded before VAE)")
        results[f'baseline:{bl}'] = dict(code_psnr=round(cp, 2))
    return results

def main():
    t = time.time()
    vae = AutoencoderKLQwenImage.from_pretrained(
        'Qwen/Qwen-Image', subfolder='vae', torch_dtype=torch.float32).to(DEV).eval()
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
    print(f"VAE loaded ({time.time()-t:.1f}s), tiling on. z_dim={vae.config['z_dim']}")
    allres = {}
    for path, tag in [('data/kloofendal_43d_clear_2k.exr', 'PolyHaven kloofendal (real sun 111k)'),
                      ('data/s2r_scene0_0000.exr', 'S2R-HDR scene_0_FM/0000 (no sun, 21x)')]:
        try:
            allres[tag] = run_image(vae, load_exr_rgb(path), tag)
        except Exception as e:
            print(f"[skip {tag}] {type(e).__name__}: {e}")
    with open('out/vae_roundtrip_results.json', 'w') as f:
        json.dump(allres, f, indent=2)
    print("\n[saved] out/vae_roundtrip_results.json")

if __name__ == '__main__':
    main()
