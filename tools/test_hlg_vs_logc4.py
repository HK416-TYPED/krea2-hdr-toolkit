"""
HLG (BT.2100 Hybrid Log-Gamma) vs LogC4 for the Krea2 scene-linear HDR pipeline.

Part 1 (CPU): invertibility, mid-gray placement, ceiling/headroom, codes/stop, banding.
Part 2 (GPU): real Qwen Image VAE round-trip on a real HDRI — code-domain PSNR and
              per-luminance-band restoration error (the metric that picked LogC4).

The question: does HLG carry scene-linear HDR into the frozen VAE as well as LogC4?
"""
import numpy as np
from hdr_encodings import (hlg_fwd, hlg_inv, logc4_fwd, logc4_inv)

GRAY = 0.18

def cpu_metrics(name, fwd, inv):
    L = GRAY * np.power(2.0, np.linspace(-8, 13, 300000)); ev = np.log2(L / GRAY)
    ceil = float(inv(np.array([1.0]))[0])
    work = (L <= ceil * 0.999)
    rt = float(np.max(np.abs(inv(fwd(L[work])) - L[work]) / np.maximum(L[work], 1e-9)))
    code_gray = float(fwd(np.array([GRAY]))[0])
    V = np.clip(fwd(L), 0, 1); L8 = inv(np.round(V * 255) / 255)
    re = np.abs(L8 - L) / np.maximum(L, 1e-6)
    m = (ev > 0) & (L <= ceil * 0.95); ov = np.where(re[m] > 0.05)[0]
    band = round(float(ev[m][ov[0]]), 2) if len(ov) else None
    def cps(t):
        if GRAY * 2**t > ceil: return 0.0
        hi = float(fwd(np.array([GRAY * 2**(t + .5)]))[0]); lo = float(fwd(np.array([GRAY * 2**(t - .5)]))[0])
        return round(abs(hi - lo) * 255, 1)
    return dict(name=name, code_gray=round(code_gray, 3), ceiling=round(ceil, 1),
                stops_over_gray=round(np.log2(ceil / GRAY), 1), roundtrip=rt,
                band_onset=band, cps2=cps(2), cps4=cps(4), cps6=cps(6))

def part1():
    print("=" * 96)
    print("PART 1  CPU metrics  (HLG at 3 peak normalizations vs LogC4)")
    print("=" * 96)
    rows = []
    for pk in (12.0, 50.0, 470.0):
        rows.append(cpu_metrics(f'HLG(peak={pk:g})', lambda L, p=pk: hlg_fwd(L, p), lambda V, p=pk: hlg_inv(V, p)))
    rows.append(cpu_metrics('LogC4', logc4_fwd, logc4_inv))
    print(f"{'curve':16s}{'code@0.18':>10s}{'ceiling':>9s}{'stopsUP':>8s}{'invErr':>9s}"
          f"{'band>5%':>8s}{'cps+2':>7s}{'cps+4':>7s}{'cps+6':>7s}")
    for r in rows:
        b = f"{r['band_onset']:.1f}" if r['band_onset'] is not None else "none"
        print(f"{r['name']:16s}{r['code_gray']:10.3f}{r['ceiling']:9.4g}{r['stops_over_gray']:8.1f}"
              f"{r['roundtrip']:9.1e}{b:>8s}{r['cps2']:7.1f}{r['cps4']:7.1f}{r['cps6']:7.1f}")
    print("\nread: on-manifold wants code@0.18 near sRGB 0.46; more 'stopsUP' = more headroom;")
    print("      HLG normalized for a reasonable mid-gray gives little headroom (its design target is")
    print("      display HDR, ~few stops over white), while pushing peak crushes mid-gray toward black.")

def part2():
    import torch, json, os
    from diffusers import AutoencoderKLQwenImage
    from dataset_prep import load_exr_rgb, lum
    exr = 'data/kloofendal_43d_clear_2k.exr'
    if not os.path.exists(exr):
        print("\n[PART 2 skipped] run dataset_prep.py first to fetch the test HDRI"); return
    print("\n" + "=" * 96)
    print("PART 2  Real Qwen VAE round-trip on a real HDRI (kloofendal): HLG(peak=12) vs LogC4")
    print("=" * 96)
    vae = AutoencoderKLQwenImage.from_pretrained('Qwen/Qwen-Image', subfolder='vae',
                                                 torch_dtype=torch.float32).to('cuda').eval()
    if hasattr(vae, 'enable_tiling'): vae.enable_tiling()
    img = np.clip(load_exr_rgb(exr), 0, None).astype(np.float32)
    h, w = img.shape[:2]; img = img[:h//16*16, :w//16*16]; L = lum(img)
    def rt(fw, iv, name):
        E = np.clip(fw(img), 0, 1).astype(np.float32)
        x = torch.from_numpy(E.transpose(2,0,1)[None,:,None]).float().cuda()*2-1
        with torch.no_grad():
            r = vae.decode(vae.encode(x).latent_dist.mode()).sample
        Eh = np.clip((r[0,:,0].permute(1,2,0).cpu().numpy()+1)/2, 0, 1)
        cp = 10*np.log10(1.0/max(float(np.mean((E-Eh)**2)), 1e-12))
        Lr = lum(iv(Eh))
        bands = {'shadow<0.05': L<0.05, 'mid0.05-1': (L>=0.05)&(L<1), 'high1-50': (L>=1)&(L<50),
                 'bright50-500': (L>=50)&(L<500)}
        be = {b: (None if m.sum()==0 else round(float(np.median(np.abs(Lr[m]-L[m])/np.maximum(L[m],1e-6))),4)) for b,m in bands.items()}
        ceil = float(iv(np.array([1.0]))[0])
        print(f"  {name:14s} codePSNR={cp:5.2f}dB  ceiling={ceil:6.0f}x  "
              f"shadow={be['shadow<0.05']} mid={be['mid0.05-1']} high={be['high1-50']} bright={be['bright50-500']}")
        return cp
    a = rt(lambda L: hlg_fwd(L, 12.0), lambda V: hlg_inv(V, 12.0), 'HLG(peak12)')
    a2 = rt(lambda L: hlg_fwd(L, 470.0), lambda V: hlg_inv(V, 470.0), 'HLG(peak470)')
    b = rt(logc4_fwd, logc4_inv, 'LogC4')
    print(f"\n  code-PSNR: LogC4 {b:.2f} vs HLG(12) {a:.2f} vs HLG(470) {a2:.2f}  "
          f"-> winner: {'LogC4' if b>=max(a,a2) else 'HLG'}")

if __name__ == '__main__':
    part1()
    try:
        part2()
    except Exception as e:
        print(f"\n[PART 2 error] {type(e).__name__}: {e}")
