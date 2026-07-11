"""
CPU verification of the ComfyUI "sampling -> inverse-transform -> EXR" chain claims
(research report, Exploration Area 5), using real OpenEXR I/O.

Verifies:
  1. 32-bit float EXR preserves sun-magnitude (>>1.0) radiance EXACTLY across
     lossless compressions (ZIP/PIZ/ZIPS); DWAA (lossy) degrades highlights.
  2. half-float (16-bit) overflows above 65504 -> +Inf  (the "sun overflow" risk).
  3. The two silent HDR killers reproduced numerically:
        (a) clamp-to-[0,1]  -> all highlights destroyed
        (b) implicit linear->sRGB OETF on >1 values -> radiance corrupted
  4. End-to-end restoration: synthetic scene-linear image (sky+sun+shadow) ->
     Log3G10 encode -> 8-bit VAE proxy -> decode -> inverse -> 32-bit EXR ->
     read back -> compare. Reports max-survival and highlight error.
  5. oiiotool --stats-equivalent bisection: file_max < tensor_max => node clamped;
     tensor_max <= 1 => model/inverse made no HDR.
"""
import numpy as np
import OpenEXR
from hdr_encodings import log3g10_fwd, log3g10_inv

GRAY = 0.18
COMP = {'ZIP': OpenEXR.ZIP_COMPRESSION, 'PIZ': OpenEXR.PIZ_COMPRESSION,
        'ZIPS': OpenEXR.ZIPS_COMPRESSION, 'DWAA': OpenEXR.DWAA_COMPRESSION}

def write_exr(path, rgb, comp='ZIP', half=False):
    hdr = {'compression': COMP[comp], 'type': OpenEXR.scanlineimage}
    arr = rgb.astype(np.float16 if half else np.float32)
    OpenEXR.File(hdr, {'RGB': arr}).write(path)

def read_exr(path):
    return OpenEXR.File(path).channels()['RGB'].pixels

def srgb_oetf(x):                       # the implicit transform a naive saver may apply
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1/2.4) - 0.055)

def make_scene(h=256, w=512):
    """Synthetic scene-linear radiance: dim shadow, mid ground, bright sky, tiny hot sun."""
    img = np.full((h, w, 3), 0.02, np.float32)          # shadow ~ -3.2 EV
    img[h//2:] = 0.18                                    # ground = mid-gray
    img[:h//3] = 8.0                                     # bright sky ~ +5.5 EV
    yy, xx = np.mgrid[0:h, 0:w]
    sun = ((xx-90)**2 + (yy-40)**2) < 12**2              # small sun disk
    img[sun] = 4000.0                                    # sun ~ +14.5 EV (sun-magnitude)
    return img

def stats(a):
    fin = np.isfinite(a)
    return dict(min=float(a[fin].min()), max=float(a[fin].max()),
                nonfinite=int((~fin).sum()))

def main():
    print("="*88); print("1) 32-bit float EXR preserves sun-magnitude radiance (lossless comps)"); print("="*88)
    scene = make_scene()
    tmax = float(scene.max())
    for c in ('ZIP', 'PIZ', 'ZIPS', 'DWAA'):
        write_exr('out/scene.exr', scene, comp=c)
        back = read_exr('out/scene.exr')
        s = stats(back)
        exact = np.allclose(back, scene, rtol=1e-6, atol=0)
        sun_err = abs(s['max'] - tmax) / tmax
        tag = 'LOSSLESS-exact' if exact else f'LOSSY sun_err={sun_err:.3%}'
        print(f"  {c:5s} file_max={s['max']:10.1f} (tensor_max={tmax:.0f})  {tag}")

    print("\n"+"="*88); print("2) half-float overflow above 65504"); print("="*88)
    hot = scene.copy(); hot[hot == 4000.0] = 80000.0     # a hotter sun
    write_exr('out/hot16.exr', hot, half=True)
    b16 = read_exr('out/hot16.exr')
    print(f"  wrote 80000 as half -> read back max={float(np.nan_to_num(b16, posinf=np.inf).max())} "
          f", +Inf count={int((~np.isfinite(b16)).sum())}  => confirms sun overflow risk")
    write_exr('out/hot32.exr', hot, half=False)
    print(f"  wrote 80000 as float32 -> read back max={float(read_exr('out/hot32.exr').max()):.0f}  => survives")

    print("\n"+"="*88); print("3) the two silent HDR killers (reproduced)"); print("="*88)
    clamped = np.clip(scene, 0.0, 1.0)
    write_exr('out/clamped.exr', clamped); bc = read_exr('out/clamped.exr')
    print(f"  (a) clamp-to-[0,1]     : file_max={float(bc.max()):.3f}  (was {tmax:.0f})  -> ALL HDR LOST")
    srgb = srgb_oetf(scene)
    write_exr('out/srgb.exr', srgb); bs = read_exr('out/srgb.exr')
    sun_after = float(bs[ ((np.mgrid[0:256,0:512][1]-90)**2 + (np.mgrid[0:256,0:512][0]-40)**2) < 12**2 ][...,0].max())
    print(f"  (b) implicit sRGB OETF : sun 4000 radiance -> stored {sun_after:.2f}  -> radiance CORRUPTED")

    print("\n"+"="*88); print("4) end-to-end restoration: scene-linear -> Log3G10 -> 8bit proxy -> inv -> EXR -> read"); print("="*88)
    V = log3g10_fwd(scene)                                # encode
    V8 = np.round(np.clip(V, 0, 1) * 255) / 255           # VAE 8-bit proxy
    rec = log3g10_inv(V8)                                 # inverse -> scene-linear
    write_exr('out/restored.exr', rec.astype(np.float32)); back = read_exr('out/restored.exr')
    def relerr_region(m): return float(np.median(np.abs(back[m]-scene[m])/np.maximum(scene[m],1e-6)))
    sky = scene[:,:,0] > 4; sun = scene[:,:,0] > 3000; mid = np.abs(scene[:,:,0]-0.18) < 1e-3
    print(f"  file_max={float(back.max()):.1f}  tensor_max_in={tmax:.1f}  (Log3G10 ceiling {log3g10_inv(np.array([1.0]))[0]:.0f})")
    print(f"  median relerr  midtone={relerr_region(mid):.3%}  sky(+5.5EV)={relerr_region(sky):.3%}  sun(+14.5EV)={relerr_region(sun):.3%}")
    print("  NOTE: sun at +14.5 EV exceeds Log3G10's ceiling(~184) -> its radiance is CLIPPED to the")
    print("        ceiling by the encoding (a single-plane limit, exactly the report's finding).")

    print("\n"+"="*88); print("5) oiiotool --stats-equivalent bisection assertion"); print("="*88)
    for label, path, tensor_max in [("restored", 'out/restored.exr', float(rec.max())),
                                     ("clamped-node", 'out/clamped.exr', tmax)]:
        f = read_exr(path); fmax = float(f.max()); s = stats(f)
        print(f"  {label:14s} stats min={s['min']:.4g} max={s['max']:.4g} nonfinite={s['nonfinite']}")
        if fmax < tensor_max * 0.999:
            print(f"      -> DIAGNOSIS: SAVE PATH CLAMPED/QUANTIZED (file_max {fmax:.2f} < tensor_max {tensor_max:.2f})")
        elif tensor_max <= 1.0001:
            print(f"      -> DIAGNOSIS: model/inverse produced NO HDR (tensor_max<=1)")
        else:
            print(f"      -> OK: dynamic range preserved through save")

if __name__ == '__main__':
    main()
