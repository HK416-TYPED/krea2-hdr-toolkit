"""
CPU verification of the "linear mapping and restoration" claims for each encoding.

Part A -- exact invertibility at float64 (encode->decode with no loss), over a
          realistic working range (-10 EV .. curve ceiling).
Part B -- UNIFORM frozen-VAE precision proxy: model the VAE as reconstructing the
          [0,1] encoded image at ~8-bit fidelity everywhere (uniform quantization).
Part C -- banding-onset vs effective bit-depth under the uniform proxy.
Part D -- NON-UNIFORM proxy: the real Qwen/SDR VAE was trained on display-referred
          sRGB whose code values rarely sit sustained near 1.0, so its effective
          precision DEGRADES toward the top of [0,1]. We model recon noise whose LSB
          grows above a knee. This is the mechanism behind the observed highlight
          banding, and it quantifies WHY you must keep the peak below V~0.9.

Honest note: the VAE is NOT a real quantizer; these are first-order proxies. Part B
shows the ENCODING math alone is not the 4-6 stop wall; Part D shows the wall comes
from the VAE's non-uniform precision, which is what the schemes (VAE conv-LoRA /
16->32 / gain map) exist to fix.
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hdr_encodings import CURVES, curve_ceiling

GRAY = 0.18
FLOOR_EV = -10.0                     # realistic shadow floor (~0.0176 scene-linear)
RNG = np.random.default_rng(0)

def quantize_uniform(V, nbits):
    levels = 2 ** nbits - 1
    return np.round(np.clip(V, 0.0, 1.0) * levels) / levels

def quantize_nonuniform(V, nbits=8, knee=0.75, top_bits=5):
    """Effective bits fall linearly from `nbits` at knee to `top_bits` at V=1.0,
    reflecting a display-referred VAE under-serving the top of the code range."""
    Vc = np.clip(V, 0.0, 1.0)
    frac = np.clip((Vc - knee) / (1.0 - knee), 0.0, 1.0)
    eff_bits = nbits - frac * (nbits - top_bits)
    lsb = 1.0 / (2.0 ** eff_bits - 1.0)
    q = np.round(Vc / lsb) * lsb
    q = q + RNG.normal(0.0, 0.5 * lsb)            # ~0.5 LSB recon noise
    return np.clip(q, 0.0, 1.0)

def relerr(L, Lhat):
    return np.abs(Lhat - L) / np.maximum(np.abs(L), 1e-9)

def run():
    L = GRAY * np.power(2.0, np.linspace(FLOOR_EV, 20, 200000))
    ev = np.log2(L / GRAY)
    results = {}
    for name, (fwd, inv) in CURVES.items():
        ceil = curve_ceiling(name)
        work = (L >= GRAY * 2**FLOOR_EV) & (L <= ceil * 0.999)

        # Part A: invertibility on working range
        rt_err = float(np.max(relerr(L[work], inv(fwd(L[work])))))

        # Part B: uniform 8-bit
        V = fwd(L)
        L8 = inv(quantize_uniform(V, 8))
        re8 = relerr(L, L8)

        # midtone fidelity (relerr near gray, +/-2 EV) -- catches curves that crush mid
        mid = (ev >= -2) & (ev <= 2)
        mid_relerr = float(np.median(re8[mid]))

        # banding onset (uniform 8-bit): first EV>0 with relerr>5% inside range
        def onset(re):
            m = (ev > 0) & (L <= ceil * 0.95)
            ov = np.where(re[m] > 0.05)[0]
            return round(float(ev[m][ov[0]]), 2) if len(ov) else None
        onset8 = onset(re8)

        # codes per stop at +4/+6/+8 EV
        def cps(t):
            if GRAY * 2**t > ceil: return 0.0
            d = 0.5
            hi = float(fwd(np.array([GRAY * 2**(t + d)]))[0])
            lo = float(fwd(np.array([GRAY * 2**(t - d)]))[0])
            return round(abs(hi - lo) / (2*d) * 255, 1)
        codes = {f'+{e}': cps(e) for e in (4, 6, 8)}

        # Part C bit sweep (uniform)
        bsweep = {nb: onset(relerr(L, inv(quantize_uniform(V, nb)))) for nb in (6, 8, 10, 12)}

        # Part D: non-uniform VAE proxy -- for fixed highlight targets, V-placement and
        # mean relerr over many noise realizations (robust; validates "keep peak low").
        targets = {}
        for t in (2, 4, 6, 8, 10):
            Lt = GRAY * 2.0 ** t
            if Lt > ceil * 0.999:
                targets[f'+{t}'] = dict(V=None, relerr=None); continue
            Vt = float(fwd(np.array([Lt]))[0])
            draws = np.array([inv(quantize_nonuniform(np.array([Vt]), 8))[0] for _ in range(400)])
            targets[f'+{t}'] = dict(V=round(Vt, 3),
                                    relerr=round(float(np.mean(np.abs(draws - Lt) / Lt)), 4))

        results[name] = dict(
            ceiling=round(ceil, 1), code_at_gray=round(float(fwd(np.array([GRAY]))[0]), 4),
            roundtrip_max_relerr=rt_err, mid_relerr_8bit=mid_relerr,
            uniform_onset_EV=onset8, codes_per_stop=codes,
            uniform_onset_by_bits=bsweep, nonuniform_targets=targets,
        )
    return results, (L, ev)

def make_plot(res, axes_data):
    L, ev = axes_data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    show = ['Log3G10', 'LogC4', 'PU21', 'PQ', 'asinh', 'LogGamma']
    for name in show:
        fwd, inv = CURVES[name]
        V = fwd(L)
        m = (ev > -6) & (ev < 18)
        ax1.plot(ev[m], np.clip(relerr(L, inv(quantize_uniform(V, 8)))[m], 1e-4, 1), lw=1.4, label=name)
        ax2.plot(ev[m], np.clip(relerr(L, inv(quantize_nonuniform(V, 8)))[m], 1e-4, 1), lw=1.4, label=name)
    for ax, t in ((ax1, 'B: UNIFORM 8-bit VAE proxy'), (ax2, 'D: NON-UNIFORM VAE proxy (degrades near V=1)')):
        ax.axhline(0.05, color='k', ls='--', lw=0.8, alpha=0.6)
        ax.set_yscale('log'); ax.set_xlabel('stops above mid-gray (EV)')
        ax.set_ylabel('relative reconstruction error'); ax.set_title(t)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('Scene-linear restoration error after encode->[0,1] VAE-proxy->decode  (dashed = 5% banding threshold)')
    fig.tight_layout(); fig.savefig('out/precision_relerr.png', dpi=110)
    print('[saved] out/precision_relerr.png')

if __name__ == '__main__':
    res, axes = run()
    print("="*96)
    print("PART A  Exact invertibility (float64, working range -10EV..ceiling)")
    print("="*96)
    for n, r in res.items():
        print(f"{n:10s} max_rel_err={r['roundtrip_max_relerr']:.2e}  "
              f"{'PASS (exact)' if r['roundtrip_max_relerr']<1e-6 else 'FAIL'}")
    print("\n"+"="*96)
    print("PART B  UNIFORM 8-bit proxy: mid-fidelity, banding onset, code allocation")
    print("="*96)
    print(f"{'curve':10s}{'ceil':>8s}{'gray':>7s}{'midErr%':>9s}{'unifOnset':>11s}"
          f"{'cps+4':>7s}{'cps+6':>7s}{'cps+8':>7s}")
    for n, r in res.items():
        u = f"{r['uniform_onset_EV']:.1f}" if r['uniform_onset_EV'] is not None else "none"
        c = r['codes_per_stop']
        print(f"{n:10s}{r['ceiling']:8.4g}{r['code_at_gray']:7.3f}{100*r['mid_relerr_8bit']:9.2f}"
              f"{u:>11s}{c['+4']:7.1f}{c['+6']:7.1f}{c['+8']:7.1f}")
    print("\n"+"="*96)
    print("PART D  NON-UNIFORM proxy: highlight relerr% at fixed targets (mean of 400 draws)")
    print("        '.' = beyond curve ceiling.  Lower relerr = highlight better preserved.")
    print("="*96)
    print(f"{'curve':10s}" + "".join(f"{'+'+str(t)+'EV':>18s}" for t in (2,4,6,8,10)))
    print(f"{'':10s}" + "".join(f"{'(V / err%)':>18s}" for _ in range(5)))
    for n, r in res.items():
        row = f"{n:10s}"
        for t in (2,4,6,8,10):
            d = r['nonuniform_targets'][f'+{t}']
            cell = '.' if d['V'] is None else f"{d['V']:.2f}/{100*d['relerr']:.1f}"
            row += f"{cell:>18s}"
        print(row)
    print("\n"+"="*96)
    print("PART C  uniform-proxy banding-onset EV vs latent bit-depth")
    print("="*96)
    print(f"{'curve':10s}{'6-bit':>8s}{'8-bit':>8s}{'10-bit':>8s}{'12-bit':>8s}")
    for n, r in res.items():
        s = r['uniform_onset_by_bits']
        g = lambda x: f"{x:.1f}" if x is not None else "none"
        print(f"{n:10s}{g(s[6]):>8s}{g(s[8]):>8s}{g(s[10]):>8s}{g(s[12]):>8s}")
    with open('out/precision_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
    print("\n[saved] out/precision_results.json")
    make_plot(res, axes)
