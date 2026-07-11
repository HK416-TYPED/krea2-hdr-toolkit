"""
HDR transfer-function encodings for the Krea2->scene-linear-HDR pipeline.

Every curve maps scene-linear radiance L (>=0, mid-gray=0.18) into a bounded code
V in ~[0,1] that a frozen SDR VAE can accept, and back. Forward/inverse are exact
numerical inverses (validated in test_invertibility.py).

Two families:
  * scene-referred camera logs (Log3G10, LogC3, LogC4): input is scene-linear radiance.
  * absolute perceptual (PU21, PQ): input is absolute luminance in cd/m^2; we map
    scene-linear -> nits with a reference (default: 1.0 scene-linear = 100 nits).

Refs (see 2026-07-08 research report, Appendix A for constants & sources).
"""
import numpy as np

# scene-linear -> absolute nits reference (X2HDR-style: diffuse white ~100 nits)
SCENE_TO_NITS = 100.0


# ----------------------------------------------------------------------------
# RED Log3G10  (18% gray -> ~1/3, ~10 stops over gray inside [0,1], linear near 0)
# ----------------------------------------------------------------------------
_L3_a, _L3_b, _L3_c, _L3_g = 0.224282, 155.975327, 0.01, 15.1927

def log3g10_fwd(L):
    x = np.asarray(L, dtype=np.float64) + _L3_c
    logarg = np.maximum(x * _L3_b + 1.0, 1e-12)          # guard unused negative branch
    return np.where(x < 0.0, x * _L3_g, _L3_a * np.log10(logarg))

def log3g10_inv(V):
    V = np.asarray(V, dtype=np.float64)
    lin = np.where(V < 0.0, V / _L3_g, (np.power(10.0, V / _L3_a) - 1.0) / _L3_b)
    return lin - _L3_c


# ----------------------------------------------------------------------------
# ARRI LogC3 (EI800)
# ----------------------------------------------------------------------------
_C3 = dict(a=5.555556, b=0.052272, c=0.247190, d=0.385537,
           e=5.367655, f=0.092809, cut=0.010591)

def logc3_fwd(L):
    L = np.asarray(L, dtype=np.float64)
    p = _C3
    logarg = np.maximum(p['a'] * L + p['b'], 1e-12)      # guard unused negative branch
    return np.where(L > p['cut'],
                    p['c'] * np.log10(logarg) + p['d'],
                    p['e'] * L + p['f'])

def logc3_inv(V):
    V = np.asarray(V, dtype=np.float64)
    p = _C3
    thr = p['e'] * p['cut'] + p['f']
    return np.where(V > thr,
                    (np.power(10.0, (V - p['d']) / p['c']) - p['b']) / p['a'],
                    (V - p['f']) / p['e'])


# ----------------------------------------------------------------------------
# ARRI LogC4 (2022 spec) -- more highlight headroom than LogC3
# ----------------------------------------------------------------------------
_C4_a = (2.0**18 - 16.0) / 117.45
_C4_b = (1023.0 - 95.0) / 1023.0
_C4_c = 95.0 / 1023.0
_C4_s = (7.0 * np.log(2.0) * 2.0**(7.0 - 14.0 * _C4_c / _C4_b)) / (_C4_a * _C4_b)
_C4_t = (2.0**(14.0 * (-_C4_c / _C4_b) + 6.0) - 64.0) / _C4_a

def logc4_fwd(L):
    L = np.asarray(L, dtype=np.float64)
    hi = (np.log2(np.maximum(_C4_a * L + 64.0, 1e-12)) - 6.0) / 14.0 * _C4_b + _C4_c
    lo = (L - _C4_t) / _C4_s
    return np.where(L >= _C4_t, hi, lo)

def logc4_inv(V):
    V = np.asarray(V, dtype=np.float64)
    hi = (np.power(2.0, 14.0 * (V - _C4_c) / _C4_b + 6.0) - 64.0) / _C4_a
    lo = V * _C4_s + _C4_t
    return np.where(V >= _C4_c, hi, lo)


# ----------------------------------------------------------------------------
# PU21 (X2HDR quadratic-in-log2 form). Operates on absolute cd/m^2 in [0.005,1e4].
# Output PU units normalized by Pmax to [0,1].
# ----------------------------------------------------------------------------
_PU_a, _PU_b = 0.001908, 0.0078
_PU_Lmin = np.log2(0.005)
_PU_Lmax_lum = 10000.0

def _pu21_units(Lnits):
    u = np.log2(np.clip(Lnits, 0.005, _PU_Lmax_lum)) - _PU_Lmin
    return _PU_a * u * u + _PU_b * u

PU21_PMAX = float(_pu21_units(_PU_Lmax_lum))  # ~595

def pu21_fwd(L, scene_to_nits=SCENE_TO_NITS):
    Lnits = np.asarray(L, dtype=np.float64) * scene_to_nits
    return _pu21_units(Lnits) / PU21_PMAX

def pu21_inv(V, scene_to_nits=SCENE_TO_NITS):
    Vpu = np.asarray(V, dtype=np.float64) * PU21_PMAX
    disc = np.sqrt(np.maximum(_PU_b * _PU_b + 4.0 * _PU_a * Vpu, 0.0))
    u = (-_PU_b + disc) / (2.0 * _PU_a)
    Lnits = np.power(2.0, u + _PU_Lmin)
    return Lnits / scene_to_nits


# ----------------------------------------------------------------------------
# PQ / SMPTE ST 2084 (inverse-EOTF: linear nits -> code). Ceiling 10000 nits.
# ----------------------------------------------------------------------------
_PQ_m1, _PQ_m2 = 0.1593017578125, 78.84375
_PQ_c1, _PQ_c2, _PQ_c3 = 0.8359375, 18.8515625, 18.6875

def pq_fwd(L, scene_to_nits=SCENE_TO_NITS):
    Y = np.clip(np.asarray(L, dtype=np.float64) * scene_to_nits / 10000.0, 0.0, 1.0)
    Ym = np.power(Y, _PQ_m1)
    return np.power((_PQ_c1 + _PQ_c2 * Ym) / (1.0 + _PQ_c3 * Ym), _PQ_m2)

def pq_inv(V, scene_to_nits=SCENE_TO_NITS):
    Vm = np.power(np.clip(np.asarray(V, dtype=np.float64), 0.0, 1.0), 1.0 / _PQ_m2)
    num = np.maximum(Vm - _PQ_c1, 0.0)
    den = _PQ_c2 - _PQ_c3 * Vm
    Y = np.power(num / den, 1.0 / _PQ_m1)
    return Y * 10000.0 / scene_to_nits


# ----------------------------------------------------------------------------
# asinh / Lupton stretch (astronomy): linear near black, log at top.
# ----------------------------------------------------------------------------
def asinh_fwd(L, beta=0.02, Lmax=512.0):
    L = np.asarray(L, dtype=np.float64)
    return np.arcsinh(L / beta) / np.arcsinh(Lmax / beta)

def asinh_inv(V, beta=0.02, Lmax=512.0):
    V = np.asarray(V, dtype=np.float64)
    return beta * np.sinh(V * np.arcsinh(Lmax / beta))


# ----------------------------------------------------------------------------
# DiffHDR Log-Gamma  (bounded [0,1], knobs g and M)
# ----------------------------------------------------------------------------
def loggamma_fwd(L, g=10.0, M=512.0):
    L = np.asarray(L, dtype=np.float64)
    return np.power(np.log(1.0 + g * np.maximum(L, 0.0)) / np.log(1.0 + g * M), 1.0 / g)

def loggamma_inv(V, g=10.0, M=512.0):
    V = np.asarray(V, dtype=np.float64)
    return (np.exp(np.power(np.clip(V, 0.0, 1.0), g) * np.log(1.0 + g * M)) - 1.0) / g


# ----------------------------------------------------------------------------
# mu-law companding (audio). Input normalized by Lmax to [0,1] first.
# ----------------------------------------------------------------------------
def mulaw_fwd(L, mu=255.0, Lmax=512.0):
    x = np.clip(np.asarray(L, dtype=np.float64) / Lmax, 0.0, 1.0)
    return np.log1p(mu * x) / np.log1p(mu)

def mulaw_inv(V, mu=255.0, Lmax=512.0):
    V = np.clip(np.asarray(V, dtype=np.float64), 0.0, 1.0)
    return (np.power(1.0 + mu, V) - 1.0) / mu * Lmax


# ----------------------------------------------------------------------------
# HLG — Hybrid Log-Gamma (BT.2100 / ARIB STD-B67).
# Operates on scene-linear normalized to [0,1] by a peak. Its OETF is sqrt in the
# shadows and log in the highlights. Ceiling = HLG_PEAK (scene-linear at code 1.0).
# HLG was designed for display HDR (~modest highlight over diffuse white), so its
# headroom-vs-mid-gray tradeoff differs from a camera log — see test_hlg_vs_logc4.py.
# ----------------------------------------------------------------------------
_HLG_a, _HLG_b, _HLG_c = 0.17883277, 0.28466892, 0.55991073
HLG_PEAK = 12.0   # scene-linear value mapped to code 1.0 (white 1.0 -> code ~0.5)

def hlg_fwd(L, peak=HLG_PEAK):
    E = np.clip(np.asarray(L, dtype=np.float64) / peak, 0.0, 1.0)
    lo = np.sqrt(3.0 * E)
    hi = _HLG_a * np.log(np.maximum(12.0 * E - _HLG_b, 1e-12)) + _HLG_c
    return np.where(E <= 1.0 / 12.0, lo, hi)

def hlg_inv(V, peak=HLG_PEAK):
    V = np.clip(np.asarray(V, dtype=np.float64), 0.0, 1.0)
    lo = (V * V) / 3.0
    hi = (np.exp((V - _HLG_c) / _HLG_a) + _HLG_b) / 12.0
    E = np.where(V <= 0.5, lo, hi)
    return E * peak


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
CURVES = {
    'HLG':      (hlg_fwd,     hlg_inv),
    'Log3G10':  (log3g10_fwd, log3g10_inv),
    'LogC3':    (logc3_fwd,   logc3_inv),
    'LogC4':    (logc4_fwd,   logc4_inv),
    'PU21':     (pu21_fwd,    pu21_inv),
    'PQ':       (pq_fwd,      pq_inv),
    'asinh':    (asinh_fwd,   asinh_inv),
    'LogGamma': (loggamma_fwd, loggamma_inv),
    'mu-law':   (mulaw_fwd,   mulaw_inv),
}

# The peak scene-linear radiance each curve maps to code V=1.0 (its "ceiling").
def curve_ceiling(name):
    fwd, inv = CURVES[name]
    return float(inv(np.array([1.0]))[0])

if __name__ == '__main__':
    print(f"PU21_PMAX = {PU21_PMAX:.3f}")
    print(f"{'curve':10s} {'code@0.18':>10s} {'ceiling(V=1)':>14s}")
    for n in CURVES:
        fwd, inv = CURVES[n]
        print(f"{n:10s} {float(fwd(np.array([0.18]))[0]):10.4f} {curve_ceiling(n):14.3g}")
