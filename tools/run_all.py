"""Run the CPU (no-GPU) validation suite. GPU steps (real VAE) are skipped if torch/diffusers absent."""
import subprocess, sys
STEPS = [
    ("Encoding sanity (mid-gray + ceilings)", "hdr_encodings.py"),
    ("Invertibility + precision + banding",   "test_precision.py"),
    ("EXR round-trip + failure modes",        "test_exr_roundtrip.py"),
    ("Dataset prep + real-data restoration",  "dataset_prep.py"),
    ("HLG vs LogC4 comparison",               "test_hlg_vs_logc4.py"),
]
for title, script in STEPS:
    print("\n" + "#" * 90 + f"\n# {title}\n" + "#" * 90)
    if subprocess.run([sys.executable, script]).returncode != 0:
        print(f"!! {script} failed (may need optional GPU deps); continuing")
print("\nDone. Artifacts in out/.")
