"""Time CosmoFit's CAMB theory per point on Planck lite + lowE + lensing."""

import sys
import time
import warnings

import numpy as np

from cosmofit.core import get_model

PLANCK = {"H0": 67.36, "Omega_m": 0.3153, "Omega_b": 0.02237 / 0.6736 ** 2, "m_nu": 0.06}

theory = sys.argv[1] if len(sys.argv) > 1 else "camb"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

warnings.simplefilter("ignore")

info = {
    "theory": {"background": None, theory: None},
    "likelihood": {"cmb.planck_lite": None, "cmb.planck_lowe": None, "cmb.planck_lensing": None},
    "params": {
        "H0": {"prior": {"min": 60, "max": 75}},
        "Omega_m": {"prior": {"min": 0.25, "max": 0.4}},
        "Omega_b": PLANCK["Omega_b"], "m_nu": 0.06,
    },
}

m = get_model(info)
rng = np.random.default_rng(0)
times = []
for _ in range(n):
    p = {"H0": rng.uniform(65, 70), "Omega_m": rng.uniform(0.3, 0.33)}
    t = time.perf_counter()
    r = m.logposterior(p)
    times.append(time.perf_counter() - t)
    print(f"{p} logpost={r.logpost:.3f} {times[-1]:.3f}s")
print(f"median {np.median(times):.3f}s")
