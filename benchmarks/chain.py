"""
A real chain: CosmoFit's MCMC on Planck 2018 lite + lowE + lensing,
LCDM, with ``fastcmb``. Then:

* the wall time, against what CAMB at every evaluation would have cost
  (CAMB's median time per point, measured here, times the evaluations);
* ``|Delta ln L|`` against CAMB on points drawn from the chain;
* the posterior.

    python benchmarks/chain.py --Rminus1 0.02 --max-time 3600
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings

import numpy as np

from cosmofit.core.run import run

from cosmofit.core import get_model

#: Planck 2018 TT,TE,EE+lowE+lensing: mean and sigma.
PLANCK = {
    "H0": (67.36, 0.54),
    "Omega_m": (0.3153, 0.0073),
    "Omega_b": (0.02237 / 0.6736 ** 2, 0.00045),
    "ln1e10As": (3.044, 0.014),
    "n_s": (0.9649, 0.0042),
    "tau_reio": (0.0544, 0.0073),
}

LIKELIHOODS = {"cmb.planck_lite": None, "cmb.planck_lowe": None, "cmb.planck_lensing": None}

#: A dark sector beyond LCDM: its parameters, as (prior min, max, ref mean, sigma).
#: ``log10_x`` is sampled and gives the sector ``x = 10**log10_x``.
DARK_ENERGY = {
    "lcdm": {},
    "cpl": {"w0": (-3.0, 1.0, -0.8, 0.1), "wa": (-3.0, 2.0, -0.6, 0.3)},
    # z_c over 1000-10000, uniform in its logarithm.
    "ede": {"f_ede": (0.0, 0.3, 0.01, 0.005), "log10_z_c": (3.0, 4.0, 3.5, 0.1)},
}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--Rminus1", type=float, default=0.02)
    parser.add_argument("--max-time", type=float, default=3600)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--validate", type=int, default=60)
    parser.add_argument("--training-file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--dark-energy", choices=sorted(DARK_ENERGY), default="lcdm")
    parser.add_argument("--bao", action="store_true", help="add DESI BAO")
    parser.add_argument("--covmat", default=None, help="proposal covariance to start from")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    params = {
        name: {
            "prior": {"min": mean - 20 * sigma, "max": mean + 20 * sigma},
            "ref": {"dist": "norm", "loc": mean, "scale": sigma},
            "proposal": sigma,
        }
        for name, (mean, sigma) in PLANCK.items()
    }
    params.update(
        m_nu=0.06,
        A_planck={"prior": {"min": 0.9, "max": 1.1}, "ref": {"dist": "norm", "loc": 1.0, "scale": 0.0025},
                  "proposal": 0.0025},
    )

    for name, (lo, hi, mean, sigma) in DARK_ENERGY[args.dark_energy].items():
        params[name] = {"prior": {"min": lo, "max": hi},
                        "ref": {"dist": "norm", "loc": mean, "scale": sigma}, "proposal": sigma}
        if name.startswith("log10_"):
            params[name]["drop"] = True
            params[name[len("log10_"):]] = f"lambda {name}: 10 ** {name}"

    likelihoods = dict(LIKELIHOODS, **({"bao.desi": None} if args.bao else {}))
    background = None if args.dark_energy == "lcdm" else {"dark_energy": args.dark_energy}
    # BAO needs r_d, from the early-universe theory.
    theories = {"background": background, **({"early_universe": None} if args.bao else {})}

    options = {"tolerance": args.tolerance}
    if args.training_file:
        options["training_file"] = args.training_file

    info = {
        "theory": {**theories, "fastcmb": options},
        "likelihood": likelihoods,
        "params": params,
        "sampler": {"mcmc": {"chains": 4, "Rminus1_stop": args.Rminus1, "max_time": args.max_time,
                              "processes": args.processes, "covmat": args.covmat}},
    }
    if args.output:
        info["output"] = args.output
        info["resume"] = args.resume

    start = time.perf_counter()
    _, sampler = run(info, seed=args.seed)
    wall = time.perf_counter() - start

    fast = sampler.model
    theory = fast.theories["fastcmb"]
    stats = theory.stats()
    products = sampler.products(skip=0.3)

    evaluations = stats["camb"] + stats["emulated"]

    # CAMB's time per point, at points of the chain -- with the same
    # parameters as the chain's (A_planck sampled, not fixed).
    slow = get_model({"theory": {**theories, "camb": None},
                      "likelihood": likelihoods, "params": params})
    rng = np.random.default_rng(1)
    samples, weights = products["samples"], products["weights"]
    picks = rng.choice(len(samples), size=args.validate, p=weights / weights.sum())

    diffs, camb_times = [], []
    for row in samples[picks]:
        point = dict(zip(sampler.names, row))
        t = time.perf_counter()
        b = slow.logposterior(point)
        camb_times.append(time.perf_counter() - t)
        a = fast.logposterior(point)
        if a.rejected is None and b.rejected is None:
            diffs.append(abs(a.loglike - b.loglike))

    diffs = np.array(diffs)
    t_camb = float(np.median(camb_times))

    print()
    print(f"converged: {products['converged']}, R-1 = {products['Rminus1']:.4f}, "
          f"acceptance {products['acceptance']}")
    print(f"samples in the chains: {len(products['samples'])} (after skipping 30%)")
    if args.processes > 1:
        print("(the counts below are the first process's)")
    print(f"evaluations: {evaluations} (CAMB {stats['camb']}, emulated {stats['emulated']}), "
          f"training points {stats['training_points']}, checks {stats['checks']} "
          f"(median error {stats['check_error_median']}, max {stats['check_error_max']})")
    if "screened" in products:
        out, inside = products["screened"]
        print(f"slow proposals screened: {out} rejected approximately, {inside} then computed exactly")
    print(f"wall time {wall / 60:.1f} min; CAMB at every evaluation: "
          f"{evaluations * t_camb / 60:.1f} min ({t_camb:.2f} s/point) -> {evaluations * t_camb / wall:.1f}x")
    print(f"|Delta ln L| vs CAMB on {diffs.size} chain points: median {np.median(diffs):.4f}, "
          f"95% {np.quantile(diffs, 0.95):.4f}, max {diffs.max():.4f}")

    print("\nposterior (fastcmb):")
    w = weights / weights.sum()
    for j, name in enumerate(sampler.names):
        mean = (w * samples[:, j]).sum()
        sd = np.sqrt((w * (samples[:, j] - mean) ** 2).sum())
        print(f"  {name:10s} {mean:.5f} +- {sd:.5f}")


if __name__ == "__main__":
    main()
