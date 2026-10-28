"""
Check a finished chain against CAMB: points drawn from its files are
evaluated again with ``camb``, and their recorded ``chi2`` (from
``fastcmb``) compared, per likelihood. Also each chain's mean and
length, to see why ``R - 1`` is what it is.

    python benchmarks/validate_chain.py chains/lcdm_fast --points 80
"""

import argparse
import glob
import warnings

import numpy as np

from cosmofit.core import get_model

from chain import LIKELIHOODS, PLANCK


parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("--points", type=int, default=80)
parser.add_argument("--skip", type=float, default=0.3)
args = parser.parse_args()

warnings.simplefilter("ignore")

files = sorted(glob.glob(args.root + ".[0-9]*.txt"))
header = open(files[0]).readline().lstrip("#").split()
chains = [np.loadtxt(f) for f in files]

names = [c for c in header[2:] if c in PLANCK or c == "A_planck"]
chi2_columns = [c for c in header if c.startswith("chi2__")]

print("chain  rows  " + "  ".join(f"{n:>9s}" for n in names))
for i, c in enumerate(chains):
    c = c[int(args.skip * len(c)):]
    w = c[:, 0] / c[:, 0].sum()
    means = [(w * c[:, header.index(n)]).sum() for n in names]
    print(f"{i + 1:5d} {len(c):5d}  " + "  ".join(f"{m:9.5f}" for m in means))

params = {n: {"prior": {"min": -1e9, "max": 1e9}} for n in PLANCK}
params.update(m_nu=0.06, A_planck={"prior": {"min": 0.0, "max": 2.0}})
slow = get_model({"theory": {"background": None, "camb": None}, "likelihood": LIKELIHOODS, "params": params})

rows = np.vstack([c[int(args.skip * len(c)):] for c in chains])
rng = np.random.default_rng(0)
picks = rng.choice(len(rows), size=args.points, p=rows[:, 0] / rows[:, 0].sum())

diffs = []
for row in rows[picks]:
    point = {n: row[header.index(n)] for n in names}
    r = slow.logposterior(point)
    if r.rejected is not None:
        continue
    per = {c: row[header.index(c)] - (-2.0 * r.loglikes[c[6:].replace("_", ".", 1)]) for c in chi2_columns}
    diffs.append(per)

total = np.array([abs(sum(d.values())) / 2.0 for d in diffs])
print(f"\n|Delta ln L| vs CAMB on {total.size} chain points: median {np.median(total):.4f}, "
      f"95% {np.quantile(total, 0.95):.4f}, max {total.max():.4f}")
for c in chi2_columns:
    v = np.abs([d[c] for d in diffs]) / 2.0
    print(f"  {c[6:]:24s} median {np.median(v):.4f}  max {v.max():.4f}")
