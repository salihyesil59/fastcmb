"""
Offline study of the emulator on saved CAMB points -- a training file
(``training_file``, one ``.npz`` per process):
fit on the first ``--train``, predict the rest, and break the error down
by block -- actual against predicted, and what the PCA alone loses.

    python benchmarks/study.py chains/emu/lcdm.1234.npz --train 200
"""

import argparse
import time

import numpy as np

from fastcmb.compress import BLOCKS, from_blocks
from fastcmb.metric import fisher_weights
from fastcmb.online import OnlineEmulator


parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("--train", type=int, default=200)
args = parser.parse_args()

data = np.load(args.data)
names = list(data["names"])
lmax = int(data["lmax"])
X = data["X"]
Y = {b: data[f"Y_{b}"].astype(float) for b in BLOCKS}
n = X.shape[0]
print(f"{n} points, inputs {names}")

e = OnlineEmulator(names, lmax, min_train=10 ** 9)
for i in range(args.train):
    e.add(X[i], {b: Y[b][i] for b in BLOCKS})

t = time.perf_counter()
e.fit()
print(f"fit: {time.perf_counter() - t:.1f} s")

for b, (pca, gp, G) in e._model["blocks"].items():
    print(f"  {b:8s} {pca.size:3d} comps, ell {np.round(np.exp(gp.log_ell), 2)}, nugget {np.exp(gp.log_nugget):.1e}")

actual = {b: [] for b in BLOCKS if b != "scalars"}
predicted = {b: [] for b in actual}
pca_only = {b: [] for b in actual}
total_actual, total_pred, times = [], [], []

for i in range(args.train, n):

    truth = {b: Y[b][i] for b in BLOCKS}
    t = time.perf_counter()
    p = e.predict(X[i])
    times.append(time.perf_counter() - t)

    Cl, *_ = from_blocks(truth, lmax)
    F = fisher_weights(Cl)
    m = e._model
    xs = (X[i][m["varying"]] - m["center"][m["varying"]]) / m["scale"][m["varying"]]

    for b in actual:
        pca, gp, G = m["blocks"][b]
        _, var = gp.predict(xs)
        actual[b].append(np.sqrt((p.blocks[b] - truth[b]) ** 2 @ F[b]))
        predicted[b].append(np.sqrt(var @ G) if G is not None else 0.0)
        r = pca.reconstruct(pca.project(truth[b])) - truth[b]
        pca_only[b].append(np.sqrt(r ** 2 @ F[b]))

    total_actual.append(e.error_between(p.blocks, truth))
    total_pred.append(p.error)

print(f"predict: {np.median(times) * 1e3:.2f} ms")
print(f"{'block':8s} {'actual med':>11s} {'actual max':>11s} {'pred med':>9s} {'pca med':>9s}")
for b in actual:
    print(f"{b:8s} {np.median(actual[b]):11.4f} {np.max(actual[b]):11.4f} "
          f"{np.median(predicted[b]):9.4f} {np.median(pca_only[b]):9.4f}")
ta, tp = np.array(total_actual), np.array(total_pred)
print(f"total    actual median {np.median(ta):.4f} max {ta.max():.4f}; predicted median {np.median(tp):.4f}")
print(f"fraction trusted at 0.05: {(tp <= 0.05).mean():.2f}, at 0.1: {(tp <= 0.1).mean():.2f}, at 0.2: {(tp <= 0.2).mean():.2f}")
