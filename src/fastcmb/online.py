"""
An emulator that learns while a chain runs.

It holds every point the Boltzmann code computed, and refits on them as
they accumulate: the principal components of each block
(:mod:`fastcmb.compress`), then a Gaussian process per block
(:mod:`fastcmb.gp`). A prediction comes with its error in units of the
log-likelihood (:mod:`fastcmb.metric`); the caller decides from it
whether to trust the prediction or run the code.

Training points can be kept on disk. Each process writes its own file,
``<path>.<pid>.npz``, and reads every other one's when it refits -- so
chains running in parallel learn from each other's points, and a later
run of the same model starts from all of them.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

import numpy as np

from .compress import BLOCKS, PCA, from_blocks
from .gp import GP
from .metric import fisher_weights


__all__ = ["OnlineEmulator", "Prediction"]

log = logging.getLogger("fastcmb")


@dataclass
class Prediction:
    blocks: dict
    #: Predicted error of ``ln L``, ``sqrt(delta^T F delta)``.
    error: float
    #: Largest predicted relative error of the two ``sigma8``.
    scalar_error: float


class OnlineEmulator:
    """
    Parameters
    ----------
    names : list of str
        The inputs, in the order of the vectors given to :meth:`add`
        and :meth:`predict`.
    lmax : int
        Highest multipole of the spectra.
    tag : str
        What the spectra were computed for -- the model, the accuracy.
        Points stored under another tag are not read.
    min_train : int
        Points before the first fit.
    retrain_fraction : float
        Refit once the points added since the last fit are this
        fraction of those it was fitted on (and at least 5).
    max_train : int
        Most points fitted on; past it, the earliest are left out --
        those of the burn-in, furthest from where the chain is.
    path : str, optional
        Where to keep the points, without the extension.
    """

    def __init__(self, names, lmax: int, tag: str = "", min_train: int = 30,
                 retrain_fraction: float = 0.1, max_train: int = 800, path: str | None = None):

        self.names = list(names)
        self.lmax = int(lmax)
        self.tag = str(tag)
        self.min_train = int(min_train)
        self.retrain_fraction = float(retrain_fraction)
        self.max_train = int(max_train)
        self.path = path

        self.X: list[np.ndarray] = []
        #: Whether each point was computed by this process.
        self._own: list[bool] = []
        self.Y: dict[str, list[np.ndarray]] = {b: [] for b in BLOCKS}

        self.trained_on = 0
        #: Points held when the kernels were last optimized.
        self.optimized_at = 0
        self.n_fits = 0
        self._model = None
        self._theta: dict[str, dict] = {}
        self._others_seen: dict[str, float] = {}

        if path is not None:
            self._read_others()
            if self.size >= self.min_train:
                self.fit()

    # ---------------------------------------------------------
    # Points
    # ---------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.X)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def add(self, x, blocks: dict) -> None:
        """A point the Boltzmann code computed, refitting if it is time."""

        self.X.append(np.asarray(x, dtype=float))
        self._own.append(True)

        for b in BLOCKS:
            self.Y[b].append(np.asarray(blocks[b], dtype=np.float32))

        if self._due():
            self.fit()

    def _due(self) -> bool:

        if self.size < self.min_train:
            return False

        new = self.size - self.trained_on

        return not self.ready or new >= max(5, self.retrain_fraction * self.trained_on)

    # ---------------------------------------------------------
    # Fitting
    # ---------------------------------------------------------

    def fit(self, optimize: bool | None = None) -> None:

        if self.path is not None:
            self._read_others()
            self._write()

        X = np.array(self.X[-self.max_train:])
        Y = {b: np.array(self.Y[b][-self.max_train:], dtype=float) for b in BLOCKS}

        n = X.shape[0]

        if optimize is None:
            # Kernel hyperparameters change slowly once there are enough
            # points; they are refitted when half the points fitted on
            # are new -- also once the set is capped at max_train.
            optimize = n < 100 or self.size - self.optimized_at >= 0.5 * n

        center = X.mean(0)
        scale = X.std(0)

        # An input that has not varied cannot be learned; a point that
        # differs from it in that input is not predicted.
        varying = scale > 1e-12 * np.maximum(1.0, np.abs(center))

        Xs = (X[:, varying] - center[varying]) / scale[varying]

        mean_Cl, *_ = from_blocks({b: Y[b].mean(0) for b in BLOCKS}, self.lmax)
        F = fisher_weights(mean_Cl)
        F["scalars"] = None

        model = {"center": center, "scale": scale, "varying": varying, "blocks": {}}

        for b in BLOCKS:

            pca = PCA(Y[b], weights=F[b] if F[b] is not None and F[b].any() else None)
            coefficients = pca.project(Y[b])

            # Weight of each component in ln L, per unit coefficient.
            G = None if F[b] is None else (pca.components ** 2 * F[b]).sum(1)

            weights = coefficients.var(0) * (G if G is not None and G.any() else 1.0)

            gp = GP()
            gp.theta = self._theta.get(b, {})
            gp.fit(Xs, coefficients, optimize=optimize, weights=weights)

            self._theta[b] = gp.theta

            model["blocks"][b] = (pca, gp, G)

        self._model = model
        self.trained_on = self.size
        self.n_fits += 1

        if optimize:
            self.optimized_at = self.size

        log.info(
            "fastcmb: fitted on %d points (%s components)%s", n,
            ", ".join(f"{b} {model['blocks'][b][0].size}" for b in BLOCKS),
            ", kernel refitted" if optimize else "",
        )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(self, x, tolerance: float | None = None,
                scalar_tolerance: float | None = None) -> Prediction | None:
        """
        ``None`` if not fitted yet, or if ``x`` changes an input never
        varied.

        The error is an upper bound. Each kernel's contribution to it is
        bounded, at no cost, by its prior variance; the exact variance
        -- a triangular solve -- is computed for the kernels in order of
        that bound, and only while it matters. Given a ``tolerance``,
        the solves stop once the bounds left are below a fifth of it,
        or once the error is certainly above it. ``scalar_tolerance``
        does the same for ``sigma8``. Without them every variance is
        computed.
        """

        if self._model is None:
            return None

        m = self._model
        x = np.asarray(x, dtype=float)

        fixed = ~m["varying"]
        if np.any(np.abs(x[fixed] - m["center"][fixed]) > 1e-10 * np.maximum(1.0, np.abs(m["center"][fixed]))):
            return None

        xs = (x[m["varying"]] - m["center"][m["varying"]]) / m["scale"][m["varying"]]

        blocks = {}
        # (bound on the contribution to error^2, kernel, its vector, weights)
        candidates = []
        scalar_error = 0.0

        for b, (pca, gp, G) in m["blocks"].items():

            mean, ks = gp.mean(xs)
            blocks[b] = pca.reconstruct(mean)

            for (columns, kernel), k in zip(gp.parts, ks):

                prior = kernel.prior_variance
                if kernel is gp.shared:
                    prior = prior[columns]

                if b == "scalars":
                    bound = float(np.sqrt(prior.max()))
                    if scalar_tolerance is None or bound > scalar_tolerance:
                        var = kernel.variance(k)
                        var = var[columns] if kernel is gp.shared else var
                        bound = float(np.sqrt(var.max()))
                    scalar_error = max(scalar_error, bound)
                    continue

                if G is None or not G[columns].any():
                    continue

                candidates.append((float(prior @ G[columns]), kernel, k, columns, G[columns], kernel is gp.shared))

        candidates.sort(key=lambda c: -c[0])

        remaining = sum(c[0] for c in candidates)
        exact = 0.0

        for bound, kernel, k, columns, weights, shared in candidates:

            if tolerance is not None and (remaining <= (0.2 * tolerance) ** 2 or exact > tolerance ** 2):
                break

            var = kernel.variance(k)
            exact += float((var[columns] if shared else var) @ weights)
            remaining -= bound

        error = float(np.sqrt(exact + max(remaining, 0.0)))

        return Prediction(blocks=blocks, error=error, scalar_error=scalar_error)

    def error_between(self, blocks: dict, truth: dict) -> float:
        """The ``ln L`` error of ``blocks`` against ``truth``, by the same measure."""

        mean_Cl, *_ = from_blocks(truth, self.lmax)
        F = fisher_weights(mean_Cl)

        return float(np.sqrt(sum(
            ((np.asarray(blocks[b]) - np.asarray(truth[b])) ** 2 @ F[b]) for b in F
        )))

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    def _own_file(self) -> str:
        return f"{self.path}.{os.getpid()}.npz"

    def _write(self) -> None:
        """This process's own points, to its own file."""

        target = self._own_file()
        directory = os.path.dirname(target)
        if directory:
            os.makedirs(directory, exist_ok=True)

        rows = np.flatnonzero(self._own)

        temporary = target[:-4] + ".tmp.npz"
        np.savez(
            temporary,
            tag=self.tag, names=np.array(self.names), lmax=self.lmax,
            X=np.array(self.X)[rows].reshape(rows.size, len(self.names)),
            **{f"Y_{b}": np.array(self.Y[b])[rows] for b in BLOCKS},
        )
        os.replace(temporary, target)

    def _read_others(self) -> None:
        """Points other processes, or earlier runs, stored under the same tag."""

        own = os.path.abspath(self._own_file())

        for name in sorted(glob.glob(f"{self.path}.*.npz")):

            name = os.path.abspath(name)

            if name.endswith(".tmp.npz") or name == own:
                continue

            stamp = os.path.getmtime(name)
            if self._others_seen.get(name) == stamp:
                continue

            try:
                with np.load(name, allow_pickle=False) as data:
                    same = (str(data["tag"]) == self.tag and list(data["names"]) == self.names
                            and int(data["lmax"]) == self.lmax)
                    if same:
                        X = data["X"]
                        Y = {b: data[f"Y_{b}"] for b in BLOCKS}
            except (OSError, KeyError, ValueError) as error:
                log.warning("fastcmb: could not read %s (%s).", name, error)
                continue

            self._others_seen[name] = stamp

            if not same:
                log.warning("fastcmb: %s is for another model or accuracy; not read.", name)
                continue

            known = {row.tobytes() for row in self.X}

            for i, row in enumerate(X):
                if row.tobytes() not in known:
                    self.X.append(np.asarray(row, dtype=float))
                    self._own.append(False)
                    for b in BLOCKS:
                        self.Y[b].append(Y[b][i])
