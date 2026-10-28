"""
What is emulated, and how it is made smooth and small.

Each spectrum is a *block*. From ``l = 2``:

* ``tt``, ``ee``, ``bb``, ``pp``: ``ln C_l`` -- positive, and spanning
  orders of magnitude;
* ``te``: the correlation ``C_TE / sqrt(C_TT C_EE)``, which crosses zero
  where ``C_TE`` does and stays within [-1, 1];
* ``scalars``: ``ln sigma8`` (all matter) and ``ln sigma8`` of the cold
  matter.

A block of spectra is projected on its leading principal components:
as many as it takes to reproduce every training spectrum to
:data:`PCA_TOLERANCE` in each multipole and, where the block has
likelihood weights (:mod:`fastcmb.metric`), to :data:`PCA_LOGL` in
the log-likelihood.
"""

from __future__ import annotations

import numpy as np


__all__ = ["BLOCKS", "PCA", "PCA_TOLERANCE", "to_blocks", "from_blocks"]

BLOCKS = ("tt", "te", "ee", "bb", "pp", "scalars")

#: Largest error, over the training set, a block's PCA may make: in
#: ``ln C_l`` (so 1e-3 is 0.1 %) or in the TE correlation.
PCA_TOLERANCE = 1e-3

#: Largest log-likelihood error, ``sqrt(delta^T F delta)``, a block's
#: PCA may make on a training spectrum.
PCA_LOGL = 0.01

#: Most components a block keeps.
MAX_COMPONENTS = 60


def to_blocks(Cl: dict, sigma8: float, sigma8_cb: float) -> dict:
    """A CosmoFit ``Cl`` dict and the two ``sigma8`` as block vectors."""

    tt, ee, te = Cl["tt"][2:], Cl["ee"][2:], Cl["te"][2:]

    return {
        "tt": np.log(tt),
        "te": te / np.sqrt(tt * ee),
        "ee": np.log(ee),
        "bb": np.log(Cl["bb"][2:]),
        "pp": np.log(Cl["pp"][2:]),
        "scalars": np.log([sigma8, sigma8_cb]),
    }


def from_blocks(blocks: dict, lmax: int) -> tuple[dict, float, float]:
    """The inverse of :func:`to_blocks`: ``(Cl, sigma8, sigma8_cb)``."""

    zeros = np.zeros(2)

    tt = np.exp(blocks["tt"])
    ee = np.exp(blocks["ee"])

    Cl = {
        "ell": np.arange(lmax + 1),
        "tt": np.concatenate([zeros, tt]),
        "ee": np.concatenate([zeros, ee]),
        "bb": np.concatenate([zeros, np.exp(blocks["bb"])]),
        "te": np.concatenate([zeros, blocks["te"] * np.sqrt(tt * ee)]),
        "pp": np.concatenate([zeros, np.exp(blocks["pp"])]),
    }

    sigma8, sigma8_cb = np.exp(blocks["scalars"])

    return Cl, float(sigma8), float(sigma8_cb)


def _svd(Z):
    """
    The thin SVD of ``Z`` ``(n, l)``, from the eigenvectors of the
    smaller of its two Gram matrices -- for ``n`` a few hundred points of
    ``l`` ~ 2500 multipoles, much faster than LAPACK's SVD of ``Z``. The
    components it loses to rounding are ~1e-8 of the largest, far below
    what any tolerance here keeps.
    """

    n, l = Z.shape

    if n <= l:
        w, U = np.linalg.eigh(Z @ Z.T)
        order = np.argsort(w)[::-1]
        s = np.sqrt(np.maximum(w[order], 0.0))
        U = U[:, order]
        keep = s > s[0] * 1e-12
        Vt = (U[:, keep].T @ Z) / s[keep, None]
        return U[:, keep], s[keep], Vt

    w, V = np.linalg.eigh(Z.T @ Z)
    order = np.argsort(w)[::-1]
    s = np.sqrt(np.maximum(w[order], 0.0))
    V = V[:, order]
    keep = s > s[0] * 1e-12
    U = (Z @ V[:, keep]) / s[keep]
    return U, s[keep], V[:, keep].T


class PCA:
    """Principal components of a block's training vectors ``(n, l)``."""

    def __init__(self, Y, weights=None, tolerance: float = PCA_TOLERANCE,
                 logl: float = PCA_LOGL, max_components: int = MAX_COMPONENTS):

        Y = np.asarray(Y, dtype=float)

        self.mean = Y.mean(0)
        Z = Y - self.mean

        if Y.shape[1] <= 2:
            # Nothing to compress: the scalars are their own components.
            self.components = np.eye(Y.shape[1])
            return

        U, s, Vt = _svd(Z)

        top = min(max_components, Vt.shape[0])

        # The fewest components that reproduce every training vector; the
        # residual loses one component at a time.
        residual = Z.copy()
        for k in range(1, top + 1):
            residual -= np.outer(U[:, k - 1] * s[k - 1], Vt[k - 1])
            if np.abs(residual).max() >= tolerance:
                continue
            if weights is None or np.sqrt((residual ** 2 @ weights).max()) < logl:
                break

        self.components = Vt[:k]

    @property
    def size(self) -> int:
        return self.components.shape[0]

    def project(self, Y):
        return (np.asarray(Y, dtype=float) - self.mean) @ self.components.T

    def reconstruct(self, coefficients):
        return self.mean + coefficients @ self.components
