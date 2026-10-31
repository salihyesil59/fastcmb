"""
How large an emulator error is, in units of the log-likelihood.

An error ``delta`` in the spectra moves a Gaussian likelihood's
``chi^2`` by ``2 delta^T F r + delta^T F delta``, with ``r`` the
residual and ``F`` the inverse covariance of the data. For a point a
chain visits, the whitened residual is of order one in each mode, so
``|Delta ln L| ~ sqrt(delta^T F delta)``.

``F`` here is that of an idealized Planck: cosmic variance on
``f_sky = 0.6`` of the sky plus white noise with Planck 143 GHz's
level and beam, and the lensing potential for ``8 <= L <= 1000``
(Planck's and ACT DR6's ranges). Planck's real errors are larger (foregrounds, calibration), so the
error this gives is an upper bound on the one the likelihoods see.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PLANCK", "fisher_weights"]

_ARCMIN = np.pi / 180.0 / 60.0

PLANCK = {
    "f_sky": 0.6,
    "noise_T": 33.0,          # muK arcmin
    "noise_P": 70.0,          # muK arcmin
    "fwhm": 7.3,              # arcmin
    "noise_pp": 1.5e-7,       # [L(L+1)]^2 N_L / 2 pi
    "L_pp": (8, 1000),       # Planck 8-400, ACT DR6 40-763
}


def fisher_weights(Cl: dict, experiment: dict = PLANCK) -> dict:
    """
    Per multipole from ``l = 2``, the weight ``F_l`` of an error in
    each block's variable: ``ln C_l`` for ``tt``, ``ee``, ``pp``; the
    TE correlation for ``te``. ``bb`` gets none: no likelihood here
    reads it.
    """

    ell = Cl["ell"][2:].astype(float)
    nu = experiment["f_sky"] * (2.0 * ell + 1.0)

    theta = experiment["fwhm"] * _ARCMIN
    beam = np.exp(ell * (ell + 1.0) * theta ** 2 / (8.0 * np.log(2.0)))

    N_T = (experiment["noise_T"] * _ARCMIN) ** 2 * beam
    N_P = (experiment["noise_P"] * _ARCMIN) ** 2 * beam

    tt, ee, te, pp = (Cl[name][2:] for name in ("tt", "ee", "te", "pp"))

    var_te = ((tt + N_T) * (ee + N_P) + te ** 2) / nu

    lo, hi = experiment["L_pp"]
    in_range = (ell >= lo) & (ell <= hi)

    return {
        "tt": 0.5 * nu * (tt / (tt + N_T)) ** 2,
        "ee": 0.5 * nu * (ee / (ee + N_P)) ** 2,
        # An error in the correlation r moves C_TE by delta r sqrt(TT EE).
        "te": tt * ee / var_te,
        "pp": np.where(in_range, 0.5 * nu * (pp / (pp + experiment["noise_pp"])) ** 2, 0.0),
        "bb": np.zeros_like(ell),
    }
