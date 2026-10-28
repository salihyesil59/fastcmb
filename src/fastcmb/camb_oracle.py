"""
CAMB as an oracle for :class:`~fastcmb.emulator.CMBEmulator`, with
parameters named as cobaya and most codes name them.

Parameters (missing ones take the default in brackets):

* ``H0`` -- or ``cosmomc_theta`` (``100 theta_MC``, as ``theta_MC_100``);
* ``ombh2``, ``omch2``; ``omk`` [0];
* ``mnu`` [0.06 eV, one massive species], ``nnu`` [3.044];
* ``ln1e10As`` (or ``logA``) or ``As``; ``ns``; ``tau``;
* ``w`` [-1], ``wa`` [0]: a CPL dark energy through CAMB's PPF module.

CAMB is imported on first use.
"""

from __future__ import annotations

import math

import numpy as np


__all__ = ["CAMBOracle"]

_DEFAULTS = {"omk": 0.0, "mnu": 0.06, "nnu": 3.044, "w": -1.0, "wa": 0.0}


class CAMBOracle:
    """
    Parameters
    ----------
    lmax : int
        Highest multipole of the spectra.
    lens_potential_accuracy : int
        CAMB's; 1 is enough for Planck's CMB spectra, 4 for its lensing.
    halofit_version : str
        CAMB's non-linear model for the lensing potential.
    TCMB : float
        CMB temperature [K].
    accuracy : dict, optional
        Passed to ``CAMBparams.set_accuracy`` (``AccuracyBoost``, ...).
    """

    def __init__(self, lmax: int = 2508, lens_potential_accuracy: int = 1,
                 halofit_version: str = "mead2020", TCMB: float = 2.7255,
                 accuracy: dict | None = None):

        self.lmax = int(lmax)
        self.lens_potential_accuracy = int(lens_potential_accuracy)
        self.halofit_version = halofit_version
        self.TCMB = float(TCMB)
        self.accuracy = dict(accuracy or {})
        self._camb = None

    @property
    def tag(self) -> str:
        accuracy = ",".join(f"{k}={v}" for k, v in sorted(self.accuracy.items()))
        return (f"camb|lmax={self.lmax}|lens={self.lens_potential_accuracy}"
                f"|halofit={self.halofit_version}|T={self.TCMB}|{accuracy}")

    def _import(self):
        if self._camb is None:
            import camb
            self._camb = camb
        return self._camb

    def parameters(self, params: dict):
        """The ``CAMBparams`` for ``params``."""

        camb = self._import()
        p = {**_DEFAULTS, **params}

        pars = camb.CAMBparams()

        theta = p.get("cosmomc_theta", p.get("theta_MC_100"))
        cosmology = dict(
            ombh2=p["ombh2"], omch2=p["omch2"], omk=p["omk"], mnu=p["mnu"], nnu=p["nnu"],
            num_massive_neutrinos=1 if p["mnu"] > 0 else 0, TCMB=self.TCMB, tau=p["tau"],
        )

        if theta is not None:
            pars.set_cosmology(cosmomc_theta=float(theta) / 100.0, **cosmology)
        else:
            pars.set_cosmology(H0=p["H0"], **cosmology)

        if "As" in p:
            As = p["As"]
        else:
            As = math.exp(p.get("ln1e10As", p.get("logA"))) * 1.0e-10

        pars.InitPower.set_params(As=As, ns=p["ns"])

        if p["w"] != -1.0 or p["wa"] != 0.0:
            pars.set_dark_energy(w=p["w"], wa=p["wa"], dark_energy_model="ppf")

        pars.set_matter_power(redshifts=[0.0], kmax=2.0, silent=True)
        pars.set_for_lmax(self.lmax + 500, lens_potential_accuracy=self.lens_potential_accuracy)
        pars.NonLinearModel.set_params(halofit_version=self.halofit_version)

        if self.accuracy:
            pars.set_accuracy(**self.accuracy)

        return pars

    def __call__(self, params: dict) -> dict | None:

        camb = self._import()

        try:
            results = camb.get_results(self.parameters(params))

            total = results.get_cmb_power_spectra(
                lmax=self.lmax, spectra=("total",), CMB_unit="muK", raw_cl=True,
            )["total"]
            pp = results.get_lens_potential_cls(lmax=self.lmax)[:, 0]

            sigma8 = float(results.get_sigma8_0())
            sigma8_cb = float(results.get_sigmaR(8.0, z_indices=[-1], var1="delta_nonu",
                                                 var2="delta_nonu")[0])

        except camb.CAMBError:
            return None

        Cl = {
            "ell": np.arange(self.lmax + 1),
            "tt": total[:, 0], "ee": total[:, 1], "bb": total[:, 2], "te": total[:, 3], "pp": pp,
        }

        # CAMB can return NaN without raising; not a point to learn.
        if not all(np.all(np.isfinite(v)) for v in Cl.values()) or not (
            math.isfinite(sigma8) and math.isfinite(sigma8_cb)
        ):
            return None

        return {"Cl": Cl, "sigma8": sigma8, "sigma8_cb": sigma8_cb}
