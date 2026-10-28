"""
fastcmb as a cobaya theory.

In a cobaya input, in place of ``camb``::

    theory:
      fastcmb.cobaya.FastCAMB:
        tolerance: 0.05
        training_file: chains/emu/lcdm

It provides ``Cl`` -- what cobaya's CMB likelihoods ask for -- and the
derived ``sigma8``, from the parameters :mod:`fastcmb.camb_oracle`
names (``H0`` or ``cosmomc_theta``, ``ombh2``, ``omch2``, ``logA`` or
``As``, ``ns``, ``tau``, and optionally ``mnu``, ``nnu``, ``omk``,
``w``, ``wa``). Where the emulator is not sure, CAMB runs.

cobaya is imported here only, so ``fastcmb`` itself does not need it.
"""

from __future__ import annotations

import numpy as np
from cobaya.theory import Theory

from .camb_oracle import CAMBOracle
from .emulator import CMBEmulator


__all__ = ["FastCAMB"]

_INPUTS = ("H0", "cosmomc_theta", "theta_MC_100", "ombh2", "omch2", "omk", "mnu", "nnu",
           "logA", "ln1e10As", "As", "ns", "tau", "w", "wa")


class FastCAMB(Theory):

    tolerance: float = 0.05
    sigma8_tolerance: float = 1e-3
    min_train: int = 30
    retrain_fraction: float = 0.1
    max_train: int = 800
    training_file: str | None = None
    check_every: int = 200
    lens_potential_accuracy: int = 1
    halofit_version: str = "mead2020"
    TCMB: float = 2.7255
    lmax: int = 0

    def initialize(self):
        self._lmax = int(self.lmax)
        self._lens = int(self.lens_potential_accuracy)
        self.emulator = None

    def get_can_support_params(self):
        return list(_INPUTS)

    def get_can_provide(self):
        return ["Cl"]

    def get_can_provide_params(self):
        return ["sigma8"]

    def must_provide(self, **requirements):

        Cl = requirements.get("Cl")
        if Cl:
            self._lmax = max(self._lmax, *(int(v) for v in Cl.values()))

        return {}

    def initialize_with_provider(self, provider):

        super().initialize_with_provider(provider)

        if self._lmax == 0:
            self._lmax = 2508

        oracle = CAMBOracle(lmax=self._lmax, lens_potential_accuracy=self._lens,
                            halofit_version=self.halofit_version, TCMB=self.TCMB)

        self.emulator = CMBEmulator(
            oracle, inputs=sorted(self.input_params), tolerance=self.tolerance,
            sigma8_tolerance=self.sigma8_tolerance, min_train=self.min_train,
            retrain_fraction=self.retrain_fraction, max_train=self.max_train,
            training_file=self.training_file, check_every=self.check_every,
        )

    def calculate(self, state, want_derived=True, **params_values_dict):

        result = self.emulator(params_values_dict)

        if result is None:
            return False

        state["Cl"] = result.Cl
        state["emulated"] = result.emulated

        if want_derived:
            state["derived"] = {"sigma8": result.sigma8}

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        """As cobaya's ``camb`` returns them: ``tt``, ``te``, ``ee``, ``bb``, ``pp``."""

        Cl = self.current_state["Cl"]

        # cobaya's convention: the dimensionless spectra times u^2, with
        # u from its table; the spectra here are in muK^2 at TCMB.
        units_factor = {
            "1": 1.0, "muK2": self.TCMB * 1e6, "K2": self.TCMB,
            "FIRASmuK2": 2.7255e6, "FIRASK2": 2.7255,
        }
        if units not in units_factor:
            raise ValueError(f"Units {units!r} not recognized; use one of {list(units_factor)}.")

        factor = (units_factor[units] / (self.TCMB * 1e6)) ** 2

        ell = Cl["ell"].astype(float)
        out = {"ell": Cl["ell"]}

        scale = np.ones_like(ell)
        if ell_factor:
            scale = ell * (ell + 1.0) / (2.0 * np.pi)

        for name in ("tt", "te", "ee", "bb"):
            out[name] = Cl[name] * factor * scale
        out["et"] = out["te"]

        # pp is [L(L+1)]^2 C_L / 2 pi already; raw C_L without ell_factor.
        pp = Cl["pp"].copy()
        if not ell_factor:
            pp[1:] /= (ell[1:] * (ell[1:] + 1.0)) ** 2 / (2.0 * np.pi)
        out["pp"] = pp

        return out
