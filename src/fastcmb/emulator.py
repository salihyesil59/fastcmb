"""
The emulator as a function: parameters in, spectra out.

:class:`CMBEmulator` wraps any Boltzmann code -- an *oracle*, a function
from a parameter dict to spectra. At each call it predicts, and answers
with the prediction when its error, in units of ``ln L``, is below the
tolerance; otherwise it runs the oracle and learns the point.

This is all an analysis code needs: CosmoFit's ``fastcmb`` theory, the
cobaya theory in :mod:`fastcmb.cobaya`, or a script of one's own::

    from fastcmb import CAMBOracle, CMBEmulator

    emulator = CMBEmulator(CAMBOracle(lmax=2508))
    result = emulator({"H0": 67.4, "ombh2": 0.0224, "omch2": 0.12,
                       "ln1e10As": 3.044, "ns": 0.965, "tau": 0.054})
    result.Cl["tt"], result.sigma8, result.emulated

An oracle returns a dict with ``Cl`` -- ``ell``, ``tt``, ``te``, ``ee``,
``bb`` (``C_l`` in muK^2) and ``pp`` (``[L(L+1)]^2 C_L / 2 pi``), from
``l = 0`` to ``lmax`` -- and ``sigma8`` and ``sigma8_cb``; or ``None``
when it cannot compute the point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .compress import from_blocks, to_blocks
from .online import OnlineEmulator


__all__ = ["CMBEmulator", "Result"]

log = logging.getLogger("fastcmb")


@dataclass
class Result:
    Cl: dict
    sigma8: float
    sigma8_cb: float
    #: Whether the spectra are the emulator's (else the oracle's).
    emulated: bool
    #: The predicted ``ln L`` error, when there was a prediction.
    error: float | None = None


class CMBEmulator:
    """
    Parameters
    ----------
    oracle : callable
        ``oracle(params) -> dict | None``, as described in the module.
        It may also be given per call, to :meth:`__call__`.
    lmax : int, optional
        Highest multipole; by default the oracle's ``lmax``.
    inputs : list of str, optional
        The parameters the spectra depend on, in a fixed order. By
        default the keys of the first call's parameters, sorted. A
        parameter not listed is ignored -- it must not change the
        spectra.
    tag : str, optional
        What the spectra are computed for (model, accuracy); points
        stored under another tag are not read. By default the oracle's
        ``tag``.
    tolerance : float
        Largest predicted ``ln L`` error the emulator answers with.
        The measure is that of an idealized Planck (see
        :mod:`fastcmb.metric`).
    sigma8_tolerance : float
        Largest predicted relative error of ``sigma8``.
    min_train, retrain_fraction, max_train, training_file
        As :class:`~fastcmb.online.OnlineEmulator`'s ``min_train``,
        ``retrain_fraction``, ``max_train`` and ``path``.
    check_every : int
        After this many predictions, the oracle runs as well and the
        error is measured; a prediction off by more than three times
        the tolerance is learned. 0 never. While the checks find the
        error within the tolerance, the interval between them doubles,
        up to ten times ``check_every``; a check that does not brings
        it back.
    """

    def __init__(self, oracle: Callable | None = None, lmax: int | None = None,
                 inputs: list[str] | None = None, tag: str | None = None,
                 tolerance: float = 0.05, sigma8_tolerance: float = 1e-3,
                 min_train: int = 30, retrain_fraction: float = 0.1, max_train: int = 800,
                 training_file: str | None = None, check_every: int = 200):

        self.oracle = oracle
        self.lmax = lmax if lmax is not None else getattr(oracle, "lmax", None)
        self.inputs = None if inputs is None else list(inputs)
        self.tag = tag if tag is not None else str(getattr(oracle, "tag", ""))
        self.tolerance = float(tolerance)
        self.sigma8_tolerance = float(sigma8_tolerance)
        self.check_every = int(check_every)
        self._options = dict(min_train=min_train, retrain_fraction=retrain_fraction,
                             max_train=max_train, path=training_file)

        self.online: OnlineEmulator | None = None

        self.n_oracle = 0
        self.n_emulated = 0
        self.check_errors: list[float] = []
        self._check_interval = self.check_every
        self._next_check = self.check_every

    # ---------------------------------------------------------

    def _vector(self, params) -> np.ndarray:

        if self.inputs is None:
            self.inputs = sorted(params)

        try:
            return np.array([float(params[name]) for name in self.inputs])
        except KeyError as error:
            raise KeyError(f"fastcmb: the emulator's input {error} is missing from the parameters.") from None

    def _online(self) -> OnlineEmulator:

        if self.online is None:
            if self.lmax is None:
                raise ValueError("fastcmb: give lmax, or an oracle that has one.")
            self.online = OnlineEmulator(self.inputs, self.lmax, tag=self.tag, **self._options)

        return self.online

    def __call__(self, params: dict, oracle: Callable | None = None) -> Result | None:
        """The spectra at ``params``; ``None`` if the oracle cannot compute them."""

        oracle = oracle or self.oracle
        if oracle is None:
            raise ValueError("fastcmb: no oracle to compute the spectra with.")

        x = self._vector(params)
        online = self._online()

        prediction = online.predict(x, tolerance=self.tolerance, scalar_tolerance=self.sigma8_tolerance)

        trusted = (
            prediction is not None
            and prediction.error <= self.tolerance
            and prediction.scalar_error <= self.sigma8_tolerance
        )

        if not trusted:
            return self._run(oracle, params, x, learn=True)

        self.n_emulated += 1

        if not (self.check_every and self.n_emulated >= self._next_check):
            Cl, sigma8, sigma8_cb = from_blocks(prediction.blocks, online.lmax)
            return Result(Cl, sigma8, sigma8_cb, emulated=True, error=prediction.error)

        # A check: the oracle too, and the error measured.
        result = self._run(oracle, params, x, learn=False)
        self._next_check = self.n_emulated + self._check_interval

        if result is not None:

            truth = to_blocks(result.Cl, result.sigma8, result.sigma8_cb)
            error = online.error_between(prediction.blocks, truth)
            self.check_errors.append(error)

            # Checks grow rarer while they pass.
            if error <= self.tolerance:
                self._check_interval = min(2 * self._check_interval, 10 * self.check_every)
            else:
                self._check_interval = self.check_every
                self._next_check = self.n_emulated + self._check_interval

            if error > 3.0 * self.tolerance:
                log.warning(
                    "fastcmb: a prediction was off by %.3g in ln L (predicted %.3g); "
                    "the point is learned.", error, prediction.error,
                )
                online.add(x, truth)

        return result

    def _run(self, oracle, params, x, learn: bool) -> Result | None:

        self.n_oracle += 1

        out = oracle(params)

        if out is None:
            return None

        result = Result(out["Cl"], float(out["sigma8"]), float(out["sigma8_cb"]), emulated=False)

        if learn:
            self.online.add(x, to_blocks(result.Cl, result.sigma8, result.sigma8_cb))

        return result

    # ---------------------------------------------------------

    def stats(self) -> dict:
        """Calls to the oracle and the emulator, and what the checks found."""

        errors = np.array(self.check_errors)

        return {
            "oracle": self.n_oracle,
            "emulated": self.n_emulated,
            "training_points": 0 if self.online is None else self.online.size,
            "checks": errors.size,
            "check_error_median": float(np.median(errors)) if errors.size else None,
            "check_error_max": float(errors.max()) if errors.size else None,
        }
