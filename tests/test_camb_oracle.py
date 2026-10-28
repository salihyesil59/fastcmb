import numpy as np
import pytest

pytest.importorskip("camb")

from fastcmb import CAMBOracle  # noqa: E402

PLANCK = {"H0": 67.36, "ombh2": 0.02237, "omch2": 0.12, "ln1e10As": 3.044, "ns": 0.9649, "tau": 0.0544}


def test_spectra_in_the_documented_conventions():

    out = CAMBOracle(lmax=100)(PLANCK)
    Cl = out["Cl"]

    assert set(Cl) == {"ell", "tt", "te", "ee", "bb", "pp"}
    assert Cl["ell"][-1] == 100 and Cl["tt"].shape == (101,)
    assert np.all(Cl["tt"][:2] == 0)

    # muK^2: D_l of TT near the first acoustic peak's rise, ~ 2000-6000.
    ell = Cl["ell"]
    D = ell * (ell + 1) * Cl["tt"] / (2 * np.pi)
    assert 1000 < D[100] < 6000
    # [L(L+1)]^2 C_L / 2 pi ~ 1e-7.
    assert 1e-8 < Cl["pp"][40] < 1e-6
    assert 0.7 < out["sigma8"] < 0.9


def test_theta_and_logA_are_accepted():

    oracle = CAMBOracle(lmax=50)
    a = oracle(PLANCK)
    b = oracle({**{k: v for k, v in PLANCK.items() if k not in ("H0", "ln1e10As")},
                "cosmomc_theta": 1.04092, "logA": 3.044})

    assert b["Cl"]["tt"][2:] == pytest.approx(a["Cl"]["tt"][2:], rel=2e-3)
