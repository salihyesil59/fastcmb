import numpy as np
import pytest

from fastcmb.compress import PCA, from_blocks, to_blocks


def _Cl(lmax=50, a=1.0):
    ell = np.arange(lmax + 1)
    shape = np.where(ell >= 2, a / (ell + 10.0) ** 2, 0.0)
    return {"ell": ell, "tt": 1e3 * shape, "ee": 1e1 * shape, "bb": 1e-3 * shape,
            "te": -0.3 * np.sqrt(1e4) * shape * np.cos(ell / 7.0), "pp": 1e-7 * shape}


def test_blocks_round_trip():

    Cl = _Cl()
    back, s8, s8cb = from_blocks(to_blocks(Cl, 0.81, 0.82), 50)

    for name in Cl:
        assert back[name] == pytest.approx(Cl[name], rel=1e-12, abs=0)
    assert (s8, s8cb) == pytest.approx((0.81, 0.82))


def test_pca_keeps_what_the_tolerance_asks():

    rng = np.random.default_rng(0)
    Y = np.array([to_blocks(_Cl(a=a), 0.8, 0.8)["tt"] + 1e-3 * b * np.linspace(0, 1, 49)
                  for a, b in rng.uniform(0.5, 2, size=(30, 2))])

    pca = PCA(Y)
    assert pca.size <= 3
    assert np.abs(pca.reconstruct(pca.project(Y)) - Y).max() < 1e-3
