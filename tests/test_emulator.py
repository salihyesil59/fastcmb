import numpy as np
import pytest

from fastcmb import CMBEmulator

LMAX = 200


def toy_oracle(params):
    """Smooth, CMB-like spectra in two parameters; fails for a < 0."""

    a, b = params["a"], params["b"]
    if a < 0:
        return None

    ell = np.arange(LMAX + 1).astype(float)
    shape = np.where(ell >= 2, np.exp(-ell / (150.0 * (1 + 0.3 * b))) / (ell + 5.0) ** 2, 0.0)
    tt = 1e4 * a * shape
    ee = 1e2 * a * shape * (1 + 0.1 * b)
    te = 0.5 * np.sqrt(tt * ee) * np.cos(ell / (40.0 + 5 * b))
    return {
        "Cl": {"ell": np.arange(LMAX + 1), "tt": tt, "te": te, "ee": ee,
               "bb": 1e-3 * a * shape, "pp": 1e-7 * a * shape * (ell + 1) ** 2},
        "sigma8": 0.8 * np.sqrt(a), "sigma8_cb": 0.81 * np.sqrt(a),
    }


def _points(n, seed):
    rng = np.random.default_rng(seed)
    return [{"a": rng.uniform(0.95, 1.05), "b": rng.uniform(-0.1, 0.1)} for _ in range(n)]


def test_learns_then_answers_from_the_emulator():

    calls = []

    def oracle(p):
        calls.append(p)
        return toy_oracle(p)

    emulator = CMBEmulator(oracle, lmax=LMAX, min_train=15, check_every=0)

    for p in _points(80, seed=0):
        emulator(p)

    stats = emulator.stats()
    assert stats["emulated"] > 30
    assert stats["oracle"] == len(calls) < 50

    for p in _points(10, seed=1):
        result = emulator(p)
        truth = toy_oracle(p)
        assert result.Cl["tt"][2:] == pytest.approx(truth["Cl"]["tt"][2:], rel=1e-3)
        assert result.sigma8 == pytest.approx(truth["sigma8"], rel=1e-3)
        assert set(result.Cl) == set(truth["Cl"])


def test_checks_measure_the_error():

    emulator = CMBEmulator(toy_oracle, lmax=LMAX, min_train=15, check_every=5)

    for p in _points(80, seed=2):
        emulator(p)

    stats = emulator.stats()
    assert stats["checks"] > 0
    assert stats["check_error_max"] < 3 * emulator.tolerance


def test_checks_grow_rarer_while_they_pass():

    emulator = CMBEmulator(toy_oracle, lmax=LMAX, min_train=15, check_every=2)

    for p in _points(200, seed=3):
        emulator(p)

    stats = emulator.stats()
    # Every 2 predictions would be about 80 checks; doubling up to 20, far fewer.
    assert stats["emulated"] > 150
    assert 3 <= stats["checks"] < 20
    assert emulator._check_interval == 20


def test_loose_tolerance_gives_an_approximate_answer_without_the_oracle():

    # The toy's errors are tiny; a tolerance below the far point's.
    emulator = CMBEmulator(toy_oracle, lmax=LMAX, min_train=15, check_every=0, tolerance=1e-5)

    for p in _points(30, seed=4):
        emulator(p)

    far = {"a": 3.0, "b": 3.0}
    oracle_calls, size = emulator.n_oracle, emulator.online.size

    approximate = emulator(far, loose_tolerance=1e9)
    assert approximate.approximate and approximate.emulated
    assert approximate.error > emulator.tolerance
    assert (emulator.n_oracle, emulator.online.size) == (oracle_calls, size)
    assert emulator.stats()["approximate"] == 1

    # Not allowed one, the oracle runs and the point is learned.
    exact = emulator(far)
    assert not exact.approximate and not exact.emulated
    assert emulator.online.size == size + 1

    # Within the tolerance, the answer is not approximate.
    near = emulator(_points(1, seed=4)[0], loose_tolerance=1e9)
    assert near.emulated and not near.approximate


def test_a_point_the_oracle_cannot_compute_is_none_and_not_learned():

    emulator = CMBEmulator(toy_oracle, lmax=LMAX)

    assert emulator({"a": -1.0, "b": 0.0}) is None
    assert emulator.online.size == 0


def test_inputs_are_fixed_and_required():

    emulator = CMBEmulator(toy_oracle, lmax=LMAX, inputs=["a", "b"])

    # Extra parameters are ignored; a missing input is an error.
    emulator({"a": 1.0, "b": 0.0, "nuisance": 3.0})
    with pytest.raises(KeyError, match="'b'"):
        emulator({"a": 1.0})


def test_training_file_is_shared(tmp_path):

    path = str(tmp_path / "emu" / "toy")

    first = CMBEmulator(toy_oracle, lmax=LMAX, min_train=15, training_file=path)
    for p in _points(20, seed=3):
        first(p)

    # As another process -- a parallel chain, or a later run -- sees it.
    import glob
    import os

    (own,) = glob.glob(path + ".*.npz")
    os.replace(own, path + ".999999.npz")

    second = CMBEmulator(toy_oracle, lmax=LMAX, min_train=15, training_file=path)
    second(_points(1, seed=4)[0])

    assert second.online.size >= 15
    assert second.online.ready
