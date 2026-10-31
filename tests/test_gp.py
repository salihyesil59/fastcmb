import numpy as np
import pytest
from scipy.optimize import check_grad

import fastcmb.gp as gp_module
from fastcmb.gp import GP


def _data(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 3))
    Y = np.c_[np.sin(2 * X[:, 0]) + X[:, 1], np.cos(X[:, 0] + X[:, 2]), np.sin(3 * X[:, 2])]
    return X, Y


def test_gradient_of_the_marginal_likelihood(monkeypatch):

    X, Y = _data(40)

    captured = {}
    original = gp_module.minimize

    def spy(f, x0, **kw):
        captured["f"] = f
        return original(f, x0, **kw)

    monkeypatch.setattr(gp_module, "minimize", spy)
    gp_module._optimize(X, Y, np.array([0.3, 0.3, 0.4]), np.zeros(4))

    f = captured["f"]
    theta = np.array([0.1, -0.4, 0.3, np.log(1e-5)])
    scale = np.linalg.norm(f(theta)[1])

    assert check_grad(lambda t: f(t)[0], lambda t: f(t)[1], theta) < 1e-4 * scale


@pytest.mark.parametrize("separate", [0, 1, 3])
def test_interpolates_and_knows_where_it_does_not(separate):

    X, Y = _data(80)
    gp = GP(separate=separate).fit(X, Y)

    mean, _ = gp.predict(X[3])
    assert mean == pytest.approx(Y[3], abs=1e-3 if separate == 3 else 1e-2)

    Xt, Yt = _data(50, seed=1)
    errors = np.array([gp.predict(x)[0] - y for x, y in zip(Xt, Yt)])
    assert np.sqrt((errors ** 2).mean()) < 0.03

    near = gp.predict(np.zeros(3))[1]
    far = gp.predict(np.full(3, 4.0))[1]
    assert np.all(far > 100 * near)


def test_own_kernels_help_outputs_unlike_the_rest():

    X, Y = _data(80)
    Xt, Yt = _data(50, seed=1)

    def rms(gp):
        return np.sqrt(((np.array([gp.predict(x)[0] for x in Xt]) - Yt) ** 2).mean())

    assert rms(GP(separate=3).fit(X, Y)) <= rms(GP(separate=0).fit(X, Y)) * 1.05
