"""
A multi-output Gaussian process.

The outputs are the principal components of one spectrum. The leading
ones -- those that carry most of the likelihood -- each get a kernel of
their own; the rest share one. Each kernel is a squared exponential with
a length scale per input, a nugget, and an amplitude per output.

A polynomial mean is fitted first by least squares -- quadratic once
there are enough points for it, linear before -- and the process
models what it leaves. Spectra are smooth in the parameters over a
posterior's width, so the polynomial carries most of the dependence
and the process the rest; away from the training points the prediction
falls back to the polynomial rather than to zero.

Kernel hyperparameters are fitted, by maximum likelihood, on at most
:data:`OPTIMIZE_POINTS` of the points; the process itself on all.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize


__all__ = ["GP"]

#: Bounds on the log length scales (inputs are standardized) and on the
#: log of the nugget, relative to each output's amplitude.
_LOG_ELL = (np.log(0.03), np.log(300.0))
_LOG_NUGGET = (np.log(1e-12), np.log(1e-2))

#: Most points the hyperparameters are fitted on.
OPTIMIZE_POINTS = 200


def _design(X, degree):
    """Columns of a polynomial in the inputs: 1, x_i and, for degree 2, x_i x_j."""

    X = np.atleast_2d(X)
    columns = [np.ones((X.shape[0], 1)), X]

    if degree >= 2:
        i, j = np.triu_indices(X.shape[1])
        columns.append(X[:, i] * X[:, j])

    return np.hstack(columns)


def _sq_dist(A, B, inv_ell):
    a = A * inv_ell
    b = B * inv_ell
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0)


def _optimize(X, R, w, start, fresh: bool = True):
    """
    Maximum-likelihood ``(log_ell, log_nugget)`` for the outputs ``R``
    sharing one kernel shape, weighted by ``w``. Each output's amplitude
    is profiled out. From ``start`` and, if ``fresh``, from a default
    start too, keeping the better.
    """

    n, d = X.shape

    def negloglike(theta):

        log_ell, log_nugget = theta[:d], theta[d]

        K0 = np.exp(-0.5 * _sq_dist(X, X, np.exp(-log_ell)))
        C = K0 + np.exp(log_nugget) * np.eye(n)

        try:
            factor = cho_factor(C, lower=True)
        except np.linalg.LinAlgError:
            return 1e30, np.zeros_like(theta)

        alpha = cho_solve(factor, R)
        q = np.maximum((R * alpha).sum(0), 1e-300)
        logdet = 2.0 * np.log(np.diag(factor[0])).sum()

        value = 0.5 * n * (w * np.log(q / n)).sum() + 0.5 * logdet

        Cinv = cho_solve(factor, np.eye(n))
        W = n * (alpha * (w / q)) @ alpha.T - Cinv

        grad = np.empty_like(theta)
        WK = W * K0
        for i in range(d):
            diff2 = (X[:, i, None] - X[None, :, i]) ** 2
            grad[i] = -0.5 * (WK * diff2).sum() * np.exp(-2.0 * log_ell[i])
        grad[d] = -0.5 * np.exp(log_nugget) * np.trace(W)

        return value, grad

    bounds = [_LOG_ELL] * d + [_LOG_NUGGET]
    lo, hi = np.array(bounds).T

    # From the last fit's hyperparameters and from a fresh start: the
    # last fit's alone can hold on to an optimum of fewer points that
    # is not one any more.
    default = np.r_[np.zeros(d), np.log(1e-8)]
    starts = [start, default] if fresh and not np.array_equal(start, default) else [start]

    best = None
    for s in starts:
        result = minimize(negloglike, np.clip(s, lo, hi), jac=True, method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": 100})
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result

    return np.asarray(start, dtype=float) if best is None else best.x


class _Kernel:
    """One kernel and the outputs it models: its factor and weights."""

    def __init__(self, theta, X, R):

        d = X.shape[1]

        self.theta = np.asarray(theta, dtype=float)
        self.inv_ell = np.exp(-self.theta[:d])
        self.nugget = float(np.exp(self.theta[d]))

        C = np.exp(-0.5 * _sq_dist(X, X, self.inv_ell)) + self.nugget * np.eye(X.shape[0])

        self.L = np.linalg.cholesky(C)
        self.alpha = cho_solve((self.L, True), R)
        # Each output's amplitude, at its maximum-likelihood value.
        self.amp2 = np.maximum((R * self.alpha).sum(0) / X.shape[0], 1e-300)

    def covariance(self, X, x):
        """The kernel between ``x`` and the training points."""
        return np.exp(-0.5 * _sq_dist(x[None, :], X, self.inv_ell)[0])

    def variance(self, k):
        """Each output's predictive variance, given :meth:`covariance`."""
        v = solve_triangular(self.L, k, lower=True, check_finite=False)
        return self.amp2 * max(1.0 + self.nugget - v @ v, 0.0)

    @property
    def prior_variance(self):
        """An upper bound on :meth:`variance`, at no cost."""
        return self.amp2 * (1.0 + self.nugget)


class GP:
    """
    Parameters
    ----------
    separate : int
        How many of the outputs, the most weighted, get a kernel of
        their own.
    """

    def __init__(self, separate: int = 4):

        self.separate = int(separate)
        #: Hyperparameters from the last fit, reused as a starting point:
        #: ``"shared"`` and one per output with a kernel of its own.
        self.theta: dict = {}

    def fit(self, X, Y, optimize: bool = True, weights=None) -> "GP":
        """
        ``X`` is ``(n, d)``, standardized; ``Y`` is ``(n, m)``.

        ``weights`` (``m``) say how much each output counts: which get
        a kernel of their own, and how much each weighs in fitting the
        shared one. By default its variance.
        """

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if Y.ndim == 1:
            Y = Y[:, None]

        n, d = X.shape
        m = Y.shape[1]

        # A polynomial is fitted only once there are enough points that
        # it does not interpolate them on its own.
        quadratic = (d + 1) * (d + 2) // 2
        self.degree = 2 if n > 2 * quadratic else (1 if n > 2 * (d + 1) else 0)

        A = _design(X, self.degree)

        if self.degree > 0:
            self.beta = np.linalg.lstsq(A, Y, rcond=None)[0]
        else:
            self.beta = Y.mean(0)[None, :]

        R = Y - A @ self.beta

        if weights is None:
            weights = R.var(0)

        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(m, 1.0 / m)

        own = [int(j) for j in np.argsort(weights)[::-1][: min(self.separate, m)]]
        shared = [j for j in range(m) if j not in own]
        self.shared_columns = shared

        self.X = X

        # A fixed subset for the hyperparameters: the same points every
        # time, so a refit with a few more points changes little.
        if n > OPTIMIZE_POINTS:
            subset = np.random.default_rng(0).choice(n, OPTIMIZE_POINTS, replace=False)
        else:
            subset = np.arange(n)

        optimize = optimize and n > d + 2
        default = np.r_[np.zeros(d), np.log(1e-8)]

        def theta(key, columns, w):
            start = np.asarray(self.theta.get(key, default), dtype=float)
            known = key in self.theta and start.size == d + 1
            if not known:
                start = default
            if optimize:
                # Every kernel also tries a fresh start: from the last
                # fit's alone, a kernel holds on to an optimum of other
                # points (and its component may not be the same one),
                # with length scales that leave the emulator trusting
                # few predictions.
                start = _optimize(X[subset], R[subset][:, columns], w / w.sum(), start)
            self.theta[key] = start
            return start

        # The shared kernel models every output; its predictions of the
        # outputs with a kernel of their own are replaced by those.
        columns = shared or own
        self.shared = _Kernel(theta("shared", columns, weights[columns] + 1e-300), X, R)

        self.kernels = [(j, _Kernel(theta(j, [j], np.ones(1)), X, R[:, [j]])) for j in own]

        return self

    # ---------------------------------------------------------

    @property
    def log_ell(self):
        """The shared kernel's log length scales."""
        return -np.log(self.shared.inv_ell)

    @property
    def log_nugget(self):
        return float(np.log(self.shared.nugget))

    @property
    def parts(self) -> list:
        """``(columns, kernel)`` for each kernel, with the outputs it predicts."""

        parts = [(np.array([j]), kernel) for j, kernel in self.kernels]

        if self.shared_columns:
            parts.insert(0, (np.array(self.shared_columns), self.shared))

        return parts

    def mean(self, x):
        """
        Mean ``(m,)`` of the outputs at one standardized input ``x``,
        and each part's kernel vector, for :meth:`_Kernel.variance`.
        """

        x = np.asarray(x, dtype=float)

        k_shared = self.shared.covariance(self.X, x)
        mean = k_shared @ self.shared.alpha

        ks = []
        if self.shared_columns:
            ks.append(k_shared)

        for j, kernel in self.kernels:
            k = kernel.covariance(self.X, x)
            mean[j] = k @ kernel.alpha[:, 0]
            ks.append(k)

        return _design(x, self.degree)[0] @ self.beta + mean, ks

    def predict(self, x):
        """
        Mean ``(m,)`` and variance ``(m,)`` of the outputs at one
        standardized input ``x`` ``(d,)``.
        """

        mean, ks = self.mean(x)
        var = np.empty_like(mean)

        for (columns, kernel), k in zip(self.parts, ks):
            v = kernel.variance(k)
            var[columns] = v if kernel is not self.shared else v[columns]

        return mean, var
