"""Ridge Bradley-Terry reward head — the one trainable model in this sandbox.

Linear reward r_theta(x,y) = <theta, phi(x,y)>, fit by L2-regularized
logistic regression on margins m_i = s_i <theta, z_i>. This is the *only*
model retrained per trial; the embedder that produced Z (src/data/embed.py)
is frozen and never touched here.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ..data.types import PreferenceDataset


class BTHead:
    def __init__(self, lam: float = 1e-2):
        self.lam = lam
        self.theta_: np.ndarray | None = None
        self.history_: dict[str, list[float]] | None = None
        self._hinv_cache: np.ndarray | None = None

    @property
    def theta(self) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("BTHead is not fit yet — call .fit(ds) first")
        return self.theta_

    def margins(self, ds: PreferenceDataset) -> np.ndarray:
        return ds.s * (ds.Z @ self.theta)

    def _loss_and_grad(self, theta: np.ndarray, ds: PreferenceDataset) -> tuple[float, np.ndarray]:
        m = ds.s * (ds.Z @ theta)
        loss = float(np.mean(np.logaddexp(0.0, -m)) + 0.5 * self.lam * theta @ theta)
        grad = -(ds.Z * (ds.s * expit(-m))[:, None]).mean(axis=0) + self.lam * theta
        return loss, grad

    def grad(self, ds: PreferenceDataset) -> np.ndarray:
        _, g = self._loss_and_grad(self.theta, ds)
        return g

    def hessian(self, ds: PreferenceDataset) -> np.ndarray:
        m = self.margins(ds)
        w = expit(m) * expit(-m)  # sigma'(m_i); even in m_i => label-independent
        return (ds.Z * w[:, None]).T @ ds.Z / ds.n + self.lam * np.eye(ds.d)

    def hinv(self, ds: PreferenceDataset) -> np.ndarray:
        if self._hinv_cache is None:
            self._hinv_cache = np.linalg.inv(self.hessian(ds))
        return self._hinv_cache

    def loss(self, ds: PreferenceDataset) -> float:
        loss, _ = self._loss_and_grad(self.theta, ds)
        return loss

    def accuracy(self, ds: PreferenceDataset) -> float:
        """Fraction of examples where the fitted reward ranks chosen above
        rejected in the *observed* (possibly poisoned) orientation."""
        return float(np.mean(self.margins(ds) > 0))

    def fit(
        self,
        ds: PreferenceDataset,
        tol: float = 1e-6,
        max_newton_iters: int = 20,
        track_history: bool = False,
    ) -> "BTHead":
        """L-BFGS to get close, then closed-form Newton polishing (cheap: d is
        small) until ||grad|| < tol. Post: ||grad(theta_hat)|| < tol."""
        history = {"loss": [], "grad_norm": []} if track_history else None

        def record(theta: np.ndarray) -> None:
            if history is not None:
                loss, g = self._loss_and_grad(theta, ds)
                history["loss"].append(loss)
                history["grad_norm"].append(float(np.linalg.norm(g)))

        result = minimize(
            self._loss_and_grad,
            x0=np.zeros(ds.d),
            args=(ds,),
            jac=True,
            method="L-BFGS-B",
            callback=record if track_history else None,
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
        )
        self.theta_ = result.x
        self._hinv_cache = None

        for _ in range(max_newton_iters):
            record(self.theta_)
            g = self.grad(ds)
            if np.linalg.norm(g) < tol:
                break
            self.theta_ = self.theta_ - np.linalg.solve(self.hessian(ds), g)
            self._hinv_cache = None

        self.history_ = history
        return self
