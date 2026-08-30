"""Hermite-corrected Gaussian Mixture Model (Hermite-GMM).

Implements the model from ``hermite_gmm_explained.md`` (Section 24):

    p(x) = sum_c pi_c N(x; mu_c, Sigma_c) * rho_c(u_c(x))

    rho_c(u) = (g_c(u)^2 + eps) / (||b_c||^2 + eps),
    g_c(u)   = sum_{alpha in A_m} b_{c,alpha} phi_alpha(u),
    u_c(x)   = L_c^{-1} (x - mu_c),   Sigma_c = L_c L_c^T (Cholesky)

where phi_alpha(u) = prod_j psi_{alpha_j}(u_j) is the product basis of
normalized probabilists' Hermite polynomials psi_n = He_n / sqrt(n!),
orthonormal under the standard Gaussian measure. Because
E_gamma[g^2] = ||b||^2 (Parseval), rho_c integrates to exactly 1 against
the component Gaussian, so p is a proper density for any b_c.

Key design choices, following the design doc:

* Whitening uses the inverse Cholesky factor (Section 14) -- deterministic
  across iterations, unlike eigendecomposition.
* Degrees 1 and 2 are excluded from the index set by default (Section 26):
  they duplicate what mu_c and Sigma_c already express and create a flat
  likelihood ridge. Corrections start at degree 3.
* Gauge fixing (Section 33): rho_c is invariant under b_c -> s * b_c, so the
  default optimizer constrains ||b_c|| = 1 and does projected gradient
  ascent on the unit sphere (with Armijo backtracking). The naive
  unconstrained parameterization is available via ``gauge_fix=False`` for
  the ablation.
* Regularization (Section 35): -lambda * sum_alpha |alpha|^p b_{c,alpha}^2
  with p = 2 by default.
* Fitting is block-coordinate generalized EM (Section 34). The weighted
  Gaussian M-step is not exactly optimal here (rho depends on mu, Sigma
  through u), so an optional safeguard rejects a Gaussian block move that
  decreases the observed log-likelihood, which makes the outer iteration
  provably monotone.

Setting b_c = (1, 0, ..., 0) gives rho_c = 1 exactly: the model reduces to
a plain GMM, which is also the initialization (via sklearn's
GaussianMixture).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

__all__ = ["HermiteGMM", "multi_indices", "hermite_features"]


# ---------------------------------------------------------------------------
# Hermite basis utilities
# ---------------------------------------------------------------------------

def _weak_compositions(total, parts):
    """Yield all tuples of `parts` non-negative ints summing to `total`."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in _weak_compositions(total - first, parts - 1):
            yield (first,) + rest


def multi_indices(k, m, exclude_degrees_12=True):
    """The truncated index set A_m as a list of k-tuples, grouped by degree.

    Always contains the zero index (0,...,0) first. With
    ``exclude_degrees_12`` (the default), degrees 1 and 2 are dropped so
    Hermite corrections start at degree 3 (Section 26).
    """
    degrees = [d for d in range(m + 1)
               if d == 0 or not (exclude_degrees_12 and d in (1, 2))]
    alphas = []
    for d in degrees:
        alphas.extend(_weak_compositions(d, k))
    return alphas


def _psi_table(U, m):
    """Normalized probabilists' Hermite values psi_d(U) for d = 0..m.

    U has shape (n, k); returns shape (m+1, n, k). Uses the normalized
    recurrence psi_{d+1} = (u psi_d - sqrt(d) psi_{d-1}) / sqrt(d+1).
    """
    n, k = U.shape
    psi = np.empty((m + 1, n, k))
    psi[0] = 1.0
    if m >= 1:
        psi[1] = U
    for d in range(1, m):
        psi[d + 1] = (U * psi[d] - math.sqrt(d) * psi[d - 1]) / math.sqrt(d + 1)
    return psi


def hermite_features(U, alphas):
    """Feature matrix Phi with Phi[i, a] = phi_{alphas[a]}(U[i]).

    U has shape (n, k); result has shape (n, len(alphas)).
    """
    n, k = U.shape
    m = max((sum(a) for a in alphas), default=0)
    # per-coordinate degree never exceeds total degree
    psi = _psi_table(U, m)
    A = np.asarray(alphas, dtype=np.intp)          # (n_alphas, k)
    Phi = np.ones((len(alphas), n))
    for j in range(k):
        nz = A[:, j] > 0                           # psi_0 = 1: skip
        if nz.any():
            Phi[nz] *= psi[A[nz, j], :, j]
    return Phi.T


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class _InnerDiag:
    """Per-outer-iteration diagnostics of the b_c inner optimization."""
    iteration: int
    component: int
    q_gain: float
    grad_norms: list = field(default_factory=list)
    n_backtracks: int = 0


class HermiteGMM:
    """Gaussian mixture with squared-Hermite shape corrections.

    Parameters
    ----------
    n_components : int
        Number of mixture components C.
    degree : int
        Maximum total Hermite degree m. ``degree=0`` (or any degree < 3
        with the default exclusion) is exactly a plain GMM.
    reg_lambda : float
        Regularization strength lambda on the Hermite coefficients.
    reg_power : int
        Degree weight exponent p in w_alpha = |alpha|^p (2 or 4).
    exclude_degrees_12 : bool
        Drop degrees 1-2 from the basis (default True; Section 26).
    eps : float
        Positivity floor epsilon in rho = (g^2 + eps)/(||b||^2 + eps).
    cov_ridge : float
        Ridge eta added to covariance diagonals in the M-step.
    max_iter : int
        Maximum outer EM iterations. 0 = keep the GMM initialization
        (used by the reduction sanity check).
    tol : float
        Stop when the mean per-sample log-likelihood improves by less.
    n_inner : int
        Gradient steps on each b_c per outer iteration (warm-started).
    gauge_fix : bool
        True (default): constrain ||b_c|| = 1, projected gradient on the
        sphere. False: naive gradient ascent on raw b_c (for the
        Section 33 ablation).
    safe_gaussian_update : bool
        Reject a mu/Sigma block move that decreases the observed
        log-likelihood (keeps the outer iteration monotone).
    n_init : int
        Number of initializations for the initial sklearn GMM fit.
    random_state : int or None
        Seed for the initial GMM fit.
    """

    def __init__(self, n_components=2, degree=3, reg_lambda=1e-3, reg_power=2,
                 exclude_degrees_12=True, eps=1e-6, cov_ridge=1e-6,
                 max_iter=200, tol=1e-5, n_inner=5, gauge_fix=True,
                 safe_gaussian_update=True, n_init=1, random_state=None):
        self.n_components = n_components
        self.degree = degree
        self.reg_lambda = reg_lambda
        self.reg_power = reg_power
        self.exclude_degrees_12 = exclude_degrees_12
        self.eps = eps
        self.cov_ridge = cov_ridge
        self.max_iter = max_iter
        self.tol = tol
        self.n_inner = n_inner
        self.gauge_fix = gauge_fix
        self.safe_gaussian_update = safe_gaussian_update
        self.n_init = n_init
        self.random_state = random_state

    # -- per-component pieces ----------------------------------------------

    def _chol(self, c):
        return cholesky(self.covariances_[c], lower=True)

    def _whiten(self, X, c, L=None):
        """u = L^{-1}(x - mu): shape (n, k)."""
        if L is None:
            L = self._chol(c)
        return solve_triangular(L, (X - self.means_[c]).T, lower=True).T

    def _log_gauss(self, U, L):
        """log N(x; mu, Sigma) given whitened coords and Cholesky factor."""
        k = U.shape[1]
        log_det = np.sum(np.log(np.diag(L)))
        return -0.5 * (np.sum(U * U, axis=1) + k * math.log(2 * math.pi)) - log_det

    def _log_rho(self, Phi, b):
        g = Phi @ b
        return np.log(g * g + self.eps) - math.log(b @ b + self.eps)

    def _component_log_prob(self, X):
        """Matrix of log pi_c + log q_c(x_i) + log rho_c(u_ic): (n, C)."""
        n = X.shape[0]
        C = self.n_components
        log_prob = np.empty((n, C))
        for c in range(C):
            L = self._chol(c)
            U = self._whiten(X, c, L)
            Phi = hermite_features(U, self.alphas_)
            log_prob[:, c] = (math.log(max(self.weights_[c], 1e-300))
                              + self._log_gauss(U, L)
                              + self._log_rho(Phi, self.b_[c]))
        return log_prob

    # -- public sklearn-like interface -------------------------------------

    def score_samples(self, X):
        """Per-sample log-density log p(x_i)."""
        X = np.asarray(X, dtype=float)
        log_prob = self._component_log_prob(X)
        return logsumexp(log_prob, axis=1)

    def score(self, X):
        """Mean per-sample log-likelihood."""
        return float(np.mean(self.score_samples(X)))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_prob = self._component_log_prob(X)
        return np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        log_prob = self._component_log_prob(X)
        return np.argmax(log_prob, axis=1)

    def n_effective_params(self):
        """Effective parameter count for BIC (Section 36).

        Each unit-norm b_c contributes |A_m| - 1 free parameters (the
        gauge constraint removes the scale direction; the same count is
        used for the naive parameterization since scale is non-identified
        there too). A pure-GMM basis (|A_m| = 1) contributes zero.
        """
        C, k = self.n_components, self.means_.shape[1]
        A = len(self.alphas_)
        return (C - 1) + C * k + C * k * (k + 1) // 2 + C * (A - 1)

    def bic(self, X):
        X = np.asarray(X, dtype=float)
        ll = float(np.sum(self.score_samples(X)))
        return -2.0 * ll + self.n_effective_params() * math.log(X.shape[0])

    def degree_energies(self):
        """Per-degree coefficient energy sum_{|alpha|=d} b_{c,alpha}^2.

        Returns an array of shape (C, degree+1). These are the
        interpretable, whitening-convention-invariant diagnostics
        (Section 14): degree 3 = skew, 4 = kurtosis, etc.
        """
        degs = np.array([sum(a) for a in self.alphas_])
        out = np.zeros((self.n_components, self.degree + 1))
        for c in range(self.n_components):
            for d in range(self.degree + 1):
                out[c, d] = float(np.sum(self.b_[c][degs == d] ** 2))
        return out

    # -- fitting ------------------------------------------------------------

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, k = X.shape

        self.alphas_ = multi_indices(k, self.degree, self.exclude_degrees_12)
        A = len(self.alphas_)
        self._reg_w = np.array([float(sum(a)) ** self.reg_power
                                for a in self.alphas_])

        # 1-2. Plain-GMM initialization; b_c at the GMM baseline.
        gmm = GaussianMixture(n_components=self.n_components,
                              covariance_type="full",
                              reg_covar=self.cov_ridge,
                              n_init=self.n_init,
                              random_state=self.random_state)
        gmm.fit(X)
        self.weights_ = gmm.weights_.copy()
        self.means_ = gmm.means_.copy()
        self.covariances_ = gmm.covariances_.copy()
        self.b_ = np.zeros((self.n_components, A))
        self.b_[:, 0] = 1.0

        def _penalty():
            return self.reg_lambda * float(
                np.sum(self.b_ * (self._reg_w * self.b_))) / n

        # loglik_history_ tracks the raw mean log-likelihood; the fitting
        # objective (and the provably monotone quantity, see Sec. 30) is
        # the PENALIZED version in objective_history_. With lambda > 0 the
        # b-update may trade a little raw likelihood for penalty, so only
        # the penalized objective is guaranteed non-decreasing.
        self.loglik_history_ = [self.score(X)]
        self.objective_history_ = [self.loglik_history_[0] - _penalty()]
        self.inner_diagnostics_ = []
        self.n_iter_ = 0
        self.converged_ = False

        for it in range(self.max_iter):
            # E-step
            log_prob = self._component_log_prob(X)
            log_norm = logsumexp(log_prob, axis=1, keepdims=True)
            gamma = np.exp(log_prob - log_norm)
            Nc = gamma.sum(axis=0)
            Nc = np.maximum(Nc, 1e-12)

            # M-step: mixture weights (always a valid bound-improving move)
            self.weights_ = Nc / n

            # M-step: Gaussian block (approximate -- guard if requested)
            ll_before_gauss = self.score(X) if self.safe_gaussian_update else None
            old_means = self.means_.copy()
            old_covs = self.covariances_.copy()
            for c in range(self.n_components):
                w = gamma[:, c]
                mu = (w @ X) / Nc[c]
                D = X - mu
                cov = (D * w[:, None]).T @ D / Nc[c]
                cov += self.cov_ridge * np.eye(k)
                self.means_[c] = mu
                self.covariances_[c] = cov
            if self.safe_gaussian_update:
                if self.score(X) < ll_before_gauss - 1e-10:
                    self.means_ = old_means
                    self.covariances_ = old_covs

            # M-step: Hermite coefficients (skip if basis is just degree 0).
            # Refresh the responsibilities first: the Gaussian block just
            # moved, and improving Q against stale gamma does not bound the
            # observed likelihood (it can and did produce tiny decreases).
            # With fresh gamma the accepted-ascent property of _update_b
            # makes this a valid generalized-EM block, so the outer
            # log-likelihood is monotone.
            if A > 1 and self.n_inner > 0:
                log_prob = self._component_log_prob(X)
                gamma = np.exp(log_prob - logsumexp(log_prob, axis=1,
                                                    keepdims=True))
                Nc = np.maximum(gamma.sum(axis=0), 1e-12)
                for c in range(self.n_components):
                    Phi = hermite_features(self._whiten(X, c), self.alphas_)
                    diag = self._update_b(c, Phi, gamma[:, c], Nc[c], it)
                    self.inner_diagnostics_.append(diag)

            ll = self.score(X)
            self.loglik_history_.append(ll)
            self.objective_history_.append(ll - _penalty())
            self.n_iter_ = it + 1
            if self.objective_history_[-1] - self.objective_history_[-2] < self.tol:
                self.converged_ = True
                break

        return self

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    # -- b_c inner optimization ---------------------------------------------

    def _Q(self, b, Phi, gamma, Nc):
        """Section 30 objective Q_c(b) (up to b-independent constants)."""
        g = Phi @ b
        return (float(gamma @ np.log(g * g + self.eps))
                - Nc * math.log(b @ b + self.eps)
                - self.reg_lambda * float(b @ (self._reg_w * b)))

    def _grad_Q(self, b, Phi, gamma, Nc):
        """Section 31 gradient."""
        g = Phi @ b
        data = Phi.T @ (gamma * 2.0 * g / (g * g + self.eps))
        norm = 2.0 * Nc * b / (b @ b + self.eps)
        reg = 2.0 * self.reg_lambda * self._reg_w * b
        return data - norm - reg

    def _update_b(self, c, Phi, gamma, Nc, it):
        """A few ascent steps on Q_c, warm-started from the current b_c.

        gauge_fix=True: projected gradient on the unit sphere with Armijo
        backtracking and renormalization (retraction) after each step.
        gauge_fix=False: plain gradient ascent on raw b (Section 33
        ablation). Both only accept Q-improving steps, so the EM bound
        never decreases.
        """
        b = self.b_[c].copy()
        diag = _InnerDiag(iteration=it, component=c, q_gain=0.0)
        q0 = self._Q(b, Phi, gamma, Nc)
        q = q0
        step = 1.0 / max(Nc, 1.0)
        for _ in range(self.n_inner):
            grad = self._grad_Q(b, Phi, gamma, Nc)
            if self.gauge_fix:
                grad = grad - (b @ grad) * b  # tangent projection
            gnorm = float(np.linalg.norm(grad))
            diag.grad_norms.append(gnorm)
            if gnorm < 1e-12:
                break
            # Armijo backtracking line search
            accepted = False
            t = step
            for _bt in range(30):
                b_new = b + t * grad
                if self.gauge_fix:
                    b_new = b_new / np.linalg.norm(b_new)
                q_new = self._Q(b_new, Phi, gamma, Nc)
                if q_new >= q + 1e-4 * t * gnorm ** 2:
                    accepted = True
                    break
                t *= 0.5
                diag.n_backtracks += 1
            if not accepted:
                break
            b, q = b_new, q_new
            step = min(t * 2.0, 1e6)  # mild step-size adaptation
        # canonical sign: make the degree-0 coefficient non-negative
        if b[0] < 0:
            b = -b
        self.b_[c] = b
        diag.q_gain = q - q0
        return diag
