"""Sanity tests for HermiteGMM (the three checks from the task spec).

Run:  python test_hermite_gmm.py     (or pytest test_hermite_gmm.py)

1. GMM reduction: at the baseline initialization with 0 EM iterations,
   score_samples reproduces sklearn's GaussianMixture exactly.
2. Monotone log-likelihood across EM iterations on a real dataset (iris).
3. Synthetic 1-D skew: the fitted correction recovers positive skew
   (degree-3 energy dominant, positive psi_3 coefficient) and beats a
   plain 1-component GMM; plain GMM needs more components at similar BIC.
"""

import math

import numpy as np
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

from hermite_gmm import HermiteGMM, hermite_features, multi_indices


def test_hermite_basis_orthonormal():
    """The product basis is orthonormal under the standard Gaussian.

    Uses tensor-product Gauss-Hermite(e) quadrature, exact for
    polynomials of this degree.
    """
    nodes, w = np.polynomial.hermite_e.hermegauss(12)
    w = w / math.sqrt(2 * math.pi)          # normalize to the Gaussian measure
    U = np.array([(a, b) for a in nodes for b in nodes])
    W = np.array([wa * wb for wa in w for wb in w])
    alphas = multi_indices(2, 4, exclude_degrees_12=False)
    Phi = hermite_features(U, alphas)
    G = (Phi * W[:, None]).T @ Phi
    err = np.max(np.abs(G - np.eye(len(alphas))))
    assert err < 1e-10, f"Gram matrix deviates from identity by {err:.2e}"
    print(f"PASS  basis orthonormality (quadrature, k=2, m=4, err={err:.1e})")


def test_gmm_reduction():
    """0-iteration HermiteGMM == sklearn GaussianMixture, exactly."""
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal([0, 0], 1.0, (150, 2)),
                   rng.normal([4, 3], 0.7, (100, 2))])
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          reg_covar=1e-6, random_state=0).fit(X)
    hg = HermiteGMM(n_components=2, degree=4, max_iter=0,
                    cov_ridge=1e-6, random_state=0).fit(X)
    diff = np.max(np.abs(hg.score_samples(X) - gmm.score_samples(X)))
    assert diff < 1e-9, f"reduction not exact: max |Δ log p| = {diff:.2e}"
    print(f"PASS  GMM reduction (max |Δ log p| = {diff:.2e})")


def test_monotone_loglik():
    """The EM objective is non-decreasing across iterations on real data.

    The provably monotone quantity is the penalized objective
    LL - lambda * sum b'Wb (the b-update maximizes it); with lambda > 0
    raw LL may trade tiny amounts against the penalty. Checked at two
    lambda values on iris; raw LL must also strictly improve overall.
    """
    X = load_iris().data
    for lam in (1e-3, 10.0):
        hg = HermiteGMM(n_components=3, degree=4, reg_lambda=lam,
                        max_iter=100, random_state=0).fit(X)
        obj = np.array(hg.objective_history_)
        drops = np.diff(obj) < -1e-8
        assert not drops.any(), \
            f"objective decreased (lam={lam}) at iters {np.where(drops)[0]}: {obj}"
        ll = np.array(hg.loglik_history_)
        assert ll[-1] > ll[0], "no improvement over the plain-GMM init"
        print(f"PASS  monotone EM objective on iris (lam={lam}, "
              f"{hg.n_iter_} iters, LL {ll[0]:.4f} -> {ll[-1]:.4f}, "
              f"converged={hg.converged_})")


def test_synthetic_skew_recovery():
    """1-D skew-normal data: correction finds positive skew; plain GMM loses."""
    rng = np.random.default_rng(7)
    X = stats.skewnorm.rvs(a=6, size=3000, random_state=rng).reshape(-1, 1)

    hg = HermiteGMM(n_components=1, degree=4, reg_lambda=1e-3,
                    max_iter=200, random_state=0).fit(X)
    g1 = GaussianMixture(n_components=1, random_state=0).fit(X)

    # (a) better fit than a single Gaussian
    assert hg.score(X) > g1.score(X) + 1e-3, "Hermite-GMM did not beat 1-comp GMM"

    # (b) the correction is concentrated at degree 3 (skew) and positive
    energies = hg.degree_energies()[0]          # degrees 0..4
    assert energies[3] > energies[4], \
        f"degree-3 energy should dominate degree-4: {energies}"
    idx3 = hg.alphas_.index((3,))
    assert hg.b_[0][idx3] > 0, "psi_3 coefficient should be positive for right skew"

    # (c) plain GMM needs more components to reach comparable BIC
    gmm_bics = {C: GaussianMixture(n_components=C, random_state=0).fit(X).bic(X)
                for C in range(1, 6)}
    best_C = min(gmm_bics, key=gmm_bics.get)
    assert best_C > 1, "expected BIC to select >1 Gaussian for skewed data"
    assert hg.bic(X) < gmm_bics[1], "Hermite BIC should beat 1-comp GMM BIC"

    print(f"PASS  synthetic skew: LL {g1.score(X):.4f} (GMM-1) -> "
          f"{hg.score(X):.4f} (Hermite-1); degree energies "
          f"{np.round(energies, 4)}; GMM BIC selects C={best_C}, "
          f"Hermite-1 BIC {hg.bic(X):.1f} vs GMM-1 BIC {gmm_bics[1]:.1f} "
          f"vs GMM-{best_C} BIC {gmm_bics[best_C]:.1f}")


def test_bic_effective_df_shrinkage():
    """BIC's Hermite-block d.o.f. must actually respond to regularization.

    Charging the raw |A_m|-1 coefficient count regardless of lambda made
    BIC reject every Hermite fit no matter how strongly regularized (a
    real bug found via the benchmark: raw BIC was 5-10x worse than plain
    GMM's even at heavy shrinkage). The fixed formula must:
      (a) give exactly 0 effective Hermite d.o.f. at degree=0 (pure GMM),
      (b) decrease monotonically as lambda increases, and
      (c) converge toward plain GMM's BIC (from above) as lambda -> inf,
          since the model degenerates to a plain GMM in that limit.
    """
    X = load_iris().data

    hg0 = HermiteGMM(n_components=3, degree=0, max_iter=50,
                     random_state=0).fit(X)
    df0 = hg0._hermite_effective_df(X)
    assert abs(df0) < 1e-6, f"m=0 should give exactly 0 effective df, got {df0}"

    lams = [1e-3, 1.0, 100.0, 1e6]
    dfs, bics = [], []
    for lam in lams:
        hg = HermiteGMM(n_components=3, degree=4, reg_lambda=lam,
                        max_iter=100, random_state=0).fit(X)
        dfs.append(hg._hermite_effective_df(X))
        bics.append(hg.bic(X))
    assert all(a > b for a, b in zip(dfs, dfs[1:])), \
        f"effective df should decrease monotonically in lambda: {dfs}"

    gmm_bic = GaussianMixture(n_components=3, random_state=0).fit(X).bic(X)
    assert dfs[-1] < 0.5, f"df should be ~0 as lambda->inf, got {dfs[-1]:.3f}"
    assert bics[-1] < bics[0], \
        "heavily-regularized BIC should be far below the raw-count BIC"
    assert abs(bics[-1] - gmm_bic) < 0.05 * gmm_bic, \
        (f"as lambda->inf, Hermite BIC ({bics[-1]:.1f}) should converge "
         f"toward plain GMM's BIC ({gmm_bic:.1f})")

    print(f"PASS  BIC effective df: m=0 -> {df0:.4f}; "
          f"df(lambda) {np.round(dfs, 2)} monotone decreasing; "
          f"BIC(lambda->inf)={bics[-1]:.1f} vs GMM BIC={gmm_bic:.1f}")


if __name__ == "__main__":
    test_hermite_basis_orthonormal()
    test_gmm_reduction()
    test_monotone_loglik()
    test_bic_effective_df_shrinkage()
    test_synthetic_skew_recovery()
    print("\nAll sanity tests passed.")
