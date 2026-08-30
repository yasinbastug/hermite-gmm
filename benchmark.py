"""Tier-1 benchmark: plain GMM vs Hermite-GMM (protocol from the task spec).

Per dataset:
  1. Standardize features.
  2. Plain GMM over G = 1..10, select G by BIC; ARI at BIC-selected G and
     at the true G.
  3. Hermite-GMM over the same G grid, degrees m in {0, 3, 4} (0 = plain
     GMM, confirms the reduction) and a small lambda grid, selected by
     5-fold CV held-out log-likelihood. ARI at the CV-selected settings
     and at the true G.
  4. Headline flag: Hermite at true G beats (on BIC and CV-LL) the plain
     GMM's BIC-selected model when that model used MORE components.
  5. Degree-block energies of the true-G Hermite fit (skew vs kurtosis
     diagnostic).

Results are written to results/<dataset>.json so runs are incremental.

Usage:  python benchmark.py iris wine13 ...     (or: python benchmark.py --all)
"""

import json
import os
import sys
import time
import traceback
import warnings

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from datasets import TIER1
from hermite_gmm import HermiteGMM, multi_indices

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

G_GRID = list(range(1, 11))
DEGREES = [0, 3, 4]
LAMBDAS = [0.1, 1.0, 10.0]
N_FOLDS = 5
MAX_BASIS = 10_000        # skip (k, m) combos with a bigger basis than this
SEED = 0


def _fit_hermite(X, G, m, lam, n_init=2):
    return HermiteGMM(n_components=G, degree=m, reg_lambda=lam,
                      max_iter=150, tol=1e-5, n_init=n_init,
                      random_state=SEED).fit(X)


def _cv_ll(X, G, m, lam):
    """Mean held-out per-sample log-likelihood over K folds."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in kf.split(X):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _fit_hermite(X[tr], G, m, lam)
            scores.append(model.score(X[te]))
        except Exception:
            return -np.inf
    return float(np.mean(scores))


def run_dataset(name):
    X, y, true_G = TIER1[name]()
    X = StandardScaler().fit_transform(X)
    n, k = X.shape
    t_start = time.time()
    res = {"dataset": name, "n": n, "p": k, "true_G": true_G}

    # ---- plain GMM: BIC over the G grid --------------------------------
    gmm_fits, gmm_bic = {}, {}
    for G in G_GRID:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm = GaussianMixture(n_components=G, covariance_type="full",
                                 n_init=10, random_state=SEED).fit(X)
        gmm_fits[G], gmm_bic[G] = gm, float(gm.bic(X))
    G_bic = min(gmm_bic, key=gmm_bic.get)
    res["gmm"] = {
        "G_bic": G_bic,
        "bic_at_G_bic": gmm_bic[G_bic],
        "bic_at_true_G": gmm_bic[true_G],
        "ari_at_G_bic": None if y is None else
            float(adjusted_rand_score(y, gmm_fits[G_bic].predict(X))),
        "ari_at_true_G": None if y is None else
            float(adjusted_rand_score(y, gmm_fits[true_G].predict(X))),
        "cv_ll_at_G_bic": _cv_ll(X, G_bic, 0, 0.0),
        "cv_ll_at_true_G": _cv_ll(X, true_G, 0, 0.0),
    }
    print(f"[{name}] GMM: BIC selects G={G_bic} "
          f"(ARI {res['gmm']['ari_at_G_bic']}), "
          f"ARI@trueG {res['gmm']['ari_at_true_G']}", flush=True)

    # ---- Hermite-GMM: CV over (G, m, lambda) ---------------------------
    degrees = [m for m in DEGREES
               if m == 0 or len(multi_indices(k, m)) <= MAX_BASIS]
    res["degrees_searched"] = degrees
    if degrees != DEGREES:
        print(f"[{name}] skipping degrees "
              f"{sorted(set(DEGREES) - set(degrees))} (basis > {MAX_BASIS})",
              flush=True)

    grid = {}  # G -> (cv_ll, m, lam)
    for G in G_GRID:
        best = (-np.inf, 0, 0.0)
        for m in degrees:
            for lam in (LAMBDAS if m > 0 else [0.0]):
                s = _cv_ll(X, G, m, lam)
                if s > best[0]:
                    best = (s, m, lam)
        grid[G] = best
        print(f"[{name}]   G={G}: best cv_ll={best[0]:.4f} "
              f"(m={best[1]}, lam={best[2]})", flush=True)
    G_cv = max(grid, key=lambda G: grid[G][0])

    def _refit_entry(G):
        cv_ll, m, lam = grid[G]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _fit_hermite(X, G, m, lam, n_init=5)
        return model, {
            "G": G, "m": m, "lambda": lam, "cv_ll": cv_ll,
            "bic": float(model.bic(X)),
            "train_ll": model.score(X),
            "n_iter": model.n_iter_,
            "monotone": bool(np.all(np.diff(model.objective_history_) >= -1e-8)),
            "ari": None if y is None else
                float(adjusted_rand_score(y, model.predict(X))),
            "degree_energies": model.degree_energies().tolist(),
        }

    model_cv, entry_cv = _refit_entry(G_cv)
    model_tg, entry_tg = _refit_entry(true_G)
    res["hermite"] = {"cv_selected": entry_cv, "at_true_G": entry_tg}

    # ---- headline flag --------------------------------------------------
    res["headline"] = bool(
        G_bic > true_G
        and entry_tg["bic"] < gmm_bic[G_bic]
        and entry_tg["cv_ll"] > res["gmm"]["cv_ll_at_G_bic"]
    )
    res["runtime_s"] = round(time.time() - t_start, 1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] Hermite: CV selects (G={G_cv}, m={entry_cv['m']}, "
          f"lam={entry_cv['lambda']}), ARI {entry_cv['ari']}; "
          f"ARI@trueG {entry_tg['ari']}; headline={res['headline']} "
          f"({res['runtime_s']}s)", flush=True)
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    names = list(TIER1) if (not args or args == ["--all"]) else args
    for nm in names:
        try:
            run_dataset(nm)
        except Exception:
            print(f"[{nm}] FAILED:\n{traceback.format_exc()}", flush=True)
