"""Gauge-fixing ablation (Section 33 of the design doc).

The correction rho_c is invariant under b_c -> s * b_c, so the naive
unconstrained parameterization has a near-flat direction along b (only
weakly broken by eps and the ridge). The claimed fix is to constrain
||b_c|| = 1 and optimize on the unit sphere. This script checks the claim
empirically by fitting the same model both ways and reporting:

  * the condition number of the inner objective's (negative) Hessian at
    the found optimum -- for the naive run over the full coefficient
    space, for the gauge-fixed run the Riemannian Hessian restricted to
    the tangent space of the sphere;
  * line-search backtracks (a thrash indicator), outer iterations,
    final train log-likelihood;
  * for the naive run, how far ||b_c|| drifted from 1 (the scale artifact
    the ridge induces).

Results go to results/gauge_ablation.json.

Usage:  python ablation_gauge.py [dataset ...]   (default: diabetes faithful ais)
"""

import json
import os
import sys
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler

from datasets import TIER1
from hermite_gmm import HermiteGMM, hermite_features

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 0

# (dataset, G, m, lambda): true G, degree/lambda near what CV picks
CONFIGS = {
    "diabetes": (3, 3, 1.0),
    "faithful": (2, 4, 10.0),
    "ais": (2, 3, 1.0),
}


def _num_hessian(grad_fn, b, h=1e-5):
    """Symmetrized finite-difference Hessian of Q from its analytic gradient."""
    A = b.size
    H = np.empty((A, A))
    for j in range(A):
        e = np.zeros(A)
        e[j] = h
        H[:, j] = (grad_fn(b + e) - grad_fn(b - e)) / (2 * h)
    return 0.5 * (H + H.T)


def _cond(eigvals):
    ev = np.abs(eigvals)
    return float(ev.max() / max(ev.min(), 1e-300))


def analyze(name, G, m, lam):
    X, _, _ = TIER1[name]()
    X = StandardScaler().fit_transform(X)
    out = {"dataset": name, "G": G, "m": m, "lambda": lam}

    for gauge in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = HermiteGMM(n_components=G, degree=m, reg_lambda=lam,
                               gauge_fix=gauge, max_iter=150,
                               random_state=SEED).fit(X)

        # Hessian conditioning of each component's inner objective at the
        # found b_c, using the final responsibilities.
        log_prob = model._component_log_prob(X)
        from scipy.special import logsumexp
        gamma = np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True))
        conds = []
        for c in range(G):
            Phi = hermite_features(model._whiten(X, c), model.alphas_)
            g_c, N_c = gamma[:, c], gamma[:, c].sum()
            b = model.b_[c]
            grad_fn = lambda bb: model._grad_Q(bb, Phi, g_c, N_c)
            H = _num_hessian(grad_fn, b)
            if gauge:
                # Riemannian Hessian on the sphere, restricted to the
                # tangent space: P (H - (b.grad) I) P with P = I - b b^T.
                P = np.eye(b.size) - np.outer(b, b)
                Hr = P @ (H - (b @ grad_fn(b)) * np.eye(b.size)) @ P
                ev = np.linalg.eigvalsh(0.5 * (Hr + Hr.T))
                ev = ev[np.argsort(np.abs(ev))][1:]   # drop the normal dir (0)
            else:
                ev = np.linalg.eigvalsh(H)
            conds.append(_cond(ev))

        diags = model.inner_diagnostics_
        key = "sphere" if gauge else "naive"
        out[key] = {
            "final_train_ll": model.score(X),
            "n_outer_iter": model.n_iter_,
            "converged": model.converged_,
            "monotone": bool(np.all(np.diff(model.objective_history_) >= -1e-8)),
            "total_backtracks": int(sum(d.n_backtracks for d in diags)),
            "hessian_cond_per_component": conds,
            "hessian_cond_worst": float(np.max(conds)),
            "b_norms": [float(np.linalg.norm(bc)) for bc in model.b_],
        }
    out["cond_ratio_naive_over_sphere"] = (
        out["naive"]["hessian_cond_worst"] / out["sphere"]["hessian_cond_worst"])
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    results = []
    for nm in names:
        G, m, lam = CONFIGS[nm]
        r = analyze(nm, G, m, lam)
        results.append(r)
        print(f"[{nm}] (G={G}, m={m}, lam={lam})")
        for key in ("sphere", "naive"):
            d = r[key]
            print(f"  {key:7s} LL={d['final_train_ll']:.4f} "
                  f"iters={d['n_outer_iter']:3d} "
                  f"backtracks={d['total_backtracks']:4d} "
                  f"cond(H) worst={d['hessian_cond_worst']:.3e} "
                  f"||b||={np.round(d['b_norms'], 3)}")
        print(f"  conditioning ratio naive/sphere = "
              f"{r['cond_ratio_naive_over_sphere']:.1f}x")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "gauge_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
