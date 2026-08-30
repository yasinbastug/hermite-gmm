"""Student's-t mixture baseline (fallback for the skew-t comparison).

IMPORTANT CAVEAT (stated per the task spec): no R is available on this
machine and EMMIXskew/EMMIXuskew are archived on CRAN, so this uses the
pure-Python ``studenttmixture`` package -- a SYMMETRIC multivariate
Student's-t mixture. It captures heavy tails but NOT skewness, so it is a
weaker competitor than a true skew-t mixture and is labeled "t-mix (symmetric)"
in all reports.

Protocol mirrors the plain-GMM arm: fit over G = 1..10 with estimated
degrees of freedom, select G by BIC, report ARI at BIC-selected G and at
the true G. Results go to results/<dataset>_tmix.json.

Usage:  python baseline_tmix.py [dataset ...|--all]
"""

import json
import os
import sys
import traceback
import warnings

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from studenttmixture import EMStudentMixture

from datasets import TIER1

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
G_GRID = list(range(1, 11))
SEED = 123


def run_dataset(name):
    X, y, true_G = TIER1[name]()
    X = StandardScaler().fit_transform(X)

    fits, bics = {}, {}
    for G in G_GRID:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = EMStudentMixture(n_components=G, n_init=5,
                                     fixed_df=False, random_state=SEED,
                                     max_iter=1000)
                m.fit(X)
            fits[G], bics[G] = m, float(m.bic(X))
        except Exception:
            bics[G] = np.inf
    G_bic = min(bics, key=bics.get)

    def _ari(G):
        if y is None or G not in fits:
            return None
        return float(adjusted_rand_score(y, fits[G].predict(X)))

    res = {
        "dataset": name,
        "model": "t-mixture (symmetric, studenttmixture; NOT skew-t)",
        "G_bic": G_bic,
        "bic_at_G_bic": bics[G_bic],
        "bic_at_true_G": bics.get(true_G, None),
        "ari_at_G_bic": _ari(G_bic),
        "ari_at_true_G": _ari(true_G),
        "df_at_true_G": (fits[true_G].degrees_of_freedom.tolist()
                         if true_G in fits else None),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{name}_tmix.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] t-mix: BIC selects G={G_bic} (ARI {res['ari_at_G_bic']}), "
          f"ARI@trueG {res['ari_at_true_G']}", flush=True)
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    names = [n for n in (list(TIER1) if (not args or args == ["--all"]) else args)
             if TIER1[n]()[1] is not None or True]
    for nm in names:
        try:
            run_dataset(nm)
        except Exception:
            print(f"[{nm}] FAILED:\n{traceback.format_exc()}", flush=True)
