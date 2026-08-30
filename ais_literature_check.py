"""AIS literature sanity check (Lee & McLachlan 2013, EMMIXuskew JSS paper).

Their AIS example clusters the 202 athletes by sex with G=2 on TWO
variables only: Height and Body fat percentage. Reported correct
allocations out of 202 (their Section 4.3):

    unrestricted skew-t (FM-uMST)   183
    mixsmsn skew-t                  162
    restricted skew-t (EMMIX-skew)  157
    symmetric t mixture (FM-MT)     165   (77 + 88 from their table)

This script fits plain GMM, symmetric t-mix, and Hermite-GMM at G=2 on
the same two variables (ht, pcBfat in the DAAG encoding) and reports
correct allocations for a direct comparison. Writes
results/ais_literature.json.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from studenttmixture import EMStudentMixture

from datasets import DATA_DIR
from hermite_gmm import HermiteGMM

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LITERATURE = {"skew-t (FM-uMST)": 183, "mixsmsn skew-t": 162,
              "restricted skew-t (EMMIX-skew)": 157, "sym t (FM-MT)": 165}


def n_correct(y, yhat):
    """Correct allocations under the best label permutation."""
    cm = confusion_matrix(y, yhat)
    r, c = linear_sum_assignment(-cm)
    return int(cm[r, c].sum())


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "ais.csv"))
    X = df[["ht", "pcBfat"]].to_numpy(float)
    y = pd.Categorical(df["sex"]).codes.astype(int)
    Xs = StandardScaler().fit_transform(X)

    out = {"n": len(y), "variables": ["ht", "pcBfat"], "G": 2,
           "literature_correct_of_202": LITERATURE}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm = GaussianMixture(n_components=2, n_init=10, random_state=0).fit(Xs)
        tmix = EMStudentMixture(n_components=2, n_init=10, fixed_df=False,
                                random_state=123)
        tmix.fit(Xs)
        from benchmark import _cv_ll
        best = max(((_cv_ll(Xs, 2, m, lam), m, lam)
                    for m in (0, 3, 4) for lam in (0.1, 1.0, 10.0)))
        _, m, lam = best
        hg = HermiteGMM(n_components=2, degree=m, reg_lambda=lam,
                        n_init=5, random_state=0).fit(Xs)

    for label, model in [("plain GMM", gmm), ("t-mix (symmetric)", tmix),
                         (f"Hermite-GMM (m={m}, lam={lam})", hg)]:
        yhat = model.predict(Xs)
        out[label] = {"correct_of_202": n_correct(y, yhat),
                      "ari": float(adjusted_rand_score(y, yhat))}
        print(f"{label:35s} correct {out[label]['correct_of_202']}/202  "
              f"ARI {out[label]['ari']:.3f}")
    print("literature:", LITERATURE)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ais_literature.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
