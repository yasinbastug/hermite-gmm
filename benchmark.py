"""Tier-1 benchmark: plain GMM vs Hermite-GMM (protocol from the task spec).

Per dataset:
  1. Standardize features.
  2. Plain GMM over G = 1..10, select G by BIC; ARI at BIC-selected G and
     at the true G.
  3. Hermite-GMM over the same G grid, a configurable degree grid
     (default m in {0, 3, 4, ..., 12}; m=0 = plain GMM, so the reduction
     is inside the grid) and a lambda grid, selected by 5-fold CV
     held-out log-likelihood. ARI at the CV-selected settings and at the
     true G.
  4. Headline flag: Hermite at true G beats the plain GMM's BIC-selected
     model (which used MORE components) on held-out log-likelihood.
     NOTE this is an ORACLE-G comparison for Hermite; the blind
     pipeline-vs-pipeline comparison is GMM ARI@BIC-G vs HGMM ARI@CV.
  5. Degree-block energies of the true-G Hermite fit.

Feasibility. The basis size is |A_m| = 1 + sum_{d=3..m} C(k+d-1, d),
which explodes in both k and m, and the feature matrix Phi is
n x |A_m| float64. Combinations exceeding --max-phi-gb (or --max-basis)
are SKIPPED and recorded in the result JSON under "skipped_degrees" --
they are not silently dropped. Run --dry-run first to see exactly what
your grid will attempt.

Resume. Every CV grid cell is cached in results/_cv_cache/<dataset>.json
keyed by (G, m, lambda), so an interrupted run resumes where it stopped.
Completed datasets are skipped unless --force.

Usage:
  python benchmark.py --dry-run --all            # plan: sizes, memory, cost
  python benchmark.py --all --jobs 8             # full run, 8 processes
  python benchmark.py ais wine13 --degrees 0,3,4,5,6
  python benchmark.py --all --max-phi-gb 16 --jobs 4
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from datasets import TIER1
from hermite_gmm import HermiteGMM, multi_indices

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "_cv_cache")

G_GRID = list(range(1, 11))
DEFAULT_DEGREES = [0] + list(range(3, 13))     # 0, 3, 4, ..., 12
LAMBDAS = [0.1, 1.0, 10.0]
N_FOLDS = 5
SEED = 0

# Defaults chosen so a 16 GB machine is safe. Raise --max-phi-gb on a
# bigger box; see RUNNING.md for the per-dataset memory table.
DEFAULT_MAX_PHI_GB = 4.0
DEFAULT_MAX_BASIS = 2_000_000

# Cost of one CV cell per (n x |A_m|) unit, measured on an M-series laptop
# across six (dataset, m) points spanning |A_m| = 6k..350k. The implied
# rate ranges 11.8-41.0 us (median 15.8): the spread is driven mainly by
# how many EM iterations a given cell needs, which is not predictable from
# |A_m|, so --dry-run reports a RANGE rather than false precision.
SEC_PER_UNIT_LOW = 12e-6
SEC_PER_UNIT_HIGH = 41e-6


def basis_size(k, m):
    """|A_m| with degrees 1-2 excluded, without materializing the list."""
    if m == 0:
        return 1
    return 1 + sum(math.comb(k + d - 1, d) for d in range(3, m + 1))


def phi_bytes(n, k, m):
    """Bytes for one n x |A_m| float64 feature matrix."""
    return n * basis_size(k, m) * 8


def feasible_degrees(n, k, degrees, max_phi_gb, max_basis):
    """Split `degrees` into (kept, skipped) under the memory/count caps."""
    keep, skip = [], []
    for m in degrees:
        A, gb = basis_size(k, m), phi_bytes(n, k, m) / 1e9
        if m > 0 and (gb > max_phi_gb or A > max_basis):
            skip.append({"m": m, "basis": A, "phi_gb": round(gb, 3)})
        else:
            keep.append(m)
    return keep, skip


# ---------------------------------------------------------------------------
# CV evaluation (module-level so joblib can pickle it)
# ---------------------------------------------------------------------------

def _fit_hermite(X, G, m, lam, n_init=2, max_iter=150, reg_power=2,
                 precondition=False):
    return HermiteGMM(n_components=G, degree=m, reg_lambda=lam,
                      reg_power=reg_power, precondition=precondition,
                      max_iter=max_iter, tol=1e-5,
                      n_init=n_init, random_state=SEED).fit(X)


def _cv_ll(X, G, m, lam, reg_power=2, precondition=False):
    """Mean held-out per-sample log-likelihood over K folds."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in kf.split(X):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _fit_hermite(X[tr], G, m, lam, reg_power=reg_power,
                                     precondition=precondition)
            scores.append(model.score(X[te]))
        except Exception:
            return -np.inf
    return float(np.mean(scores))


def _cv_cell(X, G, m, lam, reg_power=2, precondition=False):
    """One grid cell -> (key, cv_ll). Returns -inf on any failure."""
    try:
        return f"{G}|{m}|{lam}", _cv_ll(X, G, m, lam, reg_power,
                                        precondition)
    except Exception:
        return f"{G}|{m}|{lam}", -np.inf


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_name(name, reg_power, precondition=False):
    """Cache file per (dataset, reg_power, precondition).

    Defaults keep the bare name so existing caches stay valid; any
    non-default option gets its own file, because a cached cv_ll computed
    under different optimizer settings is NOT interchangeable.
    """
    suffix = "" if reg_power == 2 else f"_w{reg_power}"
    if precondition:
        suffix += "_pc"
    return name + suffix


def _load_cache(name):
    path = os.path.join(CACHE_DIR, f"{name}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(name, cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)          # atomic: survives an interrupt mid-write


# ---------------------------------------------------------------------------
# Main per-dataset routine
# ---------------------------------------------------------------------------

def scale_ratio(X):
    """Largest / smallest column standard deviation.

    Protocol step 1 says to standardize only "if the original units are
    wildly different across columns (check)". This records the evidence
    for that decision instead of assuming it. Empirically every Tier-1
    dataset has a ratio >= 3, so all are standardized -- including Olive,
    whose fatty-acid percentages the prompt guessed "may already be
    comparable" but which actually spans 31x (oleic acid ~7000 vs minor
    acids ~15, same unit, very different magnitude).
    """
    sd = X.std(axis=0)
    return float(sd.max() / max(sd.min(), 1e-12))


def run_dataset(name, degrees, max_phi_gb, max_basis, jobs, reg_power=2,
                precondition=False):
    X, y, true_G = TIER1[name]()
    ratio = scale_ratio(X)
    X = StandardScaler().fit_transform(X)
    n, k = X.shape
    t_start = time.time()
    res = {"dataset": name, "n": n, "p": k, "true_G": true_G,
           "feature_scale_ratio": round(ratio, 1), "standardized": True,
           "reg_power": reg_power, "precondition": precondition}

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
        "cv_ll_at_G_bic": _cv_ll(X, G_bic, 0, 0.0, reg_power, precondition),
        "cv_ll_at_true_G": _cv_ll(X, true_G, 0, 0.0, reg_power, precondition),
    }
    print(f"[{name}] GMM: BIC selects G={G_bic} "
          f"(ARI {res['gmm']['ari_at_G_bic']}), "
          f"ARI@trueG {res['gmm']['ari_at_true_G']}", flush=True)

    # ---- feasibility ----------------------------------------------------
    keep, skip = feasible_degrees(n, k, degrees, max_phi_gb, max_basis)
    res["degrees_searched"] = keep
    res["skipped_degrees"] = skip
    if skip:
        worst = ", ".join(f"m={s['m']} ({s['basis']:,} fns, "
                          f"{s['phi_gb']:.1f}GB)" for s in skip)
        print(f"[{name}] SKIPPING infeasible degrees: {worst}", flush=True)

    # ---- Hermite-GMM: CV over (G, m, lambda), cached + parallel ---------
    cache = _load_cache(_cache_name(name, reg_power, precondition))
    combos = [(G, m, lam)
              for G in G_GRID for m in keep
              for lam in (LAMBDAS if m > 0 else [0.0])]
    todo = [c for c in combos if f"{c[0]}|{c[1]}|{c[2]}" not in cache]
    print(f"[{name}] CV grid: {len(combos)} cells, {len(todo)} to compute "
          f"({len(combos) - len(todo)} cached)", flush=True)

    if todo:
        out = Parallel(n_jobs=jobs, verbose=5)(
            delayed(_cv_cell)(X, G, m, lam, reg_power, precondition)
            for G, m, lam in todo)
        for key, val in out:
            cache[key] = val
        _save_cache(_cache_name(name, reg_power, precondition), cache)

    grid = {}      # G -> (cv_ll, m, lam)
    for G in G_GRID:
        best = (-np.inf, 0, 0.0)
        for m in keep:
            for lam in (LAMBDAS if m > 0 else [0.0]):
                s = cache.get(f"{G}|{m}|{lam}", -np.inf)
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
            model = _fit_hermite(X, G, m, lam, n_init=5,
                                 reg_power=reg_power,
                                 precondition=precondition)
        return model, {
            "G": G, "m": m, "lambda": lam, "cv_ll": cv_ll,
            "bic": float(model.bic(X)),
            "train_ll": model.score(X),
            "n_iter": model.n_iter_,
            "n_basis": len(model.alphas_),
            "monotone": bool(np.all(np.diff(model.objective_history_) >= -1e-8)),
            "ari": None if y is None else
                float(adjusted_rand_score(y, model.predict(X))),
            "degree_energies": model.degree_energies().tolist(),
        }

    _, entry_cv = _refit_entry(G_cv)
    _, entry_tg = _refit_entry(true_G)
    res["hermite"] = {"cv_selected": entry_cv, "at_true_G": entry_tg}

    # ---- headline flag (oracle-G; see module docstring) -----------------
    res["headline"] = bool(
        G_bic > true_G
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


def _already_covers(name, degrees, max_phi_gb, max_basis):
    """True if an existing result already searched every feasible degree.

    Avoids the footgun where extending --degrees silently skips datasets
    that have a result file from a narrower grid: a stale result is only
    reused when it covers everything the new grid would actually attempt.
    """
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            prev = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    X, _, _ = TIER1[name]()
    n, k = X.shape
    want, _ = feasible_degrees(n, k, degrees, max_phi_gb, max_basis)
    return set(want).issubset(set(prev.get("degrees_searched", [])))


def dry_run(names, degrees, max_phi_gb, max_basis):
    """Print the plan: basis sizes, memory, and grid cost. No fitting."""
    print(f"Degree grid: {degrees}")
    print(f"Caps: max_phi_gb={max_phi_gb}, max_basis={max_basis:,}\n")
    hdr = (f"{'dataset':10s} {'n':>4s} {'p':>3s} {'feasible m':>28s} "
           f"{'skipped m':>16s} {'cells':>7s} {'peak Phi':>10s} {'core-h':>14s}")
    print(hdr); print("-" * len(hdr))
    total_cells, lo_hours, hi_hours = 0, 0.0, 0.0
    for name in names:
        X, y, G = TIER1[name]()
        n, k = X.shape
        keep, skip = feasible_degrees(n, k, degrees, max_phi_gb, max_basis)
        cells = sum(len(LAMBDAS) if m > 0 else 1 for m in keep) * len(G_GRID)
        total_cells += cells
        peak = max((phi_bytes(n, k, m) for m in keep), default=0) / 1e9
        units = sum((len(LAMBDAS) if m > 0 else 1) * n * basis_size(k, m)
                    * (G_ / 3.0) for m in keep for G_ in G_GRID)
        lo, hi = units * SEC_PER_UNIT_LOW / 3600, units * SEC_PER_UNIT_HIGH / 3600
        lo_hours += lo; hi_hours += hi
        ks = ",".join(str(m) for m in keep)
        ss = ",".join(str(s["m"]) for s in skip) or "-"
        print(f"{name:10s} {n:4d} {k:3d} {ks:>28s} {ss:>16s} {cells:7d} "
              f"{peak:9.2f}G {lo:6.0f}-{hi:<7.0f}")
    print(f"\nTotal CV cells: {total_cells:,} "
          f"(x{N_FOLDS} folds = {total_cells * N_FOLDS:,} model fits)")
    print(f"Estimated cost: {lo_hours:.0f}-{hi_hours:.0f} core-hours"
          + "".join(f" | {j} jobs ~ {lo_hours / j:.0f}-{hi_hours / j:.0f}h"
                    for j in (8, 16, 32)))
    print("Range spans a measured 12-41us per (n x |A_m|) unit per cell; the "
          "spread is EM-iteration count, not basis size, so plan for the "
          "upper end.")
    print("Cells are cached and resumable -- interrupting costs nothing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="dataset names (default: all)")
    ap.add_argument("--all", action="store_true", help="run every dataset")
    ap.add_argument("--degrees", default=None,
                    help="comma-separated degree grid "
                         f"(default: {','.join(map(str, DEFAULT_DEGREES))})")
    ap.add_argument("--max-phi-gb", type=float, default=DEFAULT_MAX_PHI_GB,
                    help=f"skip (dataset, m) whose n x |A| x 8 exceeds this "
                         f"(default {DEFAULT_MAX_PHI_GB})")
    ap.add_argument("--max-basis", type=int, default=DEFAULT_MAX_BASIS,
                    help="skip (dataset, m) with more basis functions")
    ap.add_argument("--reg-power", type=int, default=2, choices=[2, 4],
                    help="degree weight exponent p in w_alpha=|alpha|^p "
                         "(2 = default; 4 penalizes high degrees 16x harder "
                         "and suppresses the high-m mode-invention failure). "
                         "Uses a separate CV cache per power.")
    ap.add_argument("--precondition", action="store_true",
                    help="use per-degree-block preconditioning of the b_c "
                         "search direction (Sec. 14). OFF by default: "
                         "measured across 8 configs it hurt more often than "
                         "it helped, because gauge fixing already leaves the "
                         "inner Hessian at condition number 1.6-4.8.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel processes for the CV grid (default 1). "
                         "Peak RAM is roughly jobs x peak-Phi.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without fitting")
    ap.add_argument("--force", action="store_true",
                    help="recompute datasets that already have results")
    args = ap.parse_args()

    names = list(TIER1) if (args.all or not args.datasets) else args.datasets
    bad = [n for n in names if n not in TIER1]
    if bad:
        ap.error(f"unknown dataset(s): {bad}. Available: {list(TIER1)}")

    degrees = (DEFAULT_DEGREES if args.degrees is None
               else [int(d) for d in args.degrees.split(",")])

    if args.dry_run:
        dry_run(names, degrees, args.max_phi_gb, args.max_basis)
        return

    for nm in names:
        if not args.force and _already_covers(nm, degrees, args.max_phi_gb,
                                              args.max_basis):
            print(f"[{nm}] already computed over this degree grid; skipping "
                  f"(use --force to redo)", flush=True)
            continue
        try:
            run_dataset(nm, degrees, args.max_phi_gb, args.max_basis,
                        args.jobs, args.reg_power, args.precondition)
        except Exception:
            print(f"[{nm}] FAILED:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
