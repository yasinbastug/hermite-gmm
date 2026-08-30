# Hermite-GMM

Implementation and benchmark of the Hermite-corrected Gaussian Mixture Model
from `hermite_gmm_explained.md` / `claude_code_prompt.md`: a GMM in which each
component is multiplied by a positive, exactly normalized correction
`rho_c(u) = (g_c(u)^2 + eps) / (||b_c||^2 + eps)`, with `g_c` a polynomial in
the component's own whitened coordinates expanded in the orthonormal
probabilists' Hermite basis. Setting `b_c = (1, 0, ..., 0)` recovers a plain
GMM exactly (strict nested extension).

## Layout

| file | purpose |
|---|---|
| `hermite_gmm.py` | The model: `HermiteGMM(n_components, degree, reg_lambda, ...)` with `fit / predict / predict_proba / score_samples / score / bic / degree_energies`. |
| `test_hermite_gmm.py` | Sanity tests: basis orthonormality (quadrature), exact GMM reduction vs sklearn, monotone EM log-likelihood on iris, 1-D synthetic skew recovery. |
| `datasets.py` | Tier-1 loaders (iris, faithful, wine13, diabetes, crabs, banknote, thyroid, ais, olive, wine27); files cached in `data/`, provenance in the docstring. |
| `benchmark.py` | Plain GMM (BIC over G=1..10) vs Hermite-GMM ((G, m, lambda) by 5-fold CV held-out log-likelihood). Writes `results/<dataset>.json`. |
| `baseline_tmix.py` | Symmetric Student's-t mixture baseline (`studenttmixture`). **Not** a skew-t (no R available; EMMIXskew archived on CRAN) — labeled accordingly. |
| `ablation_gauge.py` | Section-33 gauge-fixing ablation: sphere-constrained vs naive `b` parameterization; Hessian condition numbers, `||b||` drift. |
| `make_report.py` | Assembles `results/*.json` into `results/REPORT.md`. |

## Setup / run

```bash
python3.11 -m venv .venv
.venv/bin/pip install numpy scipy scikit-learn pandas pyreadr studenttmixture
.venv/bin/python test_hermite_gmm.py       # sanity checks
.venv/bin/python benchmark.py --all        # main comparison (writes results/)
.venv/bin/python baseline_tmix.py --all    # t-mixture baseline
.venv/bin/python ablation_gauge.py         # gauge-fixing ablation
.venv/bin/python make_report.py            # results/REPORT.md
```

## Implementation notes (mapping to the design doc)

- **Whitening** (Sec. 14): inverse Cholesky factor `L^{-1}` per component —
  deterministic across EM iterations; coefficients are convention-dependent
  but the reported degree-block energies `sum_{|alpha|=d} b^2` are invariant.
- **Basis**: normalized recurrence
  `psi_{d+1} = (u psi_d - sqrt(d) psi_{d-1}) / sqrt(d+1)`; product features
  vectorized over the whole index set (no per-sample Python loops).
- **Degrees 1–2 excluded** by default (Sec. 26), toggleable via
  `exclude_degrees_12=False`.
- **Gauge fixing** (Sec. 33): default `gauge_fix=True` keeps `||b_c|| = 1`
  and takes projected-gradient steps (`grad - (b.grad) b`) with Armijo
  backtracking and renormalization. `gauge_fix=False` is the naive raw-`b`
  ascent, kept for the ablation. Both only accept objective-improving steps.
- **EM** (Sec. 34): block-coordinate generalized EM; a few warm-started inner
  steps for each `b_c` per outer iteration (`n_inner=5`). The weighted
  Gaussian M-step is not exactly optimal for this model (u depends on
  mu, Sigma), so `safe_gaussian_update=True` rejects a Gaussian block move
  that would lower the observed log-likelihood, making the outer loop
  provably monotone — the monotonicity assertion in the tests relies on it.
- **Regularization** (Sec. 35): `-lambda * sum |alpha|^p b_alpha^2`, `p=2`
  default.
- **BIC**: effective parameter count charges `|A_m| - 1` per component for
  the Hermite block (the unit-norm constraint removes the scale direction).
- **eps**: fixed `1e-6` by default; with the gauge fixed, `||b|| = 1` so the
  scale-aware `1e-3 ||b||^2` variant is just a constant too.
