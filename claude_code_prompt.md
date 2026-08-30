# Implement and benchmark the Hermite-GMM

## Context

I have a design document, `hermite_gmm_explained.md` (attached in this folder — read it first,
especially Parts VII–IX, Section 24 for the exact model equation, Section 33 for a gauge-fixing
issue you must not skip, and Section 34 for the algorithm). It's a from-scratch explainer, not a
paper, so it's verbose — skim for the boxed equations and the numbered algorithm rather than
reading linearly.

The model: a Gaussian Mixture Model where each component is multiplied by a positive, exactly-
normalized correction built from Hermite polynomials in the component's own whitened coordinates.
Setting all correction coefficients to the baseline recovers a plain GMM exactly, so it's a strict,
nested extension — every fit should be at least as good as plain GMM on training likelihood.

## The exact model (Section 24)

$$p_\theta(x) = \sum_{c=1}^{C}\pi_c\,\mathcal{N}(x;\mu_c,\Sigma_c)\;
\frac{\big(\sum_{\alpha\in\mathcal{A}_m}b_{c,\alpha}\,\phi_\alpha(\Sigma_c^{-1/2}(x-\mu_c))\big)^2+\varepsilon}
{\sum_{\alpha\in\mathcal{A}_m}b_{c,\alpha}^2+\varepsilon}$$

- $\mathcal{A}_m = \{\alpha \in \mathbb{N}_0^k : |\alpha|\le m\}$, product Hermite basis
  $\phi_\alpha(u)=\prod_j\psi_{\alpha_j}(u_j)$, $\psi_n = \mathrm{He}_n/\sqrt{n!}$ (probabilists'
  Hermite polynomials, normalized).
- Setting $b_{c,0}=1$, all other $b_{c,\alpha}=0$ gives $\rho_c\equiv1$ — exactly a GMM. Use this
  as the initialization.
- **Gauge freedom (Section 33 — do not skip):** $\rho_c$ is invariant under $b_c \to c\cdot b_c$
  for any scalar $c\ne0$. This makes naive L-BFGS on raw $b_c$ ill-conditioned. Fix it by
  constraining $\|b_c\|=1$ and optimizing on the unit sphere (so $\rho_c = g_c^2+\varepsilon$ with
  no denominator), using the projected gradient $\nabla Q - (b^\top\nabla Q)b$. Confirm this
  actually improves conditioning versus the naive unconstrained parameterization — report both if
  easy, so we know the fix mattered.
- **Degree 1–2 redundancy (Section 26):** degrees 1 and 2 duplicate what $\mu_c,\Sigma_c$ already
  express and create a flat likelihood ridge. Default to **excluding degrees 1 and 2** from
  $\mathcal{A}_m$ (start Hermite corrections at degree 3) rather than just regularizing them — make
  this a toggleable option, but exclusion should be the default.
- **Regularization (Section 35):** penalize coefficients by $w_\alpha = |\alpha|^2$ (default) or
  $|\alpha|^4$, added to the $b_c$-objective as $-\lambda\sum_\alpha w_\alpha b_{c,\alpha}^2$.
- **$\varepsilon$ floor:** needed because $g_c$ is a polynomial with real roots, and without the
  floor the density hits exactly zero there. Use something scale-aware, e.g.
  $\varepsilon \approx 10^{-3}\|b_c\|^2$ (or just a small fixed constant like $10^{-6}$ to start —
  whichever is simpler to get working first).

## Algorithm (Section 34)

Block-coordinate EM / generalized EM:

1. Fit a standard GMM (sklearn's `GaussianMixture` is fine) → initial $\pi_c,\mu_c,\Sigma_c$.
2. Initialize $b_c$ at the GMM baseline.
3. Repeat:
   - E-step: compute $u_{ic}=\Sigma_c^{-1/2}(x_i-\mu_c)$, evaluate the basis $\Phi_i$, compute
     $q_c(x_i)$, $\rho_c(u_{ic})$, then responsibilities $\gamma_{ic}$ (use log-sum-exp).
   - Update $\pi_c$, and $\mu_c,\Sigma_c$ via the standard weighted-Gaussian M-step formulas
     (+ ridge $\eta I_k$ on the covariance).
   - Update $b_c$ by a few steps of projected gradient / L-BFGS-on-sphere, warm-started from the
     previous iterate. Don't fully converge the inner problem each outer iteration — a handful of
     steps is enough since responsibilities are about to change anyway.
4. Stop on log-likelihood improvement below tolerance. Track log-likelihood every iteration and
   assert it's non-decreasing (modulo small numerical slack) as a correctness check — if it isn't,
   something in the $b_c$ update or gauge-fixing is wrong.
5. Hard assignment via $\arg\max_c \gamma_{ic}$ at the end for evaluation against true labels.

Whitening convention: use the **inverse Cholesky factor** for $\Sigma_c^{-1/2}$ (Section 14) —
deterministic across iterations, unlike eigendecomposition, which can flip signs/reorder near-
degenerate eigenvalues between calls and silently rotate the coordinate frame mid-optimization.

## What to build

1. A clean, well-documented Python module (`hermite_gmm.py`) implementing the above as a class
   with an sklearn-like interface: `fit(X)`, `predict(X)`, `score_samples(X)`, `bic()`. Vectorize
   the basis evaluation and E-step; don't loop over data points in Python.
2. A small test script verifying:
   - At the GMM-baseline initialization, fitting for 0 iterations reproduces `sklearn`'s
     `GaussianMixture` log-likelihood on a toy dataset (sanity check that the reduction is exact).
   - Log-likelihood is monotone non-decreasing across EM iterations on at least one real dataset.
   - A synthetic 1-D skewed test: sample from a known skewed distribution, confirm the fitted
     Hermite correction recovers positive skew, and that a plain GMM fit to the same data has
     visibly worse log-likelihood / more components needed at comparable BIC.

## Tier 1 benchmark datasets (initial test set)

These are the standard small model-based-clustering benchmarks, chosen because they're documented
cases where plain GMM/`mclust` clustering is known to struggle (fragmenting a real cluster into
extra ellipsoidal pieces, or over/under-selecting $G$ by BIC) — exactly the failure mode this model
targets.

| Dataset | n | p | true G | where to get it |
|---|---|---|---|---|
| Crabs | 200 | 5 | 2 (or 4, by sex×colour) | `MASS::crabs` (R) — or find the CSV mirror |
| Banknote (Swiss) | 200 | 6 | 2 | `mclust::banknote` (R) |
| AIS (Australian athletes) | 202 | 11 | 2 | `sn::ais` (R) or search "AIS dataset athletes" |
| Diabetes | 145 | 3 | 3 | `mclust::diabetes` (R) |
| Thyroid | 215 | 5 | 3 | `mclust::thyroid` (R) |
| Wine (13 vars) | 178 | 13 | 3 | UCI Wine dataset / `pgmm::wine` |
| Wine (27 vars) | 178 | 27 | 3 | `pgmm::wine` (R) — same wines, more chemistry vars |
| Olive oils | 572 | 8 | 3 (region) or 9 (area) | `pgmm::olive` (R) |
| Old Faithful | 272 | 2 | 2 (unlabeled) | built into R (`faithful`), also on UCI/many mirrors |
| Iris | 150 | 4 | 3 | trivially available (sklearn.datasets, UCI, seaborn) |

**Practical note on data access:** several of these live only in R packages. Options, in order of
preference: (a) check if `rdatasets` (Python) or a pinned CSV mirror already has them — search
first; (b) if you have R available, extract with `write.csv` and load the CSVs from Python; (c) as
a fallback, Iris, Old Faithful, and Wine (13-var, standard UCI version) are trivially available in
Python already (`sklearn.datasets`) and can be a first pass while you sort out access to the rest.
Don't spend excessive time on data-wrangling before getting the pipeline working — start with
whichever 2–3 datasets are easiest to obtain, confirm the full pipeline runs end to end, then
fill in the rest.

## Evaluation protocol

For each dataset:

1. Standardize/scale features if the original units are wildly different across columns (check —
   some of these, like Olive's fatty acid percentages, may already be comparable).
2. Fit plain GMM (`sklearn.mixture.GaussianMixture`) across a small grid of $G$ (say 1–10),
   select by BIC, report **ARI** against true labels at both the BIC-selected $G$ and at the
   true $G$ (both numbers matter — Section on Crabs in particular is a known case where BIC
   over-selects $G$ substantially).
3. Fit Hermite-GMM at the same $G$ values, small grid of degree $m\in\{0,3,4\}$ (0 = plain GMM,
   confirms the reduction) and regularization $\lambda$, select by held-out log-likelihood via
   5-fold CV or a simple train/validation split (small $n$ here, so be mindful of using CV rather
   than a single held-out split where possible). Report ARI at BIC/CV-selected settings and at
   true $G$.
4. Report the comparison as a table: dataset, plain-GMM ARI, Hermite-GMM ARI, and the selected
   $(G, m, \lambda)$ for each. Flag any dataset where Hermite-GMM's *fitted density* at true $G$
   beats plain GMM's BIC-selected model that used *more* components — that's the headline result
   this method is going for (fewer, more flexible components beating more, rigid ones).
5. Also report a degree-block diagnostic for at least one dataset where Hermite-GMM helps: the
   per-degree energy $\sum_{|\alpha|=d} b_{c,\alpha}^2$ for each fitted component, to show *which*
   kind of non-Gaussianity (skew vs. kurtosis vs. interaction) the model found and that it's
   concentrated in a low degree rather than spread across many high-degree terms (which would
   suggest overfitting instead of genuine structure).
6. Once the above is working, add the skew-t baseline described next and fold it into the same
   comparison table.

## Skew-t baseline comparison

In addition to plain GMM, also fit a **mixture of skew-t distributions** as a baseline. This is
the real competitor family for this model in the literature — it buys skewness cheaply (roughly
one extra shape parameter per component, versus this model's $\binom{k+m}{m}$), so the honest
framing for our results is: *can this model win on generality (arbitrary shape, not just skew) and
on interpretability (degree-block diagnostics), even where skew-t wins on parameter-efficiency for
pure skew?* — not "does it beat skew-t outright."

**Primary route — R via `rpy2`:** the standard packages are `EMMIXskew` / `EMMIXuskew`
(restricted and unrestricted multivariate skew-t mixtures, Lee & McLachlan) and `teigen`
(parsimonious t-mixtures, symmetric — useful as a second, simpler baseline for heavy tails without
skew). Both are on CRAN. Notably, `EMMIXuskew`'s own paper uses the **AIS dataset** — already in
our Tier 1 list — as a worked example, so that's a natural first target for this comparison; check
its exact reported clustering result on AIS if you want a literature number to sanity-check
against, not just our own re-run. Don't assume the exact function signature — check
`?EmSkew`/`?teigen` in R (or the package vignette/CRAN PDF) once you have the package installed,
since APIs can drift between versions.

Setup: `install.packages(c("EMMIXskew","teigen"))` in R, `pip install rpy2` in Python, then call
via `rpy2.robjects`. If R/rpy2 setup turns out to be a time sink, don't block the rest of the
pipeline on it — get plain-GMM-vs-Hermite-GMM working and reported first, then add skew-t.

**Fallback — pure Python, no R required:** if R access is unavailable or too much friction,
`pip install studenttmixture` gives a symmetric multivariate Student's-t mixture (EM-based, similar
interface to `sklearn.mixture.GaussianMixture`). This is **not a full substitute** — it gets heavy
tails but not skewness — so if you use it instead of a true skew-t package, say so explicitly in
the results rather than presenting it as an equivalent comparison. There's also a general-purpose
`pip install Mixture-Models` library (GMM, PGMM, Student's-t, various parsimonious variants via
gradient-based/AD inference rather than EM) — check its docs for whether it has added a skew
variant since publication; if so it'd be a cleaner single-ecosystem comparison than mixing R and
Python.

Report skew-t's ARI alongside plain-GMM's and Hermite-GMM's in the same comparison table (dataset,
plain-GMM ARI, skew-t ARI, Hermite-GMM ARI, selected settings for each). Also report each model's
selected/fitted number of components side by side — since the argument for this model partly rests
on getting comparable or better fit with a similar or smaller component count, not just better ARI
in isolation.

## Priorities, in order

1. Get the model class correct and passing the sanity tests (GMM-reduction, monotone likelihood,
   synthetic skew recovery) before touching real data.
2. Get *any* 2–3 Tier 1 datasets running end to end with the plain-GMM-vs-Hermite-GMM comparison
   table (skew-t not required yet).
3. Fill in the remaining datasets.
4. Add the skew-t baseline (R via rpy2, or the Python fallback if that's a blocker) to the
   comparison table.
5. Only then: gauge-fixing ablation, degree-block diagnostics, CV tuning refinement.

Ask me if anything in the design doc is ambiguous rather than guessing silently on anything that
affects correctness (the gauge-fixing and degree-1/2 exclusion are the two places worth double-
checking against the doc rather than assuming).
