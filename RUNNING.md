# Running the extended-degree benchmark

Instructions for running the full `m ∈ {0, 3, 4, …, 12}` sweep on a larger
machine and delivering the results back.

The previous published run only searched `m ∈ {0, 3, 4}`. This sweep tests
whether higher-degree Hermite corrections (degree 5+ = "fine local shape",
per Section 26 of the design doc) buy anything the earlier grid could not see.

---

## 1. Setup

```bash
git clone https://github.com/yasinbastug/hermite-gmm.git
cd hermite-gmm
```

```bash
python3.11 -m venv .venv && .venv/bin/pip install -U pip
```

```bash
.venv/bin/pip install -r requirements.txt
```

**Python 3.10, 3.11, or 3.12.** 3.10 and 3.11 are both verified: the full test
suite passes and CV log-likelihoods agree to four decimals across them, even
though pip resolves different dependency versions (3.10 gets numpy 2.2 /
pandas 2.3 / sklearn 1.7; 3.11 gets numpy 2.4 / pandas 3.0 / sklearn 1.9).
3.13/3.14 may lack wheels for some dependencies.

If `studenttmixture` fails to build, delete that line from `requirements.txt`
and reinstall — it is only used by the t-mixture baseline, not the degree sweep.

Confirm the install and the datasets:

```bash
.venv/bin/python test_hermite_gmm.py && .venv/bin/python datasets.py
```

All five sanity tests must pass (exact GMM reduction, monotone EM objective,
BIC effective-d.o.f. shrinkage, basis orthonormality, synthetic skew recovery),
and all ten datasets must load. **If anything fails here, stop** — the data files
are committed to the repo, so failures mean an environment problem, not a
missing download.

---

## 2. Plan the run first

```bash
.venv/bin/python benchmark.py --all --dry-run
```

This prints, per dataset, which degrees are feasible, which get skipped, the
number of CV grid cells, and peak memory — without fitting anything. At the
default 4 GB cap you should see 2,860 cells / 14,300 model fits, with all
degrees feasible except `wine13` (m=11,12) and `wine27` (m≥7).

---

## 3. Run it

The cost is wildly uneven: six datasets are nearly free, four dominate. Run them
in tiers so you have deliverable results early and the expensive work is isolated.

**Tier 1 — six cheap datasets, ~4 core-hours total.** These may already be
committed with full m=0–12 results, in which case this is a no-op:

```bash
.venv/bin/python benchmark.py faithful diabetes iris crabs thyroid banknote --jobs 8 2>&1 | tee run-cheap.log
```

**Tier 2 — the expensive four, ~260 core-hours.** Run these on the big machine:

```bash
.venv/bin/python benchmark.py olive wine27 wine13 ais --jobs 16 2>&1 | tee run-heavy.log
```

Ordered cheapest-first (olive 33h, wine27 48h, wine13 69h, ais 109h) so partial
results accumulate steadily. Or just run everything:

```bash
.venv/bin/python benchmark.py --all --jobs 16 2>&1 | tee run.log
```

Set `--jobs` from **both** core count and RAM — peak memory is roughly
`--jobs × peak-Phi` (table below). On a 16-core / 64 GB box, `--jobs 16` is fine
for everything except `ais`/`wine13` at m ≥ 11, where 4–8 is safer.

The run is **resumable and incremental**:

- Every CV cell is cached in `results/_cv_cache/<dataset>.json`. Interrupt with
  Ctrl-C and rerun the same command — it picks up where it stopped.
- Datasets whose existing result already covers the requested degree grid are
  skipped automatically; extending `--degrees` later only computes the new cells.
- Cache writes are atomic, so an interrupt mid-write cannot corrupt them.

If you would rather run datasets separately (recommended if you want to
parallelize across machines, or isolate the expensive ones):

```bash
.venv/bin/python benchmark.py iris faithful diabetes crabs thyroid banknote --jobs 8
```

```bash
.venv/bin/python benchmark.py olive ais wine13 wine27 --jobs 4
```

---

## 4. Memory — the binding constraint

The feature matrix `Phi` is `n × |A_m| × 8` bytes, and `|A_m|` grows as
`C(k+m, m)`. Peak RAM is roughly **`--jobs` × peak-Phi**, so jobs and memory
trade off directly.

| dataset | p | Phi @ m=8 | Phi @ m=10 | Phi @ m=12 |
|---|---|---|---|---|
| faithful, diabetes, iris | 2–4 | <1 MB | 1 MB | 2 MB |
| thyroid, crabs | 5 | 2 MB | 5 MB | 11 MB |
| banknote | 6 | 5 MB | 13 MB | 30 MB |
| olive | 8 | 59 MB | 200 MB | 576 MB |
| ais | 11 | 122 MB | 570 MB | **2.2 GB** |
| wine13 | 13 | 290 MB | **1.6 GB** | **7.4 GB** |
| wine27 | 27 | **33 GB** | **496 GB** | **6 TB** |

The default `--max-phi-gb 4.0` skips anything above 4 GB. On a big machine you
can raise it:

```bash
.venv/bin/python benchmark.py wine13 --max-phi-gb 16 --jobs 2
```

**wine27 above m=6 is not worth attempting at any memory size.** Even m=7 needs
5.4M coefficients against n=178 data points — roughly 30,000× more parameters
than observations per component. It is not a compute limitation, it is
statistically meaningless, and the earlier run already showed wine27 collapsing
to G=1 at m=3. Leave it capped.

Anything skipped is recorded in `results/<dataset>.json` under
`"skipped_degrees"` with its basis size and memory — nothing is silently dropped.

---

## 5. Expected runtime

`--dry-run` prints a per-dataset estimate. At the default caps:

| dataset | est. core-hours | share |
|---|---|---|
| faithful, diabetes, iris, crabs, thyroid, banknote | ~5–11 combined | 2% |
| olive | 31–105 | 13% |
| wine27 (capped at m=6) | 44–150 | 18% |
| wine13 (capped at m=10) | 64–219 | 26% |
| ais | 100–342 | 41% |
| **total** | **242–827** | |

So roughly **30–103 h at 8 jobs, 15–52 h at 16, 8–26 h at 32**, assuming
near-perfect scaling (optimistic).

The range is wide because it is real. Measured cost per `n × |A_m|` unit
across six (dataset, m) points spans 12–41 µs, and the spread is driven by how
many EM iterations a cell happens to need — not by basis size, so it cannot be
predicted in advance. **Plan for the upper end.** Because every cell is cached,
overrunning is inconvenient rather than costly: stop, restart later, lose nothing.

If you want a fast partial result first:

```bash
.venv/bin/python benchmark.py --all --degrees 0,3,4,5,6,7,8 --jobs 16
```

then extend to the full grid later — the cache makes the second pass compute
only m=9–12.

---

## 6. Baselines and ablations (optional, fast)

These do not depend on the degree grid and already have committed results, so
rerun them only if you want fresh numbers:

```bash
.venv/bin/python baseline_tmix.py --all
```

```bash
.venv/bin/python ablation_gauge.py && .venv/bin/python ais_literature_check.py
```

---

## 7. Build the report

```bash
.venv/bin/python make_report.py
```

Writes `results/REPORT.md` with the comparison tables, per-degree energy
diagnostics, gauge ablation, and the AIS literature check.

---

## 8. What to send back

Commit and push, or just send these:

```bash
git add results/ run.log && git commit -m "Extended degree sweep m=0..12" && git push
```

- `results/*.json` — the per-dataset results (the actual data)
- `results/REPORT.md` — the assembled tables
- `run.log` — the console log, useful for diagnosing any failures
- `results/_cv_cache/` — optional; lets a later run extend the grid without
  recomputing anything

---

## What to look for in the results

The earlier `m ∈ {0,3,4}` run found corrections concentrated almost entirely at
degree 3 (skew), with degree 4 (kurtosis) contributing much less. Three things
worth checking in the extended results:

1. **Does CV ever select m > 4?** If the selected `m` stays at 3–4 across
   datasets, that is a genuine finding: the extra degrees add nothing and the
   original grid was sufficient.
2. **Where does the per-degree energy sit?** `results/REPORT.md` reports
   `Σ_{|α|=d} b²` per component. Energy concentrated at low degree means real
   structure; energy spread across high degrees means overfitting.
3. **Does held-out log-likelihood improve or peak and fall?** Training
   log-likelihood will rise monotonically with `m` (Section 35 predicts this —
   high-degree polynomials can spike on individual points). The held-out CV
   number is the one that matters, and where it peaks is the answer to whether
   degree 5+ is useful.

Note that BIC is not the right criterion here and will keep preferring plain
GMM; see the BIC note in `README.md`.
