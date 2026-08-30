"""Assemble results/*.json into the final comparison report (markdown).

Usage:  python make_report.py   -> writes results/REPORT.md and prints it
"""

import json
import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ORDER = ["iris", "faithful", "wine13", "diabetes", "crabs", "banknote",
         "thyroid", "ais", "olive", "wine27"]


def _load(name, suffix=""):
    path = os.path.join(RESULTS_DIR, f"{name}{suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(x, nd=3):
    if x is None:
        return "--"
    return f"{x:.{nd}f}"


def _headline(r):
    """Headline check: does the Hermite fit at TRUE G beat the plain GMM's
    BIC-selected model when the latter used MORE components? Reported
    separately on held-out CV log-likelihood and on BIC (BIC charges the
    full raw Hermite parameter count, so it over-penalizes -- Sec. 36)."""
    g, tg = r["gmm"], r["hermite"]["at_true_G"]
    if g["G_bic"] <= r["true_G"]:
        return "n/a (G_bic <= true G)"
    wins = []
    if tg["cv_ll"] is not None and tg["cv_ll"] > g["cv_ll_at_G_bic"]:
        wins.append("cv-LL")
    if tg["bic"] < g["bic_at_G_bic"]:
        wins.append("BIC")
    return "**YES (" + ", ".join(wins) + ")**" if wins else "no"


def main():
    lines = []
    lines.append("# Hermite-GMM Tier-1 benchmark results\n")
    lines.append("All features standardized. Plain GMM: G selected by BIC over "
                 "G=1..10 (n_init=10). t-mix: **symmetric** Student's-t mixture "
                 "(`studenttmixture`, estimated df) -- NOT a skew-t; no R was "
                 "available and EMMIXskew is archived on CRAN, so per the spec "
                 "this captures heavy tails only, not skewness. Hermite-GMM: "
                 "(G, m, lambda) selected by 5-fold CV held-out log-likelihood "
                 "over G=1..10, m in {0,3,4}, lambda in {0.1,1,10}; degrees 1-2 "
                 "excluded; ||b_c||=1 gauge fixing on. m=0 = plain GMM "
                 "(reduction check inside the grid).\n")

    lines.append("## Main comparison table\n")
    lines.append("| dataset | n | p | true G | GMM G(BIC) | GMM ARI@BIC-G | "
                 "GMM ARI@true-G | t-mix G(BIC) | t-mix ARI@BIC-G | "
                 "t-mix ARI@true-G | HGMM (G,m,lam) CV | HGMM ARI@CV | "
                 "HGMM ARI@true-G | headline |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ORDER:
        r = _load(name)
        t = _load(name, "_tmix")
        if r is None:
            continue
        g, h = r["gmm"], r["hermite"]
        sel = h["cv_selected"]
        row = [
            name, str(r["n"]), str(r["p"]), str(r["true_G"]),
            str(g["G_bic"]), _fmt(g["ari_at_G_bic"]), _fmt(g["ari_at_true_G"]),
            str(t["G_bic"]) if t else "--",
            _fmt(t["ari_at_G_bic"]) if t else "--",
            _fmt(t["ari_at_true_G"]) if t else "--",
            f"({sel['G']}, {sel['m']}, {sel['lambda']})",
            _fmt(sel["ari"]), _fmt(h["at_true_G"]["ari"]),
            _headline(r),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Density fit at true G (held-out CV log-likelihood, "
                 "mean per sample) and BIC\n")
    lines.append("| dataset | GMM cv-LL @BIC-G | GMM cv-LL @true-G | "
                 "HGMM cv-LL @true-G | GMM BIC @BIC-G | HGMM BIC @true-G | "
                 "HGMM m@true-G | monotone |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in ORDER:
        r = _load(name)
        if r is None:
            continue
        g, tg = r["gmm"], r["hermite"]["at_true_G"]
        lines.append(
            f"| {name} | {_fmt(g['cv_ll_at_G_bic'], 4)} | "
            f"{_fmt(g['cv_ll_at_true_G'], 4)} | {_fmt(tg['cv_ll'], 4)} | "
            f"{_fmt(g['bic_at_G_bic'], 1)} | {_fmt(tg['bic'], 1)} | "
            f"(m={tg['m']}, lam={tg['lambda']}) | "
            f"{'yes' if tg['monotone'] else 'NO'} |")

    lines.append("\n## Degree-block energies at true G "
                 "(per component, sum of b^2 by total degree)\n")
    for name in ORDER:
        r = _load(name)
        if r is None:
            continue
        tg = r["hermite"]["at_true_G"]
        if tg["m"] == 0:
            continue
        lines.append(f"**{name}** (G={tg['G']}, m={tg['m']}, "
                     f"lam={tg['lambda']}):\n")
        lines.append("| component | " + " | ".join(
            f"deg {d}" for d in range(len(tg["degree_energies"][0]))) + " |")
        lines.append("|---" * (len(tg["degree_energies"][0]) + 1) + "|")
        for c, row in enumerate(tg["degree_energies"]):
            lines.append(f"| {c} | " + " | ".join(
                f"{v:.4f}" for v in row) + " |")
        lines.append("")

    ga = _load("gauge_ablation")
    if ga:
        lines.append("\n## Gauge-fixing ablation (Section 33)\n")
        lines.append("| dataset | (G,m,lam) | cond(H) sphere | cond(H) naive | "
                     "ratio | naive ||b|| drift | LL sphere | LL naive |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in ga:
            lines.append(
                f"| {r['dataset']} | ({r['G']},{r['m']},{r['lambda']}) | "
                f"{r['sphere']['hessian_cond_worst']:.2e} | "
                f"{r['naive']['hessian_cond_worst']:.2e} | "
                f"{r['cond_ratio_naive_over_sphere']:.0f}x | "
                f"{np.min(r['naive']['b_norms']):.2f}-"
                f"{np.max(r['naive']['b_norms']):.2f} | "
                f"{r['sphere']['final_train_ll']:.4f} | "
                f"{r['naive']['final_train_ll']:.4f} |")

    lit = _load("ais_literature")
    if lit:
        lines.append("\n## AIS literature check (Lee & McLachlan 2013, "
                     "G=2 on Height + %Bfat only)\n")
        lines.append("Correct allocations out of 202 (best label "
                     "permutation). Literature numbers from the EMMIXuskew "
                     "JSS paper, our runs on the identical two variables:\n")
        lines.append("| model | correct/202 | source |")
        lines.append("|---|---|---|")
        for k, v in lit["literature_correct_of_202"].items():
            lines.append(f"| {k} | {v} | literature |")
        for k, v in lit.items():
            if isinstance(v, dict) and "correct_of_202" in v:
                lines.append(f"| {k} | {v['correct_of_202']} | this repo |")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS_DIR, "REPORT.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
