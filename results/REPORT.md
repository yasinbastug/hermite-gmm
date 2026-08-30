# Hermite-GMM Tier-1 benchmark results

All features standardized. Plain GMM: G selected by BIC over G=1..10 (n_init=10). t-mix: **symmetric** Student's-t mixture (`studenttmixture`, estimated df) -- NOT a skew-t; no R was available and EMMIXskew is archived on CRAN, so per the spec this captures heavy tails only, not skewness. Hermite-GMM: (G, m, lambda) selected by 5-fold CV held-out log-likelihood over G=1..10, m in {0,3,4}, lambda in {0.1,1,10}; degrees 1-2 excluded; ||b_c||=1 gauge fixing on. m=0 = plain GMM (reduction check inside the grid).

## Main comparison table

| dataset | n | p | true G | GMM G(BIC) | GMM ARI@BIC-G | GMM ARI@true-G | t-mix G(BIC) | t-mix ARI@BIC-G | t-mix ARI@true-G | HGMM (G,m,lam) CV | HGMM ARI@CV | HGMM ARI@true-G | headline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iris | 150 | 4 | 3 | 2 | 0.568 | 0.904 | 5 | 0.702 | 0.904 | (4, 0, 0.0) | 0.811 | 0.904 | n/a (G_bic <= true G) |
| faithful | 272 | 2 | 2 | 2 | -- | -- | 2 | -- | -- | (2, 4, 10.0) | -- | -- | n/a (G_bic <= true G) |
| wine13 | 178 | 13 | 3 | 4 | 0.908 | 0.880 | 2 | 0.493 | 0.982 | (2, 4, 10.0) | 0.461 | 0.864 | **YES (cv-LL)** |
| diabetes | 145 | 3 | 3 | 4 | 0.698 | 0.532 | 3 | 0.719 | 0.719 | (2, 3, 1.0) | 0.462 | 0.489 | **YES (cv-LL)** |
| crabs | 200 | 5 | 4 | 3 | 0.646 | 0.023 | 3 | 0.678 | 0.601 | (3, 3, 10.0) | 0.626 | 0.018 | n/a (G_bic <= true G) |
| banknote | 200 | 6 | 2 | 3 | 0.842 | 0.980 | 3 | 0.778 | 0.980 | (3, 0, 0.0) | 0.842 | 0.980 | no |
| thyroid | 215 | 5 | 3 | 4 | 0.856 | 0.863 | 3 | 0.863 | 0.863 | (3, 3, 10.0) | 0.863 | 0.863 | **YES (cv-LL)** |
| ais | 202 | 11 | 2 | 3 | 0.717 | 0.884 | 3 | 0.551 | 0.884 | (2, 4, 10.0) | 0.884 | 0.884 | **YES (cv-LL)** |
| olive | 572 | 8 | 3 | 7 | 0.476 | 0.525 | 8 | 0.381 | 0.525 | (6, 4, 10.0) | 0.511 | 0.522 | no |
| wine27 | 178 | 27 | 3 | 10 | 0.314 | 0.894 | 10 | 0.332 | 1.000 | (1, 3, 10.0) | 0.000 | 0.878 | **YES (cv-LL)** |

## Density fit at true G (held-out CV log-likelihood, mean per sample) and BIC

| dataset | GMM cv-LL @BIC-G | GMM cv-LL @true-G | HGMM cv-LL @true-G | GMM BIC @BIC-G | HGMM BIC @true-G | HGMM m@true-G | monotone |
|---|---|---|---|---|---|---|---|
| iris | -2.4266 | -2.3799 | -2.3799 | 794.7 | 801.5 | (m=0, lam=0.0) | yes |
| faithful | -1.4750 | -1.4750 | -1.4308 | 832.6 | 847.0 | (m=4, lam=10.0) | yes |
| wine13 | -16.6545 | -15.0841 | -14.9122 | 5558.8 | 6315.3 | (m=4, lam=10.0) | yes |
| diabetes | -1.6903 | -1.6261 | -1.5351 | 493.3 | 589.1 | (m=3, lam=0.1) | yes |
| crabs | 0.0905 | -0.1345 | -0.0557 | -56.1 | 384.6 | (m=3, lam=10.0) | yes |
| banknote | -6.5110 | -6.6294 | -6.6289 | 2744.5 | 2968.6 | (m=3, lam=10.0) | yes |
| thyroid | -2.8361 | -2.4845 | -2.4260 | 1191.4 | 1325.1 | (m=3, lam=10.0) | yes |
| ais | -5.8497 | -5.7057 | -4.7541 | 2517.9 | 2964.0 | (m=4, lam=10.0) | yes |
| olive | -3.9986 | -4.8057 | -4.2032 | 5088.4 | 6917.3 | (m=4, lam=10.0) | yes |
| wine27 | -342172.4422 | -45.8060 | -44.5491 | 8669.9 | 21919.9 | (m=3, lam=1.0) | yes |

## Degree-block energies at true G (per component, sum of b^2 by total degree)

**faithful** (G=2, m=4, lam=10.0):

| component | deg 0 | deg 1 | deg 2 | deg 3 | deg 4 |
|---|---|---|---|---|---|
| 0 | 0.9765 | 0.0000 | 0.0000 | 0.0203 | 0.0033 |
| 1 | 0.9824 | 0.0000 | 0.0000 | 0.0111 | 0.0065 |

**wine13** (G=3, m=4, lam=10.0):

| component | deg 0 | deg 1 | deg 2 | deg 3 | deg 4 |
|---|---|---|---|---|---|
| 0 | 0.7619 | 0.0000 | 0.0000 | 0.1145 | 0.1237 |
| 1 | 0.7122 | 0.0000 | 0.0000 | 0.1371 | 0.1507 |
| 2 | 0.7211 | 0.0000 | 0.0000 | 0.1357 | 0.1432 |

**diabetes** (G=3, m=3, lam=0.1):

| component | deg 0 | deg 1 | deg 2 | deg 3 |
|---|---|---|---|---|
| 0 | 0.8954 | 0.0000 | 0.0000 | 0.1046 |
| 1 | 0.8496 | 0.0000 | 0.0000 | 0.1504 |
| 2 | 0.8294 | 0.0000 | 0.0000 | 0.1706 |

**crabs** (G=4, m=3, lam=10.0):

| component | deg 0 | deg 1 | deg 2 | deg 3 |
|---|---|---|---|---|
| 0 | 0.9538 | 0.0000 | 0.0000 | 0.0462 |
| 1 | 0.9312 | 0.0000 | 0.0000 | 0.0688 |
| 2 | 0.9716 | 0.0000 | 0.0000 | 0.0284 |
| 3 | 0.9609 | 0.0000 | 0.0000 | 0.0391 |

**banknote** (G=2, m=3, lam=10.0):

| component | deg 0 | deg 1 | deg 2 | deg 3 |
|---|---|---|---|---|
| 0 | 0.8752 | 0.0000 | 0.0000 | 0.1248 |
| 1 | 0.9487 | 0.0000 | 0.0000 | 0.0513 |

**thyroid** (G=3, m=3, lam=10.0):

| component | deg 0 | deg 1 | deg 2 | deg 3 |
|---|---|---|---|---|
| 0 | 0.9595 | 0.0000 | 0.0000 | 0.0405 |
| 1 | 0.9405 | 0.0000 | 0.0000 | 0.0595 |
| 2 | 0.9494 | 0.0000 | 0.0000 | 0.0506 |

**ais** (G=2, m=4, lam=10.0):

| component | deg 0 | deg 1 | deg 2 | deg 3 | deg 4 |
|---|---|---|---|---|---|
| 0 | 0.6710 | 0.0000 | 0.0000 | 0.1537 | 0.1753 |
| 1 | 0.6517 | 0.0000 | 0.0000 | 0.1824 | 0.1659 |

**olive** (G=3, m=4, lam=10.0):

| component | deg 0 | deg 1 | deg 2 | deg 3 | deg 4 |
|---|---|---|---|---|---|
| 0 | 0.7839 | 0.0000 | 0.0000 | 0.1120 | 0.1041 |
| 1 | 0.7005 | 0.0000 | 0.0000 | 0.1597 | 0.1398 |
| 2 | 0.5965 | 0.0000 | 0.0000 | 0.1658 | 0.2376 |

**wine27** (G=3, m=3, lam=1.0):

| component | deg 0 | deg 1 | deg 2 | deg 3 |
|---|---|---|---|---|
| 0 | 0.0260 | 0.0000 | 0.0000 | 0.9740 |
| 1 | 0.0348 | 0.0000 | 0.0000 | 0.9652 |
| 2 | 0.0349 | 0.0000 | 0.0000 | 0.9651 |


## Gauge-fixing ablation (Section 33)

| dataset | (G,m,lam) | cond(H) sphere | cond(H) naive | ratio | naive ||b|| drift | LL sphere | LL naive |
|---|---|---|---|---|---|---|---|
| diabetes | (3,3,1.0) | 1.60e+00 | 2.16e+03 | 1355x | 0.08-0.31 | -1.0437 | -1.0394 |
| faithful | (2,4,10.0) | 1.62e+00 | 2.28e+03 | 1412x | 0.12-0.20 | -1.3484 | -1.3389 |
| ais | (2,3,1.0) | 4.77e+00 | 2.26e+03 | 474x | 0.15-0.15 | -3.0235 | -2.9481 |

## AIS literature check (Lee & McLachlan 2013, G=2 on Height + %Bfat only)

Correct allocations out of 202 (best label permutation). Literature numbers from the EMMIXuskew JSS paper, our runs on the identical two variables:

| model | correct/202 | source |
|---|---|---|
| skew-t (FM-uMST) | 183 | literature |
| mixsmsn skew-t | 162 | literature |
| restricted skew-t (EMMIX-skew) | 157 | literature |
| sym t (FM-MT) | 165 | literature |
| plain GMM | 181 | this repo |
| t-mix (symmetric) | 163 | this repo |
| Hermite-GMM (m=3, lam=1.0) | 180 | this repo |
