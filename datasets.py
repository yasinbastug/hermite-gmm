"""Tier-1 benchmark dataset loaders.

All files live in ./data (downloaded from Rdatasets CSV mirrors, the CRAN
GitHub mirrors of mclust/pgmm, and UCI). Each loader returns
``(X, y, true_G)`` with X float (n, p), y integer labels (or None for the
unlabeled Old Faithful), true_G the documented number of groups.

Provenance:
  crabs.csv        Rdatasets mirror of MASS::crabs
  banknote.txt     CRAN github mirror of mclust (mclust::banknote)
  ais.csv          Rdatasets mirror of DAAG::ais (same data as sn::ais)
  Diabetes.csv     Rdatasets mirror of heplots::Diabetes; columns
                   (glufast, glutest, instest) == mclust::diabetes's
                   (glucose, insulin, sspg) -- values verified against the
                   mclust source
  new-thyroid.data UCI "new-thyroid" (the source of mclust::thyroid)
  wine.rda         CRAN github mirror of pgmm (wine, 27 vars)
  olive.rda        CRAN github mirror of pgmm (olive)
  faithful.csv     Rdatasets mirror of R's built-in faithful
  iris, wine13     sklearn.datasets
"""

import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _codes(series):
    return pd.Categorical(series).codes.astype(int)


def load_crabs():
    df = pd.read_csv(os.path.join(DATA_DIR, "crabs.csv"))
    X = df[["FL", "RW", "CL", "CW", "BD"]].to_numpy(float)
    y = _codes(df["sp"].astype(str) + df["sex"].astype(str))  # 4 groups
    return X, y, 4


def load_crabs_sp():
    """Crabs with the 2-group (species only) labeling."""
    df = pd.read_csv(os.path.join(DATA_DIR, "crabs.csv"))
    X = df[["FL", "RW", "CL", "CW", "BD"]].to_numpy(float)
    return X, _codes(df["sp"]), 2


def load_banknote():
    df = pd.read_csv(os.path.join(DATA_DIR, "banknote.txt"),
                     sep=r"\s+")
    X = df[["Length", "Left", "Right", "Bottom", "Top", "Diagonal"]].to_numpy(float)
    return X, _codes(df["Status"]), 2


def load_ais():
    df = pd.read_csv(os.path.join(DATA_DIR, "ais.csv"))
    cols = ["rcc", "wcc", "hc", "hg", "ferr", "bmi",
            "ssf", "pcBfat", "lbm", "ht", "wt"]
    return df[cols].to_numpy(float), _codes(df["sex"]), 2


def load_diabetes():
    df = pd.read_csv(os.path.join(DATA_DIR, "Diabetes.csv"))
    # == mclust::diabetes (glucose, insulin, sspg)
    X = df[["glufast", "glutest", "instest"]].to_numpy(float)
    return X, _codes(df["group"]), 3


def load_thyroid():
    df = pd.read_csv(os.path.join(DATA_DIR, "new-thyroid.data"), header=None,
                     names=["class", "RT3U", "T4", "T3", "TSH", "DTSH"])
    X = df[["RT3U", "T4", "T3", "TSH", "DTSH"]].to_numpy(float)
    return X, df["class"].to_numpy(int), 3


def load_wine13():
    d = load_wine()
    return d.data.astype(float), d.target.astype(int), 3


def load_wine27():
    import pyreadr
    df = pyreadr.read_r(os.path.join(DATA_DIR, "wine.rda"))["wine"]
    y = df["Type"].to_numpy(int) - 1
    X = df.drop(columns=["Type"]).to_numpy(float)
    return X, y, 3


def load_olive():
    import pyreadr
    df = pyreadr.read_r(os.path.join(DATA_DIR, "olive.rda"))["olive"]
    y = df["Region"].to_numpy(int) - 1
    X = df.drop(columns=["Region", "Area"]).to_numpy(float)
    return X, y, 3


def load_olive_area():
    """Olive with the 9-area labeling."""
    import pyreadr
    df = pyreadr.read_r(os.path.join(DATA_DIR, "olive.rda"))["olive"]
    y = df["Area"].to_numpy(int) - 1
    X = df.drop(columns=["Region", "Area"]).to_numpy(float)
    return X, y, 9


def load_faithful():
    df = pd.read_csv(os.path.join(DATA_DIR, "faithful.csv"))
    return df[["eruptions", "waiting"]].to_numpy(float), None, 2


def load_iris_data():
    d = load_iris()
    return d.data.astype(float), d.target.astype(int), 3


TIER1 = {
    "iris": load_iris_data,
    "faithful": load_faithful,
    "wine13": load_wine13,
    "diabetes": load_diabetes,
    "crabs": load_crabs,
    "banknote": load_banknote,
    "thyroid": load_thyroid,
    "ais": load_ais,
    "olive": load_olive,
    "wine27": load_wine27,
}


if __name__ == "__main__":
    for name, loader in TIER1.items():
        X, y, G = loader()
        ylab = "unlabeled" if y is None else f"{len(np.unique(y))} classes"
        print(f"{name:10s} n={X.shape[0]:4d} p={X.shape[1]:2d} "
              f"true_G={G} ({ylab})")
