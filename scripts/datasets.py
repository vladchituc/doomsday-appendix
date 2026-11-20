"""The three idea-size data sets used in Figures 4 and 5: name -> (group, label, x, w, discrete).

x are the distinct values in increasing order and w their counts. Citation data are whole numbers (discrete = True)
and exclude items with no citations; patent values are continuous.
"""
import os
import numpy as np, pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "derived")


def _hist(name):
    d = pd.read_csv(os.path.join(D, name + ".csv"))
    d = d[d.x >= 1]
    return d.x.values.astype(float), d["count"].values.astype(float), True


def _npy(name):
    v = np.sort(np.load(os.path.join(D, name + ".npy")).astype(float))
    return v, np.ones_like(v), False


DATASETS = {
    "kpss_value_2000_2019": ("Patent value", "Patent market value, issued 2000-19 (KPSS)", lambda: _npy("kpss_value_2000_2019")),
    "patentsview_examiner_cites_2002_2011": ("Patent citations", "Patent citations by examiners, granted 2002-11 (PatentsView)",
                                             lambda: _hist("patentsview_examiner_cites_2002_2011")),
    "openalex_all_2000_2009": ("Paper citations", "Paper citations, all articles 2000-09 (OpenAlex)", lambda: _hist("openalex_all_2000_2009")),
}


def available():
    out = {}
    for k, (grp, label, loader) in DATASETS.items():
        x, w, disc = loader()
        out[k] = (grp, label, x, w, disc)
    return out
