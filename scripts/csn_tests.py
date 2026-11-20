"""Clauset-Shalizi-Newman (2009) tests for the three idea-size data sets (optional; not needed for any figure).

For each data set, at two thresholds (the CSN-chosen one, and the top 1% held fixed):
  * goodness-of-fit p-value for the power law, from a bootstrap of the KS distance. At the CSN threshold the bootstrap
    re-chooses the threshold in every synthetic data set (CSN section 4.1); at the top 1% the threshold is held fixed.
    The power law is ruled out if p <= 0.1.
  * likelihood-ratio tests of the power law against the lognormal and the stretched exponential (Vuong z, positive
    favours the power law, as in CSN's tables) and the nested test of the power law inside the stretched exponential.

Usage: python3 scripts/csn_tests.py [reps]     (default 500 bootstrap draws; the post's numbers used 2,500 and 1,000)
Output: results/csn_tests.csv. With a different number of draws the p-values move in the second decimal place.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csn, tailfit as tf
from datasets import available
warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
_BOOT = None


def _init(b):
    global _BOOT
    _BOOT = b


def _rep(seed):
    return _BOOT.ks_replicate(seed)


def thr(x, w, frac):
    n_ge = np.cumsum(w[::-1])[::-1]
    return float(x[min(np.searchsorted(-n_ge, -w.sum() * frac, "left"), len(x) - 1)])


def lr_cols(x, w, xmin, disc):
    lr, _ = csn.lr_tests(x, w, xmin, disc)
    return dict(vuong_pl_vs_lognormal=lr["ln"]["stat"], p_lognormal=lr["ln"]["p"],
                vuong_pl_vs_stretched=lr["se"]["stat"], p_stretched=lr["se"]["p"],
                nested_LR_stretched=lr["se_nested"]["stat"], p_nested=lr["se_nested"]["p"])


if __name__ == "__main__":
    rows = []
    for di, (key, (grp, label, x, w, disc)) in enumerate(available().items()):
        t0 = time.time(); n = w.sum()
        ch = csn.choose_xmin(x, w, disc, lo_tail=100, hi_frac=0.2)
        boot = csn.Bootstrap(x, w, disc, ch["xmin"], ch["alpha"])
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1), initializer=_init, initargs=(boot,)) as ex:
            Ds = np.array(list(ex.map(_rep, [1_000_003 * (di + 1) + r for r in range(REPS)], chunksize=10)))
        rows.append(dict(dataset=label, threshold="CSN-chosen", xmin=ch["xmin"], n_tail=int(boot.n_tail), tail_share=boot.n_tail / n,
                         alpha_density=ch["alpha"] + 1, p_gof=float(np.mean(Ds >= ch["D"])), reps=REPS, **lr_cols(x, w, ch["xmin"], disc)))
        xm = thr(x, w, 0.01); nt = int(tf.tail(x, w, xm)[1].sum())
        a = csn.fit_alpha(x, w, xm, disc); d_obs = csn.ks_distance(x, w, xm, disc, a)
        args = [(2_000_003 * (di + 1) + r, nt, xm, a, disc) for r in range(REPS)]
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
            Dd = np.array(list(ex.map(csn.fixed_xmin_ks_replicate, args, chunksize=25)))
        rows.append(dict(dataset=label, threshold="top 1% (fixed)", xmin=xm, n_tail=nt, tail_share=nt / n, alpha_density=a + 1,
                         p_gof=float(np.mean(Dd >= d_obs)), reps=REPS, **lr_cols(x, w, xm, disc)))
        for r in rows[-2:]:
            print(f"{label:60s} {r['threshold']:15s} top {100*r['tail_share']:6.3f}%  n={r['n_tail']:7,d}  GOF p={r['p_gof']:.3f}  "
                  f"Vuong (PL vs SE) {r['vuong_pl_vs_stretched']:+.2f}  (PL vs LN) {r['vuong_pl_vs_lognormal']:+.2f}", flush=True)
        print(f"  [{time.time() - t0:.0f}s]")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "csn_tests.csv"), index=False)
