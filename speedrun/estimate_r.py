"""Estimate returns to research effort, r = lam / beta, on the NanoGPT speedrun, once per input series.

Writes results/r_estimates.csv (one row per input) and results/bootstrap_<input>.csv (beta and lam per resample).
Usage: python3 estimate_r.py [bootstrap resamples, default 500]
"""
import os, sys, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import chi2
import jones_fit as F

warnings.filterwarnings("ignore")
INPUTS = {"prs_per_day_30d": "pull requests opened per day", "contributors_30d": "active contributors"}
OUT = os.path.join(F.HERE, "results")


def boot(args):
    col, seed, x0 = args
    d = F.load(col)
    n = len(d["A1"])
    idx = np.random.default_rng(seed).integers(0, n, n)          # resample transitions with replacement
    x, _ = F.fit(d, idx, starts=[x0] + F.STARTS[:2])
    return x[0], x[1]


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for col, label in INPUTS.items():
        d = F.load(col)
        n = len(d["A1"])
        x, f = F.fit(d)
        b, lam, al, mu, c = F.unpack(x)
        x0, f0 = F.fit(d, fix_lam=0.0, starts=F.STARTS + [[x[0], x[2], x[3], x[4]]])          # input has no effect
        LR = 2 * (f0 - f)
        k = int(round(0.8 * n)); tr, te = np.arange(k), np.arange(k, n)                      # first 80% / last 20%
        xt, _ = F.fit(d, tr, starts=[list(x)] + F.STARTS)
        xt0, _ = F.fit(d, tr, fix_lam=0.0, starts=F.STARTS + [list(x0)])
        cv = F.loglik_terms(xt, d, te).sum() - F.loglik_terms(xt0, d, te, 0.0).sum()
        xd, _ = F.fit(d, np.arange(n - 1), starts=[list(x)] + F.STARTS)                       # without the final record
        with Pool() as pool:
            B = np.array(pool.map(boot, [(col, s, list(x)) for s in range(nb)], chunksize=5))
        r = B[:, 1] / B[:, 0]
        pd.DataFrame(B, columns=["beta", "lam"]).assign(r=r).to_csv(os.path.join(OUT, f"bootstrap_{col}.csv"), index=False)
        q = lambda v: np.percentile(v, [2.5, 97.5])
        gA, gI, rn = F.naive(d)
        rows.append(dict(
            input=col, transitions=n, years=round(d["years"], 3), A_start=d["A1"][0], A_end=d["A2"][-1],
            beta=b, beta_lo95=q(B[:, 0])[0], beta_hi95=q(B[:, 0])[1],
            lam=lam, lam_lo95=q(B[:, 1])[0], lam_hi95=q(B[:, 1])[1],
            r=lam / b, r_se=r.std(), r_lo95=q(r)[0], r_hi95=q(r)[1], r_boot_median=np.median(r),
            share_boot_r_ge_1=(r >= 1).mean(), share_boot_lam_le_0=(B[:, 1] <= 0).mean(),
            alpha=al, logL=-f, logL_lam0=-f0, LR_lam0=LR, p_lam0=chi2.sf(LR, 1),
            cv_r_first80=F.unpack(xt)[1] / F.unpack(xt)[0], cv_heldout_nats_vs_lam0=cv, cv_heldout_transitions=len(te),
            r_without_final_record=F.unpack(xd)[1] / F.unpack(xd)[0],
            naive_gA=gA, naive_gI=gI, naive_r=rn,
            nll_table=f, nll_scipy=F.scipy_nll(x, d), bootstrap_resamples=nb))
        t = rows[-1]
        print(f"\n{label} ({n} transitions over {t['years']:.2f} years, A from {t['A_start']:.2f} to {t['A_end']:.1f})")
        print(f"  beta = {b:.2f} [{t['beta_lo95']:.2f}, {t['beta_hi95']:.2f}]   lam = {lam:.2f} [{t['lam_lo95']:.2f}, {t['lam_hi95']:.2f}]   alpha = {al:.2f}")
        print(f"  r = {t['r']:.2f}   bootstrap SE {t['r_se']:.2f}   95% [{t['r_lo95']:.2f}, {t['r_hi95']:.2f}]   share of resamples with r >= 1: {t['share_boot_r_ge_1']:.3f}")
        print(f"  lam = 0 test: LR = {LR:.1f}, p = {t['p_lam0']:.2g}   held-out log-likelihood gain over lam = 0: {cv:+.2f} nats ({len(te)} transitions)")
        print(f"  naive r = {gA:.2f} / {gI:.2f} = {rn:.2f}   without the final record r = {t['r_without_final_record']:.2f}")
        print(f"  negative log-likelihood at the optimum: table {f:.3f}, scipy {t['nll_scipy']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "r_estimates.csv"), index=False)
