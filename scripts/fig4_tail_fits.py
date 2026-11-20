"""Figure 4 and appendix Figure A1: Pareto and stretched-exponential fits to the top 1% of three idea-size proxies.

Figure 4 (figures/fig4_tail_fits.png): share of items at least this large, top 1% of each data set, with both models
fitted by maximum likelihood to that tail. Each panel reports two tests of the Pareto against the stretched exponential:
  nested LR  LR = 2 (lnL_SE - lnL_Pareto). The stretched exponential becomes the Pareto as c -> 0, and c = 0 is on the
             edge of the parameter space, so LR is asymptotically a 50:50 mix of 0 and chi2(1): p = 0.5 P(chi2_1 > LR)
             (Self and Liang 1987; Malevergne, Pisarenko and Sornette 2005).
  Vuong z    the normalised log-likelihood ratio of Clauset, Shalizi and Newman (2009); positive favours the stretched
             exponential.
R^2 in the legend is computed on log share against fitted log share and is descriptive only.

Figure A1 (figures/figA1_c_by_tail_depth.png): c and its 95% profile-likelihood interval refitted at tail depths from
the top 10% to the top 0.05%. The diamond is the threshold the CSN procedure picks (minimum KS distance; computed here,
no bootstrap needed).

Outputs: results/tail_tests_top1pct.csv, results/c_by_tail_depth.csv
Run: python3 scripts/fig4_tail_fits.py
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tailfit as tf, csn, style
from datasets import available
warnings.filterwarnings("ignore"); style.apply()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG, OUT = os.path.join(ROOT, "figures"), os.path.join(ROOT, "results")
FRAC = 0.01
HEAD = [("kpss_value_2000_2019", "Patent market values", "US patents issued 2000-19", "value, 1982 $ million"),
        ("patentsview_examiner_cites_2002_2011", "Patent citations", "examiner citations, patents granted 2002-11", "citations received"),
        ("openalex_all_2000_2009", "Paper citations", "all journal articles published 2000-09", "citations received")]
DEPTHS = np.unique(np.r_[np.geomspace(0.1, 0.0005, 17), FRAC])[::-1]


def thr(x, w, frac):
    """Smallest value such that at least `frac` of all items are at or above it."""
    n_ge = np.cumsum(w[::-1])[::-1]
    return float(x[min(np.searchsorted(-n_ge, -w.sum() * frac, "left"), len(x) - 1)])


def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


def lr_tests(pl, se, wt):
    LR = max(2 * (se.ll - pl.ll), 0.0)
    log10_p = (np.log(0.5) + stats.chi2.logsf(LR, 1)) / np.log(10) if LR > 0 else np.log10(0.5)
    z, _ = tf.vuong(se, pl, wt)
    log10_pv = (stats.norm.logsf(abs(z)) + np.log(2)) / np.log(10)
    return dict(LR=LR, log10_p=log10_p, z=z, log10_pv=log10_pv)


def fmt_p(log10_p):
    return f"p = {10 ** log10_p:.3f}" if log10_p >= -3 else "p < .001"


ds = available()
res, rows, drows = [], [], []
for key, title, sub, unit in HEAD:
    _, _, x, w, disc = ds[key]
    xmin = thr(x, w, FRAC); xt, wt = tf.tail(x, w, xmin); u = xmin - 0.5 if disc else xmin
    pl, se = tf.fit_pl(xt, wt, xmin, disc), tf.fit_se(xt, wt, xmin, disc)
    a, b, c = pl.params["alpha"], se.params["b"], se.params["c"]
    xe = xt - 0.5 if disc else xt; emp = np.log(tf.ccdf(xt, wt))
    t = lr_tests(pl, se, wt)
    r = dict(key=key, title=title, sub=sub, unit=unit, disc=disc, xt=xt, wt=wt, u=u, a=a, b=b, c=c, n=int(wt.sum()),
             r2_pl=r2(emp, tf.lnS_pl(xe, u, a)), r2_se=r2(emp, tf.lnS_se(xe, u, b, c)), lr=t)
    # c across depths, and at the CSN-chosen threshold
    r["depth"] = pd.DataFrame([dict(share=f, **dict(zip(["c", "lo", "hi", "n"], tf.se_profile_ci(x, w, thr(x, w, f), disc))))
                               for f in DEPTHS])
    ch = csn.choose_xmin(x, w, disc, lo_tail=100, hi_frac=0.2)
    ck, lo, hi, nk = tf.se_profile_ci(x, w, ch["xmin"], disc)
    r["csn"] = dict(xmin=ch["xmin"], share=nk / w.sum(), c=ck, lo=lo, hi=hi, n=nk)
    res.append(r)
    rows.append(dict(dataset=title, tail="top 1%", threshold=xmin, n=r["n"], pareto_exponent=a, c=c, LR=t["LR"],
                     p_nested=10 ** t["log10_p"], vuong_z=t["z"], p_vuong=10 ** t["log10_pv"], r2_pareto=r["r2_pl"], r2_stretched=r["r2_se"],
                     csn_threshold=ch["xmin"], csn_tail_share=nk / w.sum(), csn_n=nk, c_at_csn=ck, c_at_csn_lo=lo, c_at_csn_hi=hi))
    for _, d in r["depth"].iterrows():
        drows.append(dict(dataset=title, tail_share=d["share"], n=int(d["n"]), c=d["c"], c_lo95=d["lo"], c_hi95=d["hi"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "tail_tests_top1pct.csv"), index=False)
pd.DataFrame(drows).to_csv(os.path.join(OUT, "c_by_tail_depth.csv"), index=False)

# ------------------------------------------------------------------ Figure 4
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
for ax, r in zip(axes, res):
    xt, u, disc = r["xt"], r["u"], r["disc"]
    emp = tf.ccdf(xt, r["wt"]) * FRAC
    ax.loglog(xt, emp, ".", ms=3, color=style.DATA, alpha=0.6, rasterized=True, label="data")
    xs = np.geomspace(xt[0], xt[-1], 300); xe = xs - 0.5 if disc else xs
    ax.loglog(xs, FRAC * np.exp(tf.lnS_pl(xe, u, r["a"])), "--", color=style.PL, label=f"Pareto, R² = {r['r2_pl']:.3f}")
    ax.loglog(xs, FRAC * np.exp(tf.lnS_se(xe, u, r["b"], r["c"])), color=style.SE,
              label=f"stretched exp. (c = {r['c']:.2f}), R² = {r['r2_se']:.3f}")
    ax.set_ylim(emp.min() / 3, FRAC * 1.5)
    ax.set_title(f"{r['title']}\n", fontsize=10.5)
    ax.text(0, 1.02, f"{r['sub']}  ·  top 1%, n = {r['n']:,}", transform=ax.transAxes, fontsize=8, color=style.INK2)
    ax.set_xlabel(r["unit"]); ax.legend(loc="lower left")
    t = r["lr"]
    ax.text(0.97, 0.97, "likelihood-ratio test,\nstretched exp. vs Pareto\n"
            f"LR = {t['LR']:,.1f},  {fmt_p(t['log10_p'])}\nVuong z = {t['z']:+.2f},  {fmt_p(t['log10_pv'])}",
            transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=8.5, color=style.INK, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.4", fc=style.SURFACE, ec=style.GRID))
axes[0].set_ylabel("share of items at least this large")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_tail_fits.png")); plt.close(fig)

# ------------------------------------------------------------------ Figure A1
c_top = max(np.nanmax(np.r_[r["depth"]["hi"], r["csn"]["hi"]]) for r in res) * 1.08
fig, AX = plt.subplots(1, 3, figsize=(14.5, 4.4))
for ax, r in zip(AX, res):
    d, k = r["depth"], r["csn"]
    ax.set_title(f"{r['title']}\n", fontsize=10.5); ax.text(0, 1.02, r["sub"], transform=ax.transAxes, fontsize=8, color=style.INK2)
    ax.fill_between(d["share"], d["lo"], d["hi"].fillna(c_top), color=style.SE, alpha=0.18, lw=0, label="95% interval")
    ax.plot(d["share"], d["c"], "o-", color=style.SE, ms=3.5, lw=1.6, label="c, fitted to the top x%")
    ax.axhline(0, color=style.PL, ls="--", lw=1.4, label="c = 0 (Pareto)")
    ax.axvline(FRAC, color=style.INK2, ls=":", lw=1)
    ax.text(FRAC, c_top * 0.97, " top 1%\n (Figure 4)", fontsize=8, color=style.INK2, va="top")
    hi = k["hi"] if np.isfinite(k["hi"]) else c_top
    ax.errorbar([k["share"]], [k["c"]], yerr=[[k["c"] - k["lo"]], [hi - k["c"]]], fmt="D", ms=6, color=style.INK, mfc="white",
                mew=1.4, capsize=3, lw=1.2, zorder=4, label="threshold chosen by the CSN procedure")
    low = k["c"] < 0.05
    ax.annotate(f"CSN threshold\ntop {k['share']:.2%}, n = {k['n']:,}", (k["share"], k["c"]), xytext=(0, 26) if low else (-8, 0),
                textcoords="offset points", ha="center" if low else "right", va="bottom" if low else "center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc=style.SURFACE, ec="none", alpha=0.85))
    ax.set_xscale("log"); ax.set_xlim(0.13, 0.0004); ax.set_ylim(-0.03, c_top)
    ticks = [0.1, 0.03, 0.01, 0.003, 0.001]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([f"top {v:.1%}".replace(".0%", "%") for v in ticks]))
    ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlabel("tail used for the fit  (deeper →)")
AX[0].set_ylabel("stretched-exp. exponent c")
h, l = AX[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=4, fontsize=8.5, frameon=False, handlelength=2, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(FIG, "figA1_c_by_tail_depth.png")); plt.close(fig)

for r in res:
    t, k = r["lr"], r["csn"]
    print(f"{r['title']:22s} top 1% n={r['n']:7,d}  c={r['c']:.3f}  LR={t['LR']:7.1f}  Vuong z={t['z']:+.2f}  "
          f"R2 {r['r2_pl']:.3f}/{r['r2_se']:.3f}  | CSN threshold top {k['share']:.3%} (n={k['n']:,}) c={k['c']:.3f} [{k['lo']:.3f}, {k['hi']:.3f}]")
print("wrote figures/fig4_tail_fits.png, figures/figA1_c_by_tail_depth.png")
