"""Figure 5: "The ideas production function says β should be flat. In the data it rises".

Fits a power law to all items above a cutoff, records the exponent, raises the cutoff and repeats. Each dot is the
Clauset-Shalizi-Newman estimator (Hill's estimator for continuous data, the discrete maximum-likelihood estimator for
counts) applied to every item at or above one cutoff. Cutoffs run from the top 1% of the distribution up to the point
where only 100 items remain. The band is ±1.96 standard errors (exponent / sqrt(n above the cutoff)). The dashed line is
the exponent at the first cutoff, extended flat.

Also prints, for the same top-1% tails, the maximum-likelihood Pareto and stretched-exponential fits (c with a
profile-likelihood 95% interval, and Vuong's z for stretched exponential against Pareto).

Inputs: data/derived/ (kpss_value_2000_2019.npy, patentsview_examiner_cites_2002_2011.csv, openalex_all_2000_2009.csv)
Outputs: figures/fig5_rising_beta.png, results/rising_beta_points.csv, results/rising_beta_tail_fits.csv
Run: python3 scripts/fig5_rising_beta.py
"""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, MaxNLocator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tailfit as tf, style, csn
from datasets import available
warnings.filterwarnings("ignore"); style.apply()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

HEAD = [("kpss_value_2000_2019", "Patent market values", "US patents issued 2000-19", "value, 1982 $ million", 0.01),
        ("patentsview_examiner_cites_2002_2011", "Patent citations", "examiner citations, patents granted 2002-11", "citations received", 0.01),
        ("openalex_all_2000_2009", "Paper citations", "all journal articles published 2000-09", "citations received", 0.01)]


def thr(x, w, frac):
    n_ge = np.cumsum(w[::-1])[::-1]
    return float(x[min(np.searchsorted(-n_ge, -w.sum() * frac, "left"), len(x) - 1)])


def pct(s):
    return f"{100*s:.2g}%"


def plain_log_ticks(ax, ticks):
    ax.set_xticks(ticks); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def header(fig, title, sub, y_title=0.975, y_sub=0.915):
    fig.suptitle(title, x=0.01, y=y_title, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, y_sub, sub, fontsize=9, color=style.INK2, va="top")


def fit_one(key, title, sub, unit, frac, xmin=None, ds=None):
    """Fit Pareto and stretched exponential to one tail, given as a top share (frac) or an explicit cutoff (xmin)."""
    _, _, x, w, disc = (ds or available())[key]
    if xmin is None:
        xmin = thr(x, w, frac)
    else:
        frac = float(w[x >= xmin].sum() / w.sum())
    xt, wt = tf.tail(x, w, xmin); u = xmin - 0.5 if disc else xmin
    pl, se = tf.fit_pl(xt, wt, xmin, disc), tf.fit_se(xt, wt, xmin, disc)
    z, p = tf.vuong(se, pl, wt)
    c, lo, hi, n = tf.se_profile_ci(x, w, xmin, disc)
    print(f"{key:40s} top {pct(frac)} n={n} alpha={pl.params['alpha']:.2f} c={c:.3f} [{lo:.3f}, {hi:.3f}] z(SE vs PL)={z:+.1f}")
    return dict(key=key, title=title, sub=sub, unit=unit, frac=frac, x=x, w=w, disc=disc, xmin=xmin, xt=xt, wt=wt, u=u,
                a=pl.params["alpha"], b=se.params["b"], c=se.params["c"], c_lo=lo, c_hi=hi, z=z, p=p, n=int(wt.sum()))


def fits():
    ds = available()
    return [fit_one(*h, ds=ds) for h in HEAD]


TAIL_DESC = {"kpss_value_2000_2019": "the most valuable 1%", "patentsview_examiner_cites_2002_2011": "the most-cited 1%",
             "openalex_all_2000_2009": "the most-cited 1%"}


def hill_dots(f, min_tail, grid=30):
    x, w, disc = f["x"], f["w"], f["disc"]
    n_ge = np.cumsum(w[::-1])[::-1]
    hi = x[np.searchsorted(-n_ge, -min_tail, "right") - 1]          # stop while at least min_tail items lie above
    ts = np.geomspace(f["xmin"], hi, grid)
    ts = np.unique(np.round(ts)) if disc else ts
    out = []
    for t in ts:
        a = csn.fit_alpha(x, w, t, disc)
        out.append((t, a, a / np.sqrt(w[x >= t].sum())))
    return np.array(out).T


def post4(F4, out="fig5_rising_beta.png", title="The ideas production function says β should be flat. In the data it rises"):
    fig, axes = plt.subplots(1, len(F4), figsize=(4.4 * len(F4), 4.9)); rows = []
    for ax, f in zip(axes, F4):
        ts, a, se = hill_dots(f, 100)
        rows += [dict(dataset=f["title"], cutoff=t, beta=b, se=s, n_above=int(f["w"][f["x"] >= t].sum())) for t, b, s in zip(ts, a, se)]
        ax.fill_between(ts, a - 1.96 * se, a + 1.96 * se, color=style.DATA, alpha=0.12, lw=0, label="95% interval")
        ax.plot(ts, a, "o", ms=4, color=style.DATA, label="β measured above each cutoff", zorder=3)
        ax.axhline(a[0], ls="--", color=style.PL, lw=1.8, label="constant β, as the standard\nequation assumes")
        lo, hi = a.min(), a.max(); pad = 0.12 * (hi - lo)
        ax.set_xscale("log"); ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(ts.min() / 1.04, ts.max() * 1.04)
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(f"{f['title']}\n\n", fontsize=10.5)
        ax.text(0, 1.02, f"{f['sub']}\n{TAIL_DESC[f['key']]}", transform=ax.transAxes, fontsize=8, color=style.INK2)
        ax.set_xlabel(f"cutoff: {f['unit']}"); ax.xaxis.set_minor_formatter(NullFormatter())
        if f["key"] == "kpss_value_2000_2019":
            plain_log_ticks(ax, [200, 500, 1000])
        elif f["key"].startswith("openalex"):
            plain_log_ticks(ax, [500, 1000, 2000, 5000, 10000]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        else:
            plain_log_ticks(ax, [40, 50, 70, 100, 150])
        ax.set_xlim(ts.min() / 1.04, ts.max() * 1.04)
    axes[0].set_ylabel("β: exponent of a power law\nfitted above the cutoff")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend([h[1], h[0], h[2]], [l[1], l[0], l[2]], loc="upper left", fontsize=8)
    header(fig, title,
           "The standard ideas production function treats β, how much harder each new idea is to find, as a constant. That holds only if idea sizes follow a power law, and then\n"
           "a power law fitted to the ideas above any cutoff gives the same β wherever the cutoff is. Each dot fits a power law to the items above one cutoff, the way Clauset,\n"
           "Shalizi and Newman do. Moving right raises the cutoff. Vertical scales differ between panels.",
           y_sub=0.915)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(FIG, out)); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rising_beta_points.csv"), index=False)


if __name__ == "__main__":
    F = fits()
    pd.DataFrame([dict(dataset=f["title"], sample=f["sub"], tail_share=f["frac"], cutoff=f["xmin"], n_tail=f["n"], pareto_exponent=f["a"],
                       c=f["c"], c_lo95=f["c_lo"], c_hi95=f["c_hi"], vuong_z_se_vs_pareto=f["z"], vuong_p=f["p"]) for f in F]
                 ).to_csv(os.path.join(OUT, "rising_beta_tail_fits.csv"), index=False)
    post4(F)
    print("saved fig5_rising_beta.png")
