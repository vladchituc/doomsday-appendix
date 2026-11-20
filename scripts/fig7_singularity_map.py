"""Figure 7: "Software singularity with one assumption changed".

Davidson, Halperin, Houlden and Korinek's (2026) headline scenario (software research 100% automated, 5% everywhere
else; davidson_model.py) with the software ideas production function generalised by one parameter c:

  paper (power law):  ds/dt = g0 * exp(L_S - s / r)
  curved Jones:       ds/dt = g0 * exp(L_S - (1/r) * (e^(c s) - 1) / c)

s is the log software level relative to today and L_S the log research input, so the local exponent is
beta(s) = (1/r) e^(c s): today's beta and today's pace are the paper's, and c = 0 is the paper's law exactly. Nothing else
in the model changes.

Grid: r from 0.50 to 1.50 in steps of 0.01 (101 values) x c from 0.01 to 0.1 on a log scale (48 values), plus the power
law at every r: 4,949 runs of 40 years. Colour: the most software progress made in any 12 months, in years of progress at
today's pace (g0 = 1.10 log points a year). A power-law run is a finite-time singularity if software's pace reaches 10^5
times today's before year 40; the hatched strip is drawn from the analytic threshold r > 0.673 (the paper's explosion
condition f_Y + f_S r_S + f_H r_H + f_A r_A / (1 - alpha) > 1 in this scenario). Runs with c > 0 are integrated in a
stretched clock, d(sigma) = (1 + ds/dt) dt, so a burst of any size can be passed through and measured.

Outputs: results/singularity_sweep.npz (cached; delete it to rerun), results/singularity_sweep.csv,
         figures/fig7_singularity_map.png
Run: python3 scripts/fig7_singularity_map.py      (about a minute on 8 cores)
"""
import os, sys, math, time, csv
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# Davidson et al. (2026), section 5.3 calibration (same as davidson_model.py); software returns r is swept
alpha = 0.4
rH, rA = 5.0, 0.32; bH, bA = 1 / rH, 1 / rA
g0S, g0H = 1.10, 0.37; n = dict(S=1.10, H=0.074, A=0.04, Y=0.01); g0A = rA * n["A"]
dY, dH = 0.07, 0.20
gY0 = g0A / (1 - alpha) + n["Y"]; gKH0 = g0H + gY0
f = dict(S=1.0, H=.05, A=.05, Y=.05)
HORIZON, CAP = 40.0, 1e5
R_SING = 1 - f["Y"] - f["H"] * rH - f["A"] * rA / (1 - alpha)     # 0.673: power law singular above this r
CS = np.geomspace(0.01, 0.1, 48); RS = np.round(np.linspace(0.5, 1.5, 101), 4)


def run(job):
    law, r, c = job

    def parts(t, u):
        s, h, a, kY, kH = u
        L = {i: (1 - f[i]) * n[i] * t + f[i] * (kH + s) for i in "SHAY"}
        if law == "power":
            ds = g0S * math.exp(L["S"] - s / r)
        else:
            ds = g0S * math.exp(min(L["S"] - (math.expm1(min(c * s, 700.0)) / c) / r, 700.0))
        dh = g0H * math.exp(L["H"] - bH * h)
        da = g0A * math.exp(L["A"] - bA * a)
        y = a + (1 - alpha) * L["Y"] + alpha * kY
        dkY = (gY0 + dY) * math.exp(y - kY) - dY
        dkH = (gKH0 + dH) * math.exp(h + y - kH) - dH
        return [ds, dh, da, dkY, dkH], ds

    if law == "power":
        ev = lambda t, u: parts(t, u)[1] - CAP * g0S
        ev.terminal = True; ev.direction = 1
        sol = solve_ivp(lambda t, u: parts(t, u)[0], (0, HORIZON), [0] * 5, method="LSODA", dense_output=True,
                        rtol=1e-8, atol=1e-10, events=ev, max_step=0.05)
        if sol.t[-1] < HORIZON - 1e-6: return (law, r, c, float("inf"), sol.t[-1])
        ts = np.linspace(0, HORIZON, 4001); s = sol.sol(ts)[0]
    else:
        def rhs(sig, w):
            d, _ = parts(w[5], w[:5]); k = 1.0 / (1.0 + d[0])
            return [x * k for x in d] + [k]
        ev = lambda sig, w: w[5] - HORIZON
        ev.terminal = True; ev.direction = 1
        sol = solve_ivp(rhs, (0, 1e5), [0] * 6, method="LSODA", dense_output=True, rtol=1e-8, atol=1e-10, events=ev, max_step=2.0)
        sg = np.unique(np.concatenate([sol.t, np.linspace(0, sol.t[-1], 40001)])); W = sol.sol(sg)
        tt = np.maximum.accumulate(W[5]); keep = np.concatenate([[True], np.diff(tt) > 0])
        ts = np.linspace(0, HORIZON, 4001); s = np.interp(ts, tt[keep], W[0][keep])
    gain = s[100:] - s[:-100]; i = int(gain.argmax())          # 100 steps of 0.01 years = 12 months
    return (law, r, c, float(gain[i] / g0S), float(ts[i]))


def sweep():
    path = os.path.join(ROOT, "results", "singularity_sweep.npz")
    if os.path.exists(path): return np.load(path)
    jobs = [("jones", float(r), float(c)) for r in RS for c in CS] + [("power", float(r), 0.0) for r in RS]
    t0 = time.time()
    with Pool() as p: res = p.map(run, jobs, chunksize=8)
    G = np.full((len(RS), len(CS)), np.nan); p0 = np.full(len(RS), np.nan)
    for law, r, c, v, _ in res:
        i = int(np.argmin(abs(RS - r)))
        if law == "power": p0[i] = v
        else: G[i, int(np.argmin(abs(CS - c)))] = v
    np.savez(path, cs=CS, rs=RS, grid=G, power=p0)
    with open(path.replace(".npz", ".csv"), "w", newline="") as fo:
        wr = csv.writer(fo); wr.writerow(["law", "returns_r", "curvature_c", "years_of_todays_progress_in_fastest_year"])
        for i, r in enumerate(RS): wr.writerow(["power law", r, 0, "singularity" if np.isinf(p0[i]) else round(float(p0[i]), 3)])
        for i, r in enumerate(RS):
            for j, c in enumerate(CS): wr.writerow(["curved Jones", r, round(float(c), 5), round(float(G[i, j]), 3)])
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s")
    return np.load(path)


# ------------------------------------------------------------------ figure
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]; plt.rcParams["hatch.linewidth"] = 1.1
FS = dict(title=19.5, subtitle=14, tick=12.5, label=13, note=11.5, small=11, legend=12)


def edges(v):
    l = np.log(v); st = l[1] - l[0]
    return np.exp(np.concatenate([[l[0] - st / 2], (l[1:] + l[:-1]) / 2, [l[-1] + st / 2]]))


def plot(d, out):
    cs, rs, G, p0 = d["cs"], d["rs"], d["grid"], d["power"]
    vals = np.concatenate([G.ravel(), p0[np.isfinite(p0)]]); vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    cticks = [v for v in (0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100) if vmin <= v <= vmax]
    norm = LogNorm(vmin=vmin, vmax=vmax); cmap = plt.get_cmap("viridis")
    ce = edges(cs); dr = rs[1] - rs[0]; re = np.concatenate([[rs[0] - dr / 2], (rs[1:] + rs[:-1]) / 2, [rs[-1] + dr / 2]])
    fig = plt.figure(figsize=(9.2, 8.8), dpi=200); fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.055, 1, 0.07, 0.045], wspace=0.055, left=0.135, right=0.865, top=0.83, bottom=0.17)
    a0 = fig.add_subplot(gs[0, 0]); ax = fig.add_subplot(gs[0, 1]); cax = fig.add_subplot(gs[0, 3])
    mesh = ax.pcolormesh(ce, re, G, norm=norm, cmap=cmap, shading="flat", rasterized=True)
    ax.set_xscale("log"); ax.set_xlim(ce[0], ce[-1]); ax.set_ylim(re[0], re[-1])
    xt = [0.01, 0.02, 0.03, 0.05, 0.1]
    ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}" for v in xt]); ax.minorticks_off(); ax.set_yticks([]); ax.set_yticks([], minor=True)
    a0.pcolormesh([0, 1], re, np.where(rs[:, None] < R_SING, p0[:, None], np.nan), norm=norm, cmap=cmap, shading="flat", rasterized=True)
    y0 = re[int(np.argmax(rs >= R_SING))]
    a0.add_patch(Rectangle((0, y0), 1, re[-1] - y0, facecolor="#f0efec", edgecolor=INK, hatch="////", linewidth=0))
    a0.set_xlim(0, 1); a0.set_ylim(re[0], re[-1]); a0.set_xticks([0.5]); a0.set_xticklabels(["0"])
    yt = [0.5, 0.75, 1.0, 1.25, 1.5]; a0.set_yticks(yt); a0.set_yticklabels([f"{v:g}" for v in yt]); a0.set_yticks([], minor=True)
    a0.scatter([0.5], [1.0], s=52, color=INK, zorder=5, edgecolor=SURFACE, linewidth=0.8)
    a0.annotate("Davidson's\nown rule", (0.5, 0), xycoords="axes fraction", xytext=(0, -27), textcoords="offset points",
                ha="center", va="top", color=INK2, fontsize=FS["small"], linespacing=1.2, annotation_clip=False)
    for a in (a0, ax):
        a.set_facecolor(SURFACE)
        for sp in a.spines.values(): sp.set_visible(False)
        a.tick_params(axis="both", which="both", colors=INK2, labelsize=FS["tick"], length=0, pad=6)
    ax.axhline(0.83, color="white", lw=1.0, alpha=0.9, zorder=5)
    ax.annotate("chess-engine estimate, 0.83", (ce[-1], 0.83), xytext=(-7, 4), textcoords="offset points", ha="right", va="bottom",
                color="white", fontsize=FS["note"], zorder=6)
    levels = [v for v in (2, 3, 10, 30) if vmin < v < vmax]
    lines = ax.contour(cs, rs, G, levels=levels, colors="white", linewidths=1.3, zorder=4)
    ax.clabel(lines, fmt={v: f"{v:g}" for v in levels}, fontsize=FS["note"], colors="white", inline=True, inline_spacing=6)
    ax.set_xlabel("curvature of the ideas production function, c (log scale)\nfurther right: the gains from research fade faster",
                  color=INK2, fontsize=FS["label"], labelpad=8, linespacing=1.45)
    a0.set_ylabel("returns to software research today, r\nhigher: each doubling of research buys more", color=INK2,
                  fontsize=FS["label"], labelpad=5, linespacing=1.45)
    cb = fig.colorbar(mesh, cax=cax); cb.outline.set_visible(False)
    cb.set_ticks(cticks); cb.set_ticklabels([f"{v:g}" for v in cticks]); cb.ax.minorticks_off()
    cb.ax.tick_params(colors=INK2, labelsize=FS["tick"], length=3, width=1)
    cb.set_label("years of today's progress in the fastest year", color=INK2, fontsize=FS["label"], labelpad=9)
    handles = [Patch(facecolor="#f0efec", edgecolor=INK, hatch="////", label="finite-time singularity (only where c = 0)"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, markeredgecolor=INK, markersize=8, label="Davidson's model")]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2, frameon=False,
                     fontsize=FS["legend"], columnspacing=2.2, handletextpad=0.7)
    for tx in leg.get_texts(): tx.set_color(INK)
    fig.suptitle("Software singularity with one assumption changed", x=0.045, y=0.972, ha="left", fontsize=FS["title"], color=INK)
    fig.text(0.045, 0.925, "Davidson's model assumes ideas get harder to find at a constant rate. Below is\nDavidson's model with one "
             "additional parameter, c: the rate rises as software improves,\nand c = 0 is Davidson's model exactly.",
             ha="left", va="top", fontsize=FS["subtitle"], color=INK2, linespacing=1.4)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    d = sweep()
    G, p0, rs, cs = d["grid"], d["power"], d["rs"], d["cs"]
    print(f"analytic threshold for the power law: r > {R_SING:.3f}; singular within 40 years for r >= {rs[np.isinf(p0)].min():.2f}")
    print(f"runs with c > 0 that are singular: {int(np.isinf(G).sum())} of {G.size}")
    print("fastest year, in years of today's progress")
    pick = (0.01, 0.02, 0.03, 0.05, 0.1)
    print("   r     " + "  ".join(f"c={c:<6g}" for c in pick))
    for r in (0.83, 1.0):
        i = int(np.argmin(abs(rs - r)))
        print(f"  {rs[i]:.2f}  " + "  ".join(f"{np.interp(np.log(c), np.log(cs), G[i]):8.2f}" for c in pick))
    i = int(np.argmin(abs(rs - 1.0))); j = int(np.argmax(G[i] < 3))
    print(f"at r = 1 the fastest year falls below 3 years of progress between c = {cs[j-1]:.4f} and {cs[j]:.4f}")
    plot(d, os.path.join(ROOT, "figures", "fig7_singularity_map.png"))
    print("wrote figures/fig7_singularity_map.png")
