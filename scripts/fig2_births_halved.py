"""Figure 2: "If the world had halved its birth rate in 1960".

A cohort-scaling counterfactual on the UN's actual world population by five-year age group, 1960-2023. Survival at every
age is kept exactly as observed; only the number of births changes.

  rho[b]   size of the cohort born in year b, relative to the cohort actually born. 1 for everyone born before 1960.
  N[t]     counterfactual population by age in year t = actual population by age x the (birth-weighted) rho of the
           cohorts in that age group.
  rho[t]   = phi(t) x (potential mothers in the counterfactual / potential mothers in reality), where potential mothers
           are weighted by the share of births by mother's age group (w: 15-19 ... 45-49), and phi(t) scales fertility.

Three scenarios are computed: "actual" (phi = 1; reproduces the data as a check), "tracks" (phi = 0.5: fertility is half
of what it actually was in every year, i.e. it is halved in 1960 and then falls with history; this is the orange line in
the figure) and "held" (fertility is cut to half its 1960 level and held there until actual fertility falls below that;
in the results table only).

Inputs (data/population/, Our World in Data extracts of UN World Population Prospects 2024):
  population-by-five-year-age-group.csv, births-and-deaths-projected-to-2100.csv, children-per-woman-un.csv
Outputs: figures/fig2_births_halved.png, results/births_halved_1960.csv
Run: python3 scripts/fig2_births_halved.py
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "data", "population")
out_png = os.path.join(ROOT, "figures", "fig2_births_halved.png")
out_csv = os.path.join(ROOT, "results", "births_halved_1960.csv")
ages = list(range(0, 100, 5)) + [100]
pop = {}
with open(f"{D}/population-by-five-year-age-group.csv", newline="") as f:
    rd = csv.DictReader(f); cols = [c for c in rd.fieldnames if c.startswith("population__sex_all__age_")]
    def lo(c): return int(c.split("age_")[1].split("__")[0].replace("plus", "").split("_")[0])
    cols.sort(key=lo)
    for r in rd:
        if r["entity"] == "World": pop[int(r["year"])] = np.array([float(r[c]) for c in cols]) / 1e9
births, tfr = {}, {}
with open(f"{D}/births-and-deaths-projected-to-2100.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["entity"] == "World" and r["births__sex_all__age_all__variant_estimates"]: births[int(r["year"])] = float(r["births__sex_all__age_all__variant_estimates"]) / 1e9
with open(f"{D}/children-per-woman-un.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["entity"] == "World" and r["fertility_rate__sex_all__age_all__variant_estimates"]: tfr[int(r["year"])] = float(r["fertility_rate__sex_all__age_all__variant_estimates"])
w = np.zeros(len(ages)); w[3:10] = [0.09, 0.27, 0.27, 0.19, 0.11, 0.05, 0.02]     # share of births by mother's age group, 15-19 ... 45-49
years = list(range(1960, 2024))
def run(phi):
    rho = {b: 1.0 for b in range(1840, 1960)}; N = {}
    for t in years:
        R = np.ones(len(ages))
        for j, a in enumerate(ages):
            bs = [b for b in range(t - a - 5, t - a) if b >= 1960]
            if not bs: continue
            allb = list(range(t - a - 5, t - a)); wt = np.array([births.get(b, births[1950]) for b in allb])
            R[j] = float(np.sum(wt * np.array([rho[b] for b in allb])) / wt.sum())
        N[t] = pop[t] * R
        rho[t] = phi(t) * float(w @ N[t]) / float(w @ pop[t])
    return N, rho
half_level = 0.5 * tfr[1960]
scen = {"actual": lambda t: 1.0,
        "held": lambda t: min(1.0, half_level / tfr[t]),
        "tracks": lambda t: 0.5}
res = {k: run(v) for k, v in scen.items()}
tot = {k: np.array([res[k][0][t].sum() for t in years]) for k in res}
b_cf = {k: np.array([births[t] * res[k][1][t] for t in years]) for k in res}
old = {k: np.array([res[k][0][t][13:].sum() / res[k][0][t].sum() for t in years]) for k in res}
work = {k: np.array([res[k][0][t][3:13].sum() / res[k][0][t].sum() for t in years]) for k in res}
print(f"1960 children per woman {tfr[1960]:.2f}; halved level {half_level:.2f}; actual falls to that level in {next(t for t in years if tfr[t] <= half_level)}")
for k, name in (("actual", "actual"), ("held", "halved and held"), ("tracks", "halved, then tracking history")):
    i = int(np.argmax(tot[k]))
    print(f"{name:32s} 1980 {tot[k][20]:.2f}  2000 {tot[k][40]:.2f}  2023 {tot[k][-1]:.2f} bn | peak {tot[k][i]:.2f} in {years[i]} | births 2023 {b_cf[k][-1]*1000:.0f}M | 65+ share 2023 {old[k][-1]*100:.1f}% | aged 15-64 share {work[k][-1]*100:.1f}% | children per woman 2023 {tfr[2023]*scen[k](2023):.2f}")
print("check, actual reproduced:", round(float(tot['actual'][-1]), 3), "vs data", round(float(pop[2023].sum()), 3))
print("cumulative births avoided 1960-2023 (bn):", {k: round(float((b_cf['actual'] - b_cf[k]).sum()), 2) for k in ("held", "tracks")})
with open(out_csv, "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["year", "actual_bn", "halved_and_held_bn", "halved_then_tracking_history_bn", "share_65plus_actual", "share_65plus_held", "share_65plus_tracking"])
    for i, t in enumerate(years): wr.writerow([t] + [round(float(x), 4) for x in (tot["actual"][i], tot["held"][i], tot["tracks"][i], old["actual"][i], old["held"][i], old["tracks"][i])])

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.8), dpi=200, gridspec_kw=dict(wspace=0.24, width_ratios=[1.25, 1]))
fig.patch.set_facecolor(SURFACE)
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS); ax.tick_params(axis="both", which="both", colors=MUTED, labelsize=9.5, length=0, pad=6)
pre = list(range(1950, 1961)); a1.plot(pre, [pop[t].sum() for t in pre], color=BLUE, lw=2.4, zorder=5)
for ax, series, scale in ((a1, tot, 1), (a2, old, 100)):
    ax.plot(years, series["actual"] * scale, color=BLUE, lw=2.4, zorder=5)
    ax.plot(years, series["tracks"] * scale, color=ORANGE, lw=2.4, zorder=3)
    for k in ("actual", "tracks"):
        v = series[k][-1] * scale
        ax.annotate(f"{v:.1f}" + ("%" if scale == 100 else ""), (2023, v), xytext=(6, 0), textcoords="offset points", ha="left", va="center", color=INK, fontsize=9.2, annotation_clip=False)
    style(ax)
tv = np.linspace(1950, 2010, 600); nv = 179.0 / (2026.87 - tv) ** 0.99; mv = nv <= 9.6
a1.plot(tv[mv], nv[mv], color=INK, lw=2, ls=(0, (4, 3)), zorder=4)
a1.annotate("infinite\nin 2026", (tv[mv][-1], 9.6), xytext=(7, -2), textcoords="offset points", ha="left", va="top", color=INK, fontsize=9.2)
a1.set_xlim(1950, 2023); a1.set_ylim(0, 9.6); a1.set_xticks([1960, 1980, 2000, 2020]); a1.set_yticks([0, 2, 4, 6, 8])
a1.set_title("World population, billions", loc="left", fontsize=10.5, color=INK2, pad=10)
a2.set_xlim(1960, 2023); a2.set_ylim(0, 27); a2.set_xticks([1960, 1980, 2000, 2020]); a2.set_yticks([0, 5, 10, 15, 20, 25]); a2.set_yticklabels(["0", "5%", "10%", "15%", "20%", "25%"])
a2.set_title("Share of people aged 65 and over", loc="left", fontsize=10.5, color=INK2, pad=10)
handles = [Line2D([0], [0], color=INK, lw=2, ls=(0, (4, 3)), label="What von Foerster predicted"),
           Line2D([0], [0], color=BLUE, lw=2.4, label="What happened"),
           Line2D([0], [0], color=ORANGE, lw=2.4, label="What would have happened if we did what von Foerster asked")]
leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.87), ncol=3, frameon=False, fontsize=9.8, handlelength=1.8, columnspacing=1.8, handletextpad=0.6)
for tx in leg.get_texts(): tx.set_color(INK)
fig.subplots_adjust(left=0.06, right=0.95, top=0.73, bottom=0.14)
fig.text(0.06, 0.955, "If the world had halved its birth rate in 1960", color=INK, fontsize=15, ha="left", va="center")
fig.text(0.06, 0.905, "Von Foerster's remedy, applied from the year his paper appeared, with death rates as they actually were.", color=INK2, fontsize=10.2, ha="left", va="center")
fig.text(0.06, 0.04, "Data: UN World Population Prospects 2024. Each generation is scaled by the births it would have had; survival at every age is as observed.", color=MUTED, fontsize=8.3, ha="left", va="center")
fig.savefig(out_png, facecolor=SURFACE)
