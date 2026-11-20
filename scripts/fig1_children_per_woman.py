"""Figure 1: "What von Foerster's curve asks of mothers".

Von Foerster, Mora and Amiot (1960) fitted world population as N(t) = K / (T0 - t)^k with T0 = 2026.87, K = 179 billion and
k = 0.99. This script asks what that curve requires if, from 1960 on, nobody dies at all, so every added
person is a birth.

  births per year        B(t) = dN/dt = k K / (T0 - t)^(k+1)
  women aged 15 to 49    half of everyone born between 50 and 15 years earlier. Cohorts born before 1960 come from the UN's
                         1960 world age structure, scaled to the curve's 1960 population; cohorts born after come from the curve.
  children per woman     35 x (births per year / women aged 15 to 49), i.e. the total fertility rate if every woman bore
                         children at that rate through all 35 childbearing years.

Inputs (data/population/, Our World in Data extracts of UN World Population Prospects 2024):
  population-by-five-year-age-group.csv, children-per-woman-un.csv
Outputs: figures/fig1_children_per_woman.png, results/children_per_woman_required.csv
Run: python3 scripts/fig1_children_per_woman.py
"""
import csv, os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "data", "population")
out_png = os.path.join(ROOT, "figures", "fig1_children_per_woman.png")
out_csv = os.path.join(ROOT, "results", "children_per_woman_required.csv")
T0, K, k = 2026.87, 179.0, 0.99
N = lambda t: K / (T0 - t) ** k                 # billions
B = lambda t: k * K / (T0 - t) ** (k + 1)       # births per year if nobody dies (billions)
# 1960 world age structure (UN), five-year groups -> cumulative share older than a given age
with open(f"{D}/population-by-five-year-age-group.csv", newline="") as f:
    rd = csv.DictReader(f); cols = [c for c in rd.fieldnames if c.startswith("population__sex_all__age_")]
    row = next(r for r in rd if r["entity"] == "World" and r["year"] == "1960")
groups = []
for c in cols:
    part = c.split("age_")[1].split("__")[0]
    lo = int(part.replace("plus", "").split("_")[0]); hi = lo + 5 if "plus" not in part else lo + 15
    if "_" in part and "plus" not in part: hi = int(part.split("_")[1]) + 1
    groups.append((lo, hi, float(row[c])))
groups.sort(); tot = sum(g[2] for g in groups)
edges = [groups[0][0]] + [g[1] for g in groups]; cum = np.concatenate([[0], np.cumsum([g[2] for g in groups])]) / tot
share_younger = lambda a: float(np.interp(a, edges, cum))     # share of the 1960 population younger than age a
N60 = N(1960.0)
def C(s):
    """People born up to time s who are alive (nobody dies after 1960), billions."""
    s = np.asarray(s, float)
    return np.where(s >= 1960, N(np.minimum(s, T0 - 1e-6)), N60 * (1 - np.vectorize(share_younger)(np.maximum(1960 - s, 0))))
women = lambda t: 0.5 * (C(t - 15) - C(t - 50))               # women aged 15-49, billions
t = np.concatenate([np.arange(1960, 2026, 0.05), np.arange(2026, 2026.8, 0.005)])
W = women(t); births = B(t); per_woman_year = births / W; tfr = 35 * per_woman_year
def cross(level, arr):
    i = int(np.argmax(arr >= level)); return float(t[i])
def load_tfr():
    out = {}
    with open(f"{D}/children-per-woman-un.csv", newline="") as f:
        for r in csv.DictReader(f):
            v = r["fertility_rate__sex_all__age_all__variant_estimates"]
            if v: out[(r["entity"], int(r["year"]))] = (float(v), r["code"])
    return out
tf = load_tfr(); world = {y: v for (e, y), (v, c) in tf.items() if e == "World"}
rec = max(((v, e, y) for (e, y), (v, c) in tf.items() if len(c) == 3), key=lambda x: x[0])
print(f"highest national fertility on record: {rec[0]:.2f} children per woman, {rec[1]} {rec[2]}")
print(f"women 15-49 share of population on the curve: 1960 {women(1960.0)/N(1960.0)*100:.1f}%  2000 {float(women(2000.0))/N(2000.0)*100:.1f}%  2020 {float(women(2020.0))/N(2020.0)*100:.1f}%")
for yr in (1960, 1970, 1980, 1990, 2000, 2005, 2010, 2015, 2020, 2023, 2025, 2026, 2026.5, 2026.77):
    w = float(women(yr)); b = B(yr)
    print(f"  {yr}: population {N(yr):8.2f} bn  births/yr {b:9.3f} bn  women 15-49 {w:6.3f} bn  births per woman per year {b/w:8.3f}  children per woman {35*b/w:9.1f}  growth {k/(T0-yr)*100:6.1f}%")
print(f"crossings: children per woman passes {rec[0]:.1f} in {cross(rec[0], tfr):.1f}; passes 12 in {cross(12, tfr):.1f}; a birth per woman per year (35) in {cross(35, tfr):.1f}; one per 9 months in {cross(35/0.75, tfr):.1f}")
print(f"growth-rate crossings on the curve: 3.9% in {T0 - k/0.039:.1f}; 6% in {T0 - k/0.06:.1f}")
with open(out_csv, "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["year", "population_bn", "births_per_year_bn", "women_15_49_bn", "children_per_woman_required"])
    for yr in list(range(1960, 2026)) + [2026, 2026.5]:
        wr.writerow([yr, round(N(yr), 3), round(B(yr), 4), round(float(women(yr)), 4), round(35 * B(yr) / float(women(yr)), 2)])

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 6.0), dpi=200, gridspec_kw=dict(wspace=0.22))
fig.patch.set_facecolor(SURFACE)
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS); ax.tick_params(axis="both", which="both", colors=MUTED, labelsize=9.5, length=0, pad=6)
m = t <= 2026.6
a1.plot(t[m], births[m], color=BLUE, lw=2.4, zorder=4); a1.plot(t[m], W[m], color=ORANGE, lw=2.4, zorder=3)
a1.set_yscale("log"); a1.set_xlim(1960, 2027); a1.set_ylim(0.03, 2000)
a1.set_yticks([0.1, 1, 10, 100, 1000]); a1.set_yticklabels(["0.1", "1", "10", "100", "1,000"]); a1.minorticks_off()
a1.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2020]); style(a1)
a1.set_title("Billions (log scale)", loc="left", fontsize=10.5, color=INK2, pad=10)
xc = cross(35, tfr); yc = float(B(xc))
a1.scatter([xc], [yc], s=36, facecolor=SURFACE, edgecolor=INK, linewidth=1.4, zorder=6)
a1.annotate(f"{xc:.0f}: more births a year than\nwomen of childbearing age", (xc, yc), xytext=(-10, 22), textcoords="offset points", ha="right", va="bottom", color=INK, fontsize=8.8,
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8, shrinkB=4))
a1.annotate("births needed\neach year", (1990, float(B(1990))), xytext=(0, -12), textcoords="offset points", ha="center", va="top", color=INK, fontsize=8.8)
a1.annotate("women aged 15 to 49", (1985, float(women(1985.0))), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=8.8)
a2.plot(t[m], tfr[m], color=BLUE, lw=2.4, zorder=4)
wy = sorted(world); a2.plot(wy, [world[y] for y in wy], color=INK, lw=2.2, zorder=5)
a2.set_yscale("log"); a2.set_xlim(1960, 2027); a2.set_ylim(1, 20000)
a2.set_yticks([1, 2, 5, 10, 35, 100, 1000, 10000]); a2.set_yticklabels(["1", "2", "5", "10", "35", "100", "1,000", "10,000"]); a2.minorticks_off()
a2.set_xticks([1960, 1970, 1980, 1990, 2000, 2010, 2020]); style(a2)
a2.set_title("Children per woman (log scale)", loc="left", fontsize=10.5, color=INK2, pad=10)
for level, text in ((rec[0], f"highest national average on record: {rec[0]:.1f} ({rec[1]}, {rec[2]})"), (35.0, "a baby every year from age 15 to 49")):
    a2.axhline(level, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=2)
    a2.annotate(text, (1961, level), xytext=(0, 4), textcoords="offset points", ha="left", va="bottom", color=INK2, fontsize=8.6)
    xl = cross(level, tfr); a2.scatter([xl], [level], s=36, facecolor=SURFACE, edgecolor=INK, linewidth=1.4, zorder=6)
    a2.annotate(f"{xl:.0f}", (xl, level), xytext=(7, -10), textcoords="offset points", ha="left", va="top", color=INK, fontsize=8.8)
a2.annotate("what the curve requires\nif nobody dies", (2012, float(35 * B(2012) / women(2012.0))), xytext=(-12, 26), textcoords="offset points", ha="right", va="bottom", color=INK, fontsize=8.8)
a2.annotate("what happened", (1992, world[1992]), xytext=(0, -9), textcoords="offset points", ha="center", va="top", color=INK, fontsize=8.8)
handles = [Line2D([0], [0], color=BLUE, lw=2.4, label="Required by von Foerster's curve, with no deaths at all"),
           Line2D([0], [0], color=ORANGE, lw=2.4, label="Women aged 15 to 49 on that curve"),
           Line2D([0], [0], color=INK, lw=2.2, label="Actual world average")]
leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.875), ncol=3, frameon=False, fontsize=9.8, handlelength=1.8, columnspacing=1.8, handletextpad=0.6)
for tx in leg.get_texts(): tx.set_color(INK)
fig.subplots_adjust(left=0.065, right=0.965, top=0.74, bottom=0.12)
fig.text(0.065, 0.955, "What von Foerster's curve asks of mothers", color=INK, fontsize=15, ha="left", va="center")
fig.text(0.065, 0.905, "From 1960 nobody dies, so every added person is a birth. Left: births needed against the women available. Right: births per woman.", color=INK2, fontsize=10.2, ha="left", va="center")
fig.text(0.065, 0.03, "Starts from the UN's 1960 world age structure, scaled to the curve; half of each birth cohort is female; children per woman = 35 x births per woman aged 15 to 49. Fertility data: UN WPP 2024.",
         color=MUTED, fontsize=8.3, ha="left", va="center")
fig.savefig(out_png, facecolor=SURFACE)
