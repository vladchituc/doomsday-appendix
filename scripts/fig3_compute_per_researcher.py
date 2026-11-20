"""Figure 3: "What the growth model asks of each AI researcher".

Runs the headline scenario of Davidson et al. (2026) (software research 100% automated, 5% elsewhere; see
davidson_model.py) up to the date world output growth passes 200% a year, the paper's threshold for a singularity.

Left panel:   number of AI researchers = N0 x exp(s + kH), where s is the log gain in software efficiency and kH the log
              gain in the stock of compute. "More chips alone" is N0 x exp(kH).
Right panel:  operations per second each researcher runs on = C0 / exp(s).

Two assumptions set the units and nothing else: N0 = 20,000 AI researchers today, and C0 = 1e16 operations per second
per researcher (an estimate of the human brain within Carlsmith's range). Changing C0 moves every point on the right panel
by the same factor. Reference machines are round, approximate figures: Xbox 360 (2005, about 2.4e11 FLOPS), a Pentium
PC running Windows 95 (about 1e8), the Manchester Baby (1948, about 1.1e3 instructions a second), one calculation a
second, and Turing running his chess program by hand (about 1e-3).

Outputs: figures/fig3_compute_per_researcher.png, results/compute_per_researcher.csv
Run: python3 scripts/fig3_compute_per_researcher.py
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from davidson_model import simulate, C0
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "fig3_compute_per_researcher.png")
sol, parts = simulate(dict(S=1, H=.05, A=.05, Y=.05), t_end=40)
ts = np.linspace(0, sol.t[-1], 400001); U = sol.sol(ts); gY = np.array([parts(t, U[:, i])[1] for i, t in enumerate(ts)])
i200 = int(np.argmax(gY >= 2.0)); T = ts[i200]; m = ts <= T
t, s, kH = ts[m], U[0][m], U[4][m]
N0 = 2e4
researchers = N0 * np.exp(s + kH); chips_only = N0 * np.exp(kH); per = C0 / np.exp(s)
miles = [(lev, int(np.argmax(gY >= lev))) for lev in (0.3, 0.5, 1.0)]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, WASH = "#2a78d6", "#eb6834", "#cde2fb"
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.2), dpi=200, gridspec_kw=dict(wspace=0.3))
fig.patch.set_facecolor(SURFACE)
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS); ax.tick_params(axis="both", which="both", colors=MUTED, labelsize=9.5, length=0, pad=6)
sup = lambda e: "10" + str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
XMAX = 6.0
# left
a1.fill_between(t, chips_only, researchers, color=WASH, alpha=0.7, linewidth=0, zorder=1)
a1.plot(t, researchers, color=BLUE, lw=2.4, zorder=4); a1.plot(t, chips_only, color=ORANGE, lw=2.4, zorder=3)
a1.set_yscale("log"); a1.set_xlim(0, XMAX); a1.set_ylim(8e3, 3e26)
a1.set_yticks([1e4, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24]); a1.set_yticklabels(["10,000", "1 million", "1 billion", "1 trillion", sup(15), sup(18), sup(21), sup(24)]); a1.minorticks_off()
a1.set_xticks([0, 1, 2, 3, 4, 5]); style(a1)
a1.set_title("AI researchers", loc="left", fontsize=10.5, color=INK2, pad=10)
a1.axhline(8.2e9, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=2)
a1.annotate("one per person on Earth", (0.08, 8.2e9), xytext=(0, 3), textcoords="offset points", ha="left", va="bottom", color=INK2, fontsize=8.8)
a1.annotate("20,000 today", (0, N0), xytext=(8, 16), textcoords="offset points", ha="left", va="bottom", color=INK, fontsize=8.8)
a1.annotate("10²⁵", (T, researchers[-1]), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", color=INK, fontsize=9.2)
a1.annotate("1 million", (T, chips_only[-1]), xytext=(-6, 8), textcoords="offset points", ha="right", va="bottom", color=INK, fontsize=9.2)
a1.annotate("algorithmic\nprogress", (5.36, 2e11), ha="right", va="center", color=INK, fontsize=9.2)
a1.axvline(T, color=INK, lw=1, ls=(0, (1, 2)), zorder=2)
a1.annotate("singularity", (T, 3e26), xytext=(5, -2), textcoords="offset points", ha="left", va="top", color=INK, fontsize=8.8, annotation_clip=False)
# right
a2.plot(t, per, color=BLUE, lw=2.4, zorder=4); a2.plot(t, C0 * np.exp(-1.1 * t), color=INK, lw=2.2, zorder=5)
a2.set_yscale("log"); a2.set_xlim(0, XMAX); a2.set_ylim(3e-6, 1e18)
e2 = [-3, 0, 3, 6, 9, 12, 15]; a2.set_yticks([10.0 ** e for e in e2]); a2.set_yticklabels([sup(e) if e else "1" for e in e2]); a2.minorticks_off()
a2.set_xticks([0, 1, 2, 3, 4, 5]); style(a2)
a2.set_title("Operations a second each one runs on", loc="left", fontsize=10.5, color=INK2, pad=10)
for level, text, below in ((1e16, "Human brain (10¹⁶, within Carlsmith's range)", False), (2.4e11, "Xbox 360 (2005)", False), (1e8, "Pentium PC running Windows 95", False),
                           (1.1e3, "Manchester Baby (1948)", False), (1.0, "1 calculation a second", False), (1e-3, "Turing running his chess program by hand", True)):
    a2.axhline(level, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=2)
    a2.annotate(text, (0.08, level), xytext=(0, -4 if below else 3), textcoords="offset points", ha="left", va="top" if below else "bottom", color=INK2, fontsize=8.8)
for lev, i in miles:
    a2.scatter([ts[i]], [C0 / math.exp(U[0, i])], s=36, facecolor=SURFACE, edgecolor=INK, linewidth=1.4, zorder=6)
    a2.annotate(f"{lev*100:.0f}% growth" if lev == 0.3 else f"{lev*100:.0f}%", (ts[i], C0 / math.exp(U[0, i])), xytext=(-8, -3), textcoords="offset points", ha="right", va="top", color=INK, fontsize=8.8)
a2.scatter([T], [per[-1]], s=36, color=INK, zorder=6)
a2.annotate("200%: the model's\nsingularity threshold", (T, per[-1]), xytext=(-9, -7), textcoords="offset points", ha="right", va="top", color=INK, fontsize=8.8)
sec = a2.secondary_yaxis("right", functions=(lambda v: C0 / v, lambda v: C0 / v))
es = [0, 3, 6, 9, 12, 15, 18]; sec.set_yticks([10.0 ** e for e in es]); sec.set_yticklabels(["today"] + ["x" + (f"{10**e:,}" if e <= 3 else sup(e)) for e in es[1:]])
sec.tick_params(colors=MUTED, labelsize=9.5, length=0, pad=6); sec.spines["right"].set_visible(False); sec.minorticks_off()
a2.annotate("algorithmic\nefficiency", (1.0, 1.0), xycoords="axes fraction", xytext=(8, 4), textcoords="offset points", ha="left", va="bottom", color=INK2, fontsize=8.8, annotation_clip=False)
a2.axvline(T, color=INK, lw=1, ls=(0, (1, 2)), zorder=2)
for ax in (a1, a2): ax.set_xlabel("Years after software research is automated", color=INK2, fontsize=9.5, labelpad=8)
handles = [Line2D([0], [0], color=BLUE, lw=2.4, label="The model's path"),
           Line2D([0], [0], color=ORANGE, lw=2.4, label="More chips alone"),
           Line2D([0], [0], color=INK, lw=2.2, label="Today's pace of algorithmic progress")]
leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.875), ncol=3, frameon=False, fontsize=9.8, handlelength=1.8, columnspacing=1.8, handletextpad=0.6)
for tx in leg.get_texts(): tx.set_color(INK)
fig.subplots_adjust(left=0.07, right=0.915, top=0.735, bottom=0.2)
fig.text(0.06, 0.955, "What the growth model asks of each AI researcher", color=INK, fontsize=15, ha="left", va="center")
fig.text(0.06, 0.905, "Davidson et al.'s headline scenario: AI takes over all software research, and the singularity follows 5.5 years later.", color=INK2, fontsize=10.2, ha="left", va="center")
fig.text(0.06, 0.052, "Re-run of Davidson, Halperin, Houlden and Korinek (2026), NBER Working Paper 35155, with the human brain set to 10¹⁶ operations a second.\nPercentages are yearly growth in world output. Assumes today's AI chips do the work of 20,000 researchers. Example machines are approximate.",
         color=MUTED, fontsize=8.3, ha="left", va="center", linespacing=1.4)
fig.savefig(out_png, facecolor=SURFACE)
import csv
with open(os.path.join(os.path.dirname(out_png), "..", "results", "compute_per_researcher.csv"), "w", newline="") as fo:
    wr = csv.writer(fo); wr.writerow(["output_growth", "year", "software_gain", "compute_stock_gain", "ai_researchers", "chips_only_researchers", "ops_per_sec_each"])
    for lev in (0.1, 0.3, 0.5, 1.0, 2.0):
        i = int(np.argmax(gY >= lev)); g = (lev - gY[i - 1]) / (gY[i] - gY[i - 1]) if i else 0.0
        S = math.exp(U[0, i - 1] + g * (U[0, i] - U[0, i - 1])) if i else 1.0; KH = math.exp(U[4, i - 1] + g * (U[4, i] - U[4, i - 1])) if i else 1.0
        wr.writerow([lev, round(float(ts[i]), 3), f"{S:.4g}", f"{KH:.4g}", f"{N0 * S * KH:.4g}", f"{N0 * KH:.4g}", f"{C0 / S:.4g}"])
        print(f"  {lev:4.0%} growth at year {ts[i]:.2f}: software x{S:.3g}, AI researchers {N0*S*KH:.3g} (chips alone {N0*KH:.3g}), {C0/S:.3g} ops/s each")
print("wrote", out_png)
