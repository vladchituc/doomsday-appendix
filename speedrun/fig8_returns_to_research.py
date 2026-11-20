"""Figure 8: "Returns to research, measured on AI itself".

The two NanoGPT speedrun estimates of r (results/r_estimates.csv, written by estimate_r.py) next to the estimates of
Erdil, Besiroglu and Ho (2024): Stockfish from their Table 4 (+/- 1.96 SE) and the four author-count estimates from their
Table 8 (posterior medians and 90% intervals), which are typed in below.

Output:  ../figures/fig8_returns_to_research.png
Run:     python3 speedrun/fig8_returns_to_research.py
"""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
RES, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "..", "figures", "fig8_returns_to_research.png")
est = {r["input"]: r for r in csv.DictReader(open(RES + "/r_estimates.csv"))}
P, C = est["prs_per_day_30d"], est["contributors_30d"]
SR, SR_LO, SR_HI = float(P["r"]), float(P["r_lo95"]), float(P["r_hi95"])
CR, CR_LO, CR_HI = float(C["r"]), float(C["r_lo95"]), float(C["r_hi95"])
LIGHTBLUE = "#86b6ef"
BG, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAYMARK, GRAYLINE = "#2a78d6", "#eb6834", "#898781", "#c8c7bf"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": BG,
    "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK})
D = [  # name, sub, est, lo, hi, color, value text, strong
    ("AI training speed (NanoGPT speedrun)", "effort = attempts submitted, accepted or not", SR, SR_LO, SR_HI, BLUE, f"{SR:.2f}", True),
    ("AI training speed, counted by people", "effort = active contributors; weaker out of sample", CR, CR_LO, CR_HI, LIGHTBLUE, f"{CR:.2f}", True),
    ("Chess engine (Stockfish)", "daily data, 2013–2023", 0.825, 0.825 - 1.96 * 0.15, 0.825 + 1.96 * 0.15, ORANGE, "0.83", True),
    ("Linear programming", "paper-author counts, one comparison", 1.077, 0.245, 3.095, GRAYMARK, "1.08", False),
    ("Computer vision", "paper-author counts, one comparison", 1.437, 0.821, 2.420, GRAYMARK, "1.44", False),
    ("RL sample efficiency", "paper-author counts, one comparison", 1.583, 0.459, 3.673, GRAYMARK, "1.58", False),
    ("SAT solvers", "paper-author counts, one comparison", 3.542, 1.279, 6.897, GRAYMARK, "3.54", False),
]
fig = plt.figure(figsize=(10, 7.6))
ax = fig.add_axes([0.335, 0.215, 0.635, 0.585])
fig.text(0.02, 0.965, "Returns to research, measured on AI itself", fontsize=19, color=INK, va="top")
fig.text(0.02, 0.905,
    "Fitted with the model Erdil and colleagues used for chess, the NanoGPT speedrun comes in at 0.48 counting\n"
    "attempts submitted, or 0.72 counting active contributors. The four estimates above 1 rest on counts of\n"
    "paper authors and a single before-and-after comparison.",
    fontsize=12.5, color=INK2, va="top", linespacing=1.45)
XMAX = 4.2
ys = [0, 1, 2, 3.3, 4.3, 5.3, 6.3]
for y, (name, sub, est, lo, hi, c, vt, strong) in zip(ys, D):
    ax.plot([lo, min(hi, XMAX)], [y, y], color=c if strong else GRAYLINE, lw=3.2 if strong else 2.2, solid_capstyle="round")
    if hi > XMAX:
        ax.annotate("", xy=(XMAX + 0.07, y), xytext=(XMAX - 0.12, y), arrowprops=dict(arrowstyle="-|>", color=GRAYLINE, lw=2.2))
        ax.text(XMAX - 0.02, y + 0.14, f"to {hi:.1f}", fontsize=9.5, color=MUTED, ha="right", va="top")
    ax.plot(est, y, "o", ms=12 if strong else 8.5, color=c, mec=BG, mew=1.8, zorder=3)
    dx = 0.06 if 0.95 < est < 1.15 else 0
    ax.text(est + dx, y - 0.25, vt, fontsize=13 if strong else 10.5, color=c if strong else INK2, ha="center", va="bottom")
    ax.text(-0.025, y - 0.12, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12, color=INK)
    ax.text(-0.025, y + 0.24, sub, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10, color=MUTED)
ax.axvline(1, color=INK, lw=1.3, zorder=1)
ax.text(0.96, -0.72, "no singularity", fontsize=10.5, color=INK2, ha="right", va="center")
ax.text(1.04, -0.72, "singularity possible (r > 1)", fontsize=10.5, color=INK2, ha="left", va="center")
ax.axhline(2.65, color=GRID, lw=0.9)
ax.set_yticks([]); ax.invert_yaxis(); ax.set_ylim(6.8, -1.05); ax.set_xlim(0, XMAX + 0.12)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="x", labelsize=11, length=0, pad=6)
ax.set_xlabel("returns to research, r\nfurther right: each doubling of research effort buys more progress", fontsize=11.5, color=INK2, labelpad=8, linespacing=1.5)
fig.text(0.02, 0.035,
    "Speedrun, Oct 2024 to Aug 2026: efficiency = 45 minutes ÷ record training time (same model, target loss and hardware); effort = pull requests opened per day\n"
    "(30-day average, merged or not), or people who opened a pull request or issue or committed code in the past 30 days; bars = 95% bootstrap range (500 resamples).\n"
    "Code and data: github.com/vladchituc/doomsday-appendix, section E. Chess: Erdil, Besiroglu & Ho (2024), Table 4, ±1.96 SE. Others: their Table 8, 90% intervals.",
    fontsize=9, color=MUTED, va="bottom", linespacing=1.45)
fig.savefig(OUT, dpi=200)
print("wrote figures/fig8_returns_to_research.png")
