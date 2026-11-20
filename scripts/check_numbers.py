"""Print every number quoted in the post next to the value recomputed from results/. Run after run_all.sh."""
import os
import numpy as np, pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rd = lambda name: pd.read_csv(os.path.join(R, name))
rows = []
add = lambda claim, post, value: rows.append((claim, post, value))

cw = rd("children_per_woman_required.csv").set_index("year")
add("Doomsday date", "13 Nov 2026", "t0 = 2026.87 (von Foerster et al. 1960)")
add("Children per woman required in 2018", "35", f"{cw.loc[2018, 'children_per_woman_required']:.1f}")
add("Births exceed women aged 15-49", "2018", str(int(cw.index[(cw.births_per_year_bn >= cw.women_15_49_bn).values][0])))
bh = rd("births_halved_1960.csv").set_index("year").loc[2023]
add("Counterfactual population, share of actual (2023)", "about 40%", f"{bh.halved_then_tracking_history_bn / bh.actual_bn:.1%}")
add("Counterfactual share aged 65+, ratio to actual", "2.5x", f"{bh.share_65plus_tracking / bh.share_65plus_actual:.2f}x")

cp = rd("compute_per_researcher.csv").set_index("output_growth")
add("Years until output growth passes 200%", "5.5", f"{cp.loc[2.0, 'year']:.2f}")
add("AI researchers at the 200% threshold", "10^25", f"{cp.loc[2.0, 'ai_researchers']:.2g} (chips alone {cp.loc[2.0, 'chips_only_researchers']:.2g})")
for g, m in ((0.3, "Xbox 360, about 2.4e11"), (0.5, "Pentium PC running Windows 95, about 1e8"), (1.0, "Manchester Baby, about 1.1e3")):
    add(f"Ops/s per researcher at {g:.0%} growth", m, f"{cp.loc[g, 'ops_per_sec_each']:.2g}")
add("Ops per researcher at 200% growth", "about 1 every 15 minutes", f"1 every {1 / cp.loc[2.0, 'ops_per_sec_each'] / 60:.0f} minutes")

tt = rd("tail_tests_top1pct.csv")
for _, r in tt.iterrows():
    add(f"Figure 4, {r.dataset}", "", f"c = {r.c:.2f}, LR = {r.LR:.1f}, Vuong z = {r.vuong_z:+.2f}, R2 Pareto {r.r2_pareto:.3f} / SE {r.r2_stretched:.3f}")

d = np.load(os.path.join(R, "singularity_sweep.npz")); G, rs, cs = d["grid"], d["rs"], d["cs"]
row = lambda r: G[int(np.argmin(abs(rs - r)))]
add("Spark at r = 0.83, c from 0.1 to 0.01", "1.1x to 1.9x", f"{row(0.83)[-1]:.2f} to {row(0.83)[0]:.2f}")
add("Spark at r = 1, c from 0.1 to 0.01", "1.5x to 4.6x", f"{row(1.0)[-1]:.2f} to {row(1.0)[0]:.2f}")
j = int(np.argmax(row(1.0) < 3))
add("At r = 1, at least 3 years of progress only when", "c < 0.02", f"c < {np.sqrt(cs[j - 1] * cs[j]):.3f}")
add("Singular runs with c > 0", "none", str(int(np.isinf(G).sum())))

w = max(len(r[0]) for r in rows)
for claim, post, value in rows:
    print(f"{claim:<{w}}  post: {post:<40}  here: {value}")
