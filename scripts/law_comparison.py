"""Section 1.5 of the README: how far Figure 7's software law is from the law derived exactly from a search model.

Both laws replace the software line of Davidson, Halperin, Houlden and Korinek's (2026) headline scenario (the model in
fig7_singularity_map.py, with the calibration in davidson_model.py). s is the log software level relative to today, L_S the log
research input, r the returns to software research today, and c the Weibull shape of the idea distribution.

  curved Jones (Figure 7):  ds/dt = g0 * exp(L_S - (1/r) * (e^(c s) - 1) / c)
                            research x the chance a try beats the frontier, with the size of a step held fixed.
                            Local exponent beta(s) = (1/r) e^(c s).
  exact search law:         ds/dt = g0 * exp(L_S - c s - (1/r - c) * (e^(c s) - 1) / c)
                            the same, with the step shrinking as a Weibull tail implies (the extra term -c s).
                            Local exponent beta(s) = c + (1/r - c) e^(c s).

Both are pinned to the same pace today (g0 = 1.10 log points a year) and the same beta today (1/r). The outcome is
Figure 7's: the most software progress made in any 12 months of a 40-year run, in years of progress at today's pace.

Outputs: results/law_comparison.csv
Run: python3 scripts/law_comparison.py      (under a minute)
"""
import os, math, csv
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool
from davidson_model import alpha, beta, g0, n, dY, dH, gY0, gKH0      # Davidson et al. (2026), section 5.3 calibration

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
bH, bA = beta["H"], beta["A"]; g0S, g0H, g0A = g0["S"], g0["H"], g0["A"]
f = dict(S=1.0, H=.05, A=.05, Y=.05)        # headline scenario: software research 100% automated, 5% elsewhere
HORIZON = 40.0

RS, CS = (0.83, 1.0, 1.2, 1.5), (0.01, 0.02, 0.03, 0.05, 0.1)


def run(job):
    law, r, c = job

    def parts(t, u):
        s, h, a, kY, kH = u
        L = {i: (1 - f[i]) * n[i] * t + f[i] * (kH + s) for i in "SHAY"}
        e = math.expm1(min(c * s, 700.0)) / c
        if law == "curved":
            ds = g0S * math.exp(min(L["S"] - e / r, 700.0))
        else:
            ds = g0S * math.exp(min(L["S"] - c * s - (1 / r - c) * e, 700.0))
        dh = g0H * math.exp(L["H"] - bH * h)
        da = g0A * math.exp(L["A"] - bA * a)
        y = a + (1 - alpha) * L["Y"] + alpha * kY
        return [ds, dh, da, (gY0 + dY) * math.exp(y - kY) - dY, (gKH0 + dH) * math.exp(h + y - kH) - dH]

    def rhs(sig, w):                                   # stretched clock, as in fig7_singularity_map.py
        d = parts(w[5], w[:5]); k = 1.0 / (1.0 + d[0])
        return [x * k for x in d] + [k]
    ev = lambda sig, w: w[5] - HORIZON
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rhs, (0, 1e5), [0] * 6, method="LSODA", dense_output=True, rtol=1e-8, atol=1e-10, events=ev, max_step=2.0)
    sg = np.unique(np.concatenate([sol.t, np.linspace(0, sol.t[-1], 40001)])); W = sol.sol(sg)
    tt = np.maximum.accumulate(W[5]); keep = np.concatenate([[True], np.diff(tt) > 0])
    ts = np.linspace(0, HORIZON, 4001); s = np.interp(ts, tt[keep], W[0][keep])
    gain = s[100:] - s[:-100]                          # 100 steps of 0.01 years = 12 months
    return (law, r, c, float(gain.max() / g0S))


if __name__ == "__main__":
    jobs = [(law, r, c) for r in RS for c in CS for law in ("curved", "exact")]
    with Pool() as p: res = p.map(run, jobs)
    main = {(law, r, c): v for law, r, c, v in res}

    print("fastest 12 months of software progress, in years of progress at today's pace")
    print(f"{'r':>5} {'c':>6} | {'curved Jones':>12} {'exact search':>12} {'ratio':>6}")
    with open(os.path.join(ROOT, "results", "law_comparison.csv"), "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["returns_r", "weibull_shape_c", "fastest_year_curved_jones", "fastest_year_exact_search_law", "ratio_exact_to_curved"])
        ratios = []
        for r in RS:
            for c in CS:
                a, b = main[("curved", r, c)], main[("exact", r, c)]; ratios.append(b / a)
                print(f"{r:5.2f} {c:6.3f} | {a:12.2f} {b:12.2f} {b / a:6.3f}")
                w.writerow([r, c, round(a, 3), round(b, 3), round(b / a, 4)])
    print(f"exact / curved across the {len(ratios)} cells: {min(ratios):.3f} to {max(ratios):.3f}")

    print("wrote results/law_comparison.csv")
