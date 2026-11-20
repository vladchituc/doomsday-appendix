"""Section 1.5 of the README: simulate the search process itself and compare it with the two deterministic laws.

Software loop alone (compute held fixed, software research fully automated, so research effort is proportional to the
software level S). Every try draws an idea of quality X from a Weibull distribution with shape c; S is the best draw so
far; tries arrive at a rate proportional to S.

In units of w = (x/x0)^c a draw is Exponential(1). By memorylessness, a draw that beats the frontier W lands at
W + Exponential(1), and the number of tries until that happens is geometric with success probability e^(-W), which for
large e^W is exponential with mean e^W. The log software level is s = ln(W/W0)/c. Today's level W0 = (1/r - c)/c sets
the exact search law's local exponent today to 1/r, and the try rate is set so that today's pace is g0.

Compared with the time each deterministic law takes to raise log software by s, T(s) = (1/g0) * integral of e^(-phi(u)):
  exact search law:   phi(u) = u - c u - (1/r - c) (e^(c u) - 1) / c
  curved Jones:       phi(u) = u - (1/r) (e^(c u) - 1) / c

Output: results/search_simulation.csv
Run: python3 scripts/search_simulation.py [paths]      (default 20,000 paths per case)
"""
import os, sys, csv
import numpy as np
from scipy.integrate import quad

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
g0 = 1.10                                   # today's pace of software progress, log points a year (Davidson et al. 2026)
TARGETS = [2, 5, 10, 15, 20]                # rises in log software
CASES = [(1.2, 0.05), (1.2, 0.02), (1.0, 0.05)]     # (returns r, Weibull shape c)


def law_time(s_target, r, c, law):
    b0 = 1 / r
    if law == "exact": phi = lambda u: u - c * u - (b0 - c) * np.expm1(c * u) / c
    else:              phi = lambda u: u - b0 * np.expm1(c * u) / c
    return quad(lambda u: np.exp(-phi(u)), 0, s_target, limit=400)[0] / g0


def simulate(r, c, npath, rng):
    W0 = (1 / r - c) / c; a = g0 * c * W0            # tries a year today, in units of e^W0
    out = np.full((npath, len(TARGETS)), np.nan)
    for p in range(npath):
        W, t, k = W0, 0.0, 0
        while k < len(TARGETS):
            s = np.log(W / W0) / c
            while k < len(TARGETS) and s >= TARGETS[k]: out[p, k] = t; k += 1
            if k == len(TARGETS): break
            t += rng.exponential(np.exp(W - W0)) / (a * np.exp(s))     # wait for the next frontier-beating try
            W += rng.exponential(1.0)                                  # and step up
    return out


if __name__ == "__main__":
    npath = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = np.random.default_rng(1)
    with open(os.path.join(ROOT, "results", "search_simulation.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["returns_r", "weibull_shape_c", "rise_in_log_software", "paths", "years_simulated_median", "years_simulated_p25",
                    "years_simulated_p75", "years_exact_search_law", "years_curved_jones"])
        for r, c in CASES:
            T = simulate(r, c, npath, rng)
            print(f"\nr = {r}, c = {c}: years for log software to rise by s (software loop alone, {npath} paths)")
            print(f"{'s':>4} | {'simulated median':>16} {'25%-75%':>15} | {'exact law':>9} {'curved Jones':>12}")
            for j, s in enumerate(TARGETS):
                q = np.percentile(T[:, j], [25, 50, 75]); ex = law_time(s, r, c, "exact"); cu = law_time(s, r, c, "curved")
                print(f"{s:4d} | {q[1]:16.2f} {q[0]:7.2f}-{q[2]:<7.2f} | {ex:9.2f} {cu:12.2f}")
                w.writerow([r, c, s, npath, round(q[1], 2), round(q[0], 2), round(q[2], 2), round(ex, 2), round(cu, 2)])
    print("\nwrote results/search_simulation.csv")
