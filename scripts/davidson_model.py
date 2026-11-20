"""Re-implementation of the simulation model in Davidson, Halperin, Houlden & Korinek (2026), "When Does Automating AI
Research Produce Explosive Growth?", NBER Working Paper 35155: their eqs 47-55 with the calibration of their section 5.3.

State (all in logs, relative to the date of the automation shock): software level s, hardware quality h, general
productivity a, capital in goods production kY, compute capital kH. f[i] is the share of tasks automated in sector i
(S software research, H hardware research, A general research, Y goods production). Effective labour in sector i is
(1 - f) human labour growing at n[i], plus f AI labour = compute stock x software level.

simulate(f) integrates until world output growth passes `stop` (default 300% a year) or t_end.
Running this file prints the explosion condition and the date output growth passes 200% a year for four scenarios, and
writes the path of the paper's headline scenario (software research 100% automated, 5% everywhere else) to
results/davidson_software_path.csv.
Run: python3 scripts/davidson_model.py
"""
import os, sys, math, csv
import numpy as np
from scipy.integrate import solve_ivp
# Davidson, Halperin, Houlden & Korinek (2026), eqs 47-55, calibration of section 5.3
alpha = 0.4
r = dict(S=1.0, H=5.0, A=0.32); lam = dict(S=1.0, H=1.0, A=1.0); beta = {k: lam[k] / r[k] for k in r}
g0 = dict(S=1.10, H=0.37); n = dict(S=1.10, H=0.074, A=0.04, Y=0.01); g0["A"] = r["A"] * n["A"]
dY, dH = 0.07, 0.20
C0 = 1e16      # operations a second one AI researcher runs on at the date of automation (Figure 3)
gY0 = g0["A"] / (1 - alpha) + n["Y"]; gKH0 = g0["H"] + gY0
def simulate(f, t_end=12.0, stop=3.0):
    def parts(t, u):
        s, h, a, kY, kH = u
        Lh = {i: (1 - f[i]) * n[i] * t + f[i] * (kH + s) for i in "SHAY"}
        ds = g0["S"] * math.exp(lam["S"] * Lh["S"] - beta["S"] * s)
        dh = g0["H"] * math.exp(lam["H"] * Lh["H"] - beta["H"] * h)
        da = g0["A"] * math.exp(lam["A"] * Lh["A"] - beta["A"] * a)
        y = a + (1 - alpha) * Lh["Y"] + alpha * kY
        dkY = (gY0 + dY) * math.exp(y - kY) - dY
        dkH = (gKH0 + dH) * math.exp(h + y - kH) - dH
        gY = da + (1 - alpha) * ((1 - f["Y"]) * n["Y"] + f["Y"] * (dkH + ds)) + alpha * dkY
        return [ds, dh, da, dkY, dkH], gY
    rhs = lambda t, u: parts(t, u)[0]
    ev = lambda t, u: parts(t, u)[1] - stop
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rhs, (0, t_end), [0, 0, 0, 0, 0], method="LSODA", dense_output=True, rtol=1e-10, atol=1e-12, events=ev, max_step=0.01)
    return sol, parts
if __name__ == "__main__":
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "davidson_software_path.csv")
    print(f"calibration: beta {beta}  gY0 {gY0:.4f}  a_Y {(gY0+dY)*4:.3f}  gKH0 {gKH0:.4f}")
    for label, f in (("software 100%, 5% elsewhere", dict(S=1, H=.05, A=.05, Y=.05)), ("uniform 20%", dict(S=.2, H=.2, A=.2, Y=.2)), ("uniform 15%", dict(S=.15, H=.15, A=.15, Y=.15)), ("software 80%, 5% elsewhere", dict(S=.8, H=.05, A=.05, Y=.05))):
        EC = f["Y"] + f["S"] * r["S"] + f["H"] * r["H"] + f["A"] * r["A"] / (1 - alpha)
        sol, parts = simulate(f, t_end=40)
        T = sol.t[-1]; ts = np.linspace(0, T, 200001); U = sol.sol(ts); gY = np.array([parts(t, U[:, i])[1] for i, t in enumerate(ts)])
        t200 = float(ts[int(np.argmax(gY >= 2.0))]) if gY.max() >= 2.0 else None
        print(f"\n{label}: explosion condition {EC:.2f}; output growth passes 200% a year at {t200:.2f} yr (and 300% at {T:.2f})" if t200 is not None else f"\n{label}: EC {EC:.2f}; did not reach 200% by {T:.1f}")
        if label.startswith("software 100"):
            rows = []
            for lev in (0.10, 0.30, 0.50, 1.00, 2.00):
                i = int(np.argmax(gY >= lev)); s, h, a, kY, kH = U[:, i]
                print(f"  growth {lev*100:4.0f}% at year {ts[i]:.2f}:  S x{math.exp(s):.3g}  H x{math.exp(h):.3g}  compute stock x{math.exp(kH):.3g}  AI labour x{math.exp(s+kH):.3g}  years of progress at today's pace {s/1.1:.1f}")
                rows.append((lev, ts[i], math.exp(s), math.exp(h), math.exp(kH)))
            with open(out_csv, "w", newline="") as fo:
                w = csv.writer(fo); w.writerow(["year_after_shock", "output_growth_per_year", "software_S_multiple", "hardware_H_multiple", "compute_stock_multiple", "ops_per_sec_per_researcher"])
                for i in range(0, len(ts), 500):
                    s, h, a, kY, kH = U[:, i]; w.writerow([round(ts[i], 3), round(float(gY[i]), 4), f"{math.exp(s):.4g}", f"{math.exp(h):.4g}", f"{math.exp(kH):.4g}", f"{C0/math.exp(s):.4g}"])
            print(f"  operations a second per AI researcher ({C0:g} each at the date of automation):")
            for lev, tt, S, H, KH in rows: print(f"    {lev*100:4.0f}%: {C0/S:.3g} ops/s")
