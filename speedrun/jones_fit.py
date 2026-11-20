"""The stochastic Jones law of Erdil, Besiroglu and Ho (2024), section 4.3.3, equation 26.

For two successive observations of efficiency A at times t1 < t2, with research input I(t) in between:

    (A2^beta - A1^beta) / beta  ~  Stable(alpha, skew = 1, location = mu * J, scale = c * A1^(beta - beta/alpha) * J^(1/alpha))
    J = integral of I(t)^lam dt over (t1, t2]

beta   how fast progress slows as A rises
lam    how strongly the rate of progress responds to input
alpha  tail index of the noise (2 is Gaussian; smaller values allow jumps); the noise is maximally right-skewed
mu, c  drift and noise scale
Returns to research effort are r = lam / beta.

The stable density has no closed form. scipy's is slow, so its log-density is tabulated once on an (alpha, z) grid
(stable_table.npz) and interpolated; `scipy_nll` evaluates the same likelihood with scipy directly as a check.
"""
import os
import numpy as np, pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.stats import levy_stable

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "stable_table.npz")
START = "2024-10-04"        # record 3, the first record after the repository opened to outside pull requests


def build_table():
    """Log-density of the standardised law (location 0, scale 1, skew 1; scipy's S1 parameterisation)."""
    al = np.unique(np.concatenate([np.linspace(1.02, 1.9, 89), np.linspace(1.9, 2.0, 41)]))
    u = np.linspace(np.arcsinh(-40), np.arcsinh(2e4), 700)
    T = np.array([np.log(np.maximum(levy_stable.pdf(np.sinh(u), a, 1.0), 1e-300)) for a in al])
    np.savez(TABLE, al=al, u=u, T=T)


_tab = None
def _table():
    global _tab
    if _tab is None:
        if not os.path.exists(TABLE):
            build_table()
        z = np.load(TABLE)
        _tab = (RegularGridInterpolator((z["al"], z["u"]), z["T"], bounds_error=False, fill_value=None), z["u"])
    return _tab


def load(input_col, start=START):
    """Transitions between successive record days, with the daily input over each interval (scaled to mean 1)."""
    A = pd.read_csv(os.path.join(HERE, "data", "efficiency.csv"), parse_dates=["date"])
    A = A[A.date >= start].reset_index(drop=True)
    I = pd.read_csv(os.path.join(HERE, "data", "inputs_daily.csv"), parse_dates=["date"]).set_index("date")[input_col].astype(float)
    I = I.loc[A.date.iloc[0]:A.date.iloc[-1]]
    I = I.clip(lower=I[I > 0].min() / 2)        # days with no activity get half the smallest positive value
    I = I / I.mean()
    segs = [I.loc[a + pd.Timedelta(days=1):b].values for a, b in zip(A.date[:-1], A.date[1:])]
    return dict(A1=A.A.values[:-1], A2=A.A.values[1:], segs=segs, years=sum(len(s) for s in segs) / 365.25, daily=I.values)


def unpack(p, fix_lam=None):
    """Optimiser coordinates -> (beta, lam, alpha, mu, c). alpha is kept in (1.02, 2); mu and c are positive."""
    if fix_lam is None:
        b, lam, a, m, lc = p
    else:
        (b, a, m, lc), lam = p, fix_lam
    return b, lam, 1.02 + 0.98 / (1 + np.exp(-a)), np.exp(m), np.exp(lc)


def _pieces(p, d, idx, fix_lam):
    b, lam, al, mu, c = unpack(p, fix_lam)
    if abs(b) < 1e-4:
        b = 1e-4
    sel = range(len(d["A1"])) if idx is None else idx
    A1, A2 = d["A1"][sel], d["A2"][sel]
    J = np.array([np.sum(d["segs"][i] ** lam) for i in sel]) / 365.25
    X = (A2 ** b - A1 ** b) / b
    return X, al, mu * J, c * A1 ** (b - b / al) * J ** (1 / al), (b - 1) * np.log(A2)     # last term: Jacobian dX/dA2


def loglik_terms(p, d, idx=None, fix_lam=None):
    X, al, loc, scale, jac = _pieces(p, d, idx, fix_lam)
    f, u = _table()
    z = np.clip(np.arcsinh((X - loc) / scale), u[0], u[-1])
    return f(np.column_stack([np.full_like(z, al), z])) - np.log(scale) + jac


def nll(p, d, idx=None, fix_lam=None):
    if np.any(np.abs(p) > 30):
        return 1e10
    v = -np.sum(loglik_terms(p, d, idx, fix_lam))
    return v if np.isfinite(v) else 1e10


def scipy_nll(p, d):
    X, al, loc, scale, jac = _pieces(p, d, None, None)
    return -np.sum(np.log(np.maximum(levy_stable.pdf(X, al, 1.0, loc=loc, scale=scale), 1e-300)) + jac)


STARTS = [[0.5, 0.4, 1.0, 2.0, 0.0], [2.5, 1.5, 1.0, 5.0, 1.0], [1.0, 1.0, 1.0, 4.0, 0.5], [0.3, 0.3, 1.0, 0.0, -1.0], [1.5, 0.8, 0.0, 5.0, 1.5]]


def fit(d, idx=None, fix_lam=None, starts=None):
    """Maximum likelihood by Nelder-Mead from several starting points; returns (coordinates, negative log-likelihood)."""
    best = None
    for s in (starts or STARTS):
        s = list(s) if fix_lam is None or len(s) == 4 else [s[0], s[2], s[3], s[4]]
        r = minimize(nll, s, args=(d, idx, fix_lam), method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6, adaptive=True))
        if best is None or r.fun < best[1]:
            best = (r.x, r.fun)
    return best


def naive(d):
    """Growth rate of A divided by growth rate of the input (log-linear trend of the daily series)."""
    gA = np.log(d["A2"][-1] / d["A1"][0]) / d["years"]
    gI = np.polyfit(np.arange(len(d["daily"])) / 365.25, np.log(d["daily"]), 1)[0]
    return gA, gI, gA / gI


if __name__ == "__main__":
    build_table()
    print("wrote", TABLE)
