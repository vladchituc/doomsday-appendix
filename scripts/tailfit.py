"""Tail-fitting tools: power law vs stretched exponential vs lognormal.

Data are passed as (x, w): sorted unique values and their counts. Integer-valued data
(citations) use a binned likelihood, P(k) = S(k-1/2) - S(k+1/2); continuous data use densities.
All models are fitted conditional on x >= xmin, so they compete on the same tail.

Models (survival functions conditional on x >= u):
  power law (PL):            ln S = -a * L,                      L = ln(x/u)
  stretched exponential (SE): ln S = -(b/c) * (exp(c L) - 1)      (c -> 0 recovers PL with a = b)
                              i.e. S = exp(-(x/x0)^c + (u/x0)^c), with x0 = u (c/b)^(1/c)
  lognormal (LN):            S = Phi_bar((ln x - mu)/sigma) / Phi_bar((ln u - mu)/sigma)

Tests:
  * Vuong likelihood-ratio test between any two models (Clauset, Shalizi & Newman 2009).
  * Wilks nested test of PL inside SE (Malevergne, Pisarenko & Sornette 2005): 2*(LL_SE - LL_PL)
    is asymptotically a 50:50 mixture of 0 and chi2(1) because c = 0 is on the boundary.
  * Laherrere-Sornette (1998) rank-ordering fit: x_n^c linear in ln(rank).
"""
import numpy as np
from scipy import optimize, special, stats

# ---------------------------------------------------------------- data helpers
def hist_from_values(v):
    v = np.sort(np.asarray(v, dtype=float))
    return v, np.ones_like(v)

def tail(x, w, xmin):
    m = x >= xmin
    return x[m], w[m]

def ccdf(x, w):
    """Empirical P(X >= x) at each unique x."""
    n_ge = np.cumsum(w[::-1])[::-1]
    return n_ge / w.sum()

# ---------------------------------------------------------------- log-survival functions
def lnS_pl(x, u, a):
    return -a * np.log(x / u)

def lnS_se(x, u, b, c):
    L = np.log(x / u)
    if c < 1e-12:
        return -b * L
    return -(b / c) * np.expm1(c * L)

def lnS_ln(x, u, mu, sig):
    return special.log_ndtr(-(np.log(x) - mu) / sig) - special.log_ndtr(-(np.log(u) - mu) / sig)

def _ll_points(lnS, lnf, x, u, discrete):
    """Per-observation log-likelihood. lnS(x) is conditional log survival, lnf(x) the log density."""
    if not discrete:
        return lnf(x)
    lo, hi = lnS(x - 0.5), lnS(x + 0.5)
    return lo + np.log1p(-np.exp(np.minimum(hi - lo, -1e-300)))

class Fit:
    def __init__(self, name, params, ll_pts, w, k):
        self.name, self.params, self.ll_pts, self.k = name, params, ll_pts, k
        self.ll = float(np.dot(w, ll_pts)); self.n = float(w.sum())
        self.aic = 2 * k - 2 * self.ll

def _edge(xmin, discrete):
    return xmin - 0.5 if discrete else xmin

def fit_pl(x, w, xmin, discrete):
    u = _edge(xmin, discrete)
    def pts(a):
        return _ll_points(lambda t: lnS_pl(t, u, a), lambda t: np.log(a / u) - (a + 1) * np.log(t / u), x, u, discrete)
    a0 = w.sum() / np.dot(w, np.log(x / u))  # Hill estimator
    if discrete:
        r = optimize.minimize_scalar(lambda a: -np.dot(w, pts(a)), bounds=(1e-3, 50), method="bounded")
        a0 = r.x
    return Fit("PL", dict(alpha=a0), pts(a0), w, 1)

def fit_se(x, w, xmin, discrete, start=None):
    u = _edge(xmin, discrete)
    def pts(b, c):
        lnS = lambda t: lnS_se(t, u, b, c)
        lnf = lambda t: lnS(t) + np.log(b) + c * np.log(t / u) - np.log(t)
        return _ll_points(lnS, lnf, x, u, discrete)
    def nll(p):
        b, c = np.exp(p[0]), np.exp(p[1])
        v = -np.dot(w, pts(b, c))
        return v if np.isfinite(v) else 1e300
    a_h = w.sum() / np.dot(w, np.log(x / u))
    best = None
    for c0 in ([start[1]] if start else []) + [0.3, 0.05, 1.0]:
        b0 = start[0] if (start and c0 == start[1]) else a_h
        r = optimize.minimize(nll, [np.log(b0), np.log(c0)], method="Nelder-Mead",
                              options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    b, c = np.exp(best.x)
    x0 = u * (c / b) ** (1 / c) if c > 1e-8 else 0.0
    return Fit("SE", dict(b=b, c=c, x0=x0), pts(b, c), w, 2)

def fit_ln(x, w, xmin, discrete):
    u = _edge(xmin, discrete)
    def pts(mu, sig):
        lnS = lambda t: lnS_ln(t, u, mu, sig)
        lnf = lambda t: (stats.norm.logpdf((np.log(t) - mu) / sig) - np.log(sig * t)
                         - special.log_ndtr(-(np.log(u) - mu) / sig))
        return _ll_points(lnS, lnf, x, u, discrete)
    def nll(p):
        v = -np.dot(w, pts(p[0], np.exp(p[1])))
        return v if np.isfinite(v) else 1e300
    lx = np.log(x); m0 = np.dot(w, lx) / w.sum(); s0 = np.sqrt(np.dot(w, (lx - m0) ** 2) / w.sum())
    best = None
    for mu0, sg0 in [(m0, s0), (np.log(u), 1.0), (np.log(u) - 3, 2.0), (0.0, 2.0)]:
        r = optimize.minimize(nll, [mu0, np.log(sg0)], method="Nelder-Mead",
                              options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    mu, sig = best.x[0], np.exp(best.x[1])
    return Fit("LN", dict(mu=mu, sigma=sig), pts(mu, sig), w, 2)

def vuong(f1, f2, w):
    """Normalised log-likelihood ratio (positive favours f1) and two-sided p-value."""
    d = f1.ll_pts - f2.ll_pts
    n = w.sum(); R = np.dot(w, d); m = R / n
    var = np.dot(w, (d - m) ** 2) / n
    if var <= 0:
        return 0.0, 1.0
    z = R / np.sqrt(n * var)
    return float(z), float(special.erfc(abs(z) / np.sqrt(2)))

def wilks_se_vs_pl(f_se, f_pl):
    W = max(2 * (f_se.ll - f_pl.ll), 0.0)
    return W, 0.5 * stats.chi2.sf(W, 1)

def se_profile_ci(x, w, xmin, discrete, level=3.841):
    """MLE of the SE exponent c with a profile-likelihood 95% interval (lower bound floored at 0)."""
    xt, wt = tail(x, w, xmin)
    u = _edge(xmin, discrete)
    full = fit_se(xt, wt, xmin, discrete)
    def prof(c):
        def nll(lb):
            b = np.exp(lb)
            lnS = lambda t: lnS_se(t, u, b, c)
            lnf = lambda t: lnS(t) + np.log(b) + c * np.log(t / u) - np.log(t)
            v = -np.dot(wt, _ll_points(lnS, lnf, xt, u, discrete))
            return v if np.isfinite(v) else 1e300
        r = optimize.minimize_scalar(nll, bounds=(-12, 6), method="bounded", options=dict(xatol=1e-7))
        return -r.fun
    c_hat, ll_hat = full.params["c"], full.ll
    g = lambda c: 2 * (ll_hat - prof(c)) - level
    lo = 0.0
    if c_hat > 2e-4 and g(1e-4) > 0:
        lo = optimize.brentq(g, 1e-4, c_hat)
    hi_b = max(c_hat * 2, 0.05)
    while g(hi_b) < 0 and hi_b < 20:
        hi_b *= 2
    hi = optimize.brentq(g, max(c_hat, 1e-4), hi_b) if g(hi_b) > 0 else np.nan
    return c_hat, lo, hi, int(wt.sum())

# ---------------------------------------------------------------- local slopes
def slope_pl(x, fit): return np.full_like(np.asarray(x, float), fit.params["alpha"])
def slope_se(x, u, fit): return fit.params["b"] * (np.asarray(x, float) / u) ** fit.params["c"]
def slope_ln(x, fit):
    z = (np.log(x) - fit.params["mu"]) / fit.params["sigma"]
    return np.exp(stats.norm.logpdf(z) - special.log_ndtr(-z)) / fit.params["sigma"]

def local_slope(x, w, discrete, factor=2.0, min_count=30, grid=40, lo=None):
    """Non-parametric local tail exponent: alpha(u) = ln[S(u)/S(factor*u)] / ln(factor).

    For a pure power law this is flat in u. Returns u, alpha, standard error.
    """
    n_ge = np.cumsum(w[::-1])[::-1]
    lo = lo if lo is not None else x[0]
    us = np.unique(np.round(np.geomspace(max(lo, 1), x[-1] / factor, grid)).astype(float)) if discrete \
        else np.geomspace(lo, x[-1] / factor, grid)
    out = []
    for u in us:
        e1 = u - 0.5 if discrete else u
        e2 = factor * e1
        i1, i2 = np.searchsorted(x, e1, "left"), np.searchsorted(x, e2, "left")
        n1 = n_ge[i1] if i1 < len(x) else 0
        n2 = n_ge[i2] if i2 < len(x) else 0
        if n2 < min_count or n1 <= n2:
            continue
        a = np.log(n1 / n2) / np.log(factor)
        se = np.sqrt(1 / n2 - 1 / n1) / np.log(factor)
        out.append((e1 * np.sqrt(factor), a, se))  # report at the geometric midpoint
    return np.array(out).T if out else np.zeros((3, 0))

# ---------------------------------------------------------------- xmin by KS (CSN)
def ks_pl(x, w, xmin, discrete):
    xt, wt = tail(x, w, xmin)
    f = fit_pl(xt, wt, xmin, discrete)
    u = _edge(xmin, discrete)
    emp = ccdf(xt, wt)
    mod = np.exp(lnS_pl(xt - 0.5 if discrete else xt, u, f.params["alpha"]))
    return float(np.max(np.abs(emp - mod))), f.params["alpha"]

def choose_xmin(x, w, discrete, min_tail=1000, ncand=60):
    n_ge = np.cumsum(w[::-1])[::-1]
    ok = np.where(n_ge >= min_tail)[0]
    cand = x[ok]
    if len(cand) > ncand:
        cand = np.unique(cand[np.unique(np.round(np.geomspace(1, len(cand), ncand)).astype(int) - 1)])
    cand = cand[cand > 0]
    ks = [ks_pl(x, w, c, discrete)[0] for c in cand]
    return float(cand[int(np.argmin(ks))])

# ---------------------------------------------------------------- Laherrere-Sornette rank ordering
def rank_order_fit(x, w, xmin, max_pts=4000):
    """Fit x_n^c = -a ln n + b on the rank plot (n = number of items >= x).

    Returns c that maximises R^2, that R^2, and the R^2 of the power-law line ln x vs ln n.
    Points are the unique tail values (thinned evenly in log-rank if very many).
    """
    xt, wt = tail(x, w, xmin)
    rank = np.cumsum(wt[::-1])[::-1].astype(float)   # n(x) = count of items >= x
    if len(xt) > max_pts:   # thin evenly in log-rank
        lr = np.log(rank)
        target = np.linspace(lr[0], lr[-1], max_pts)
        idx = np.unique(np.round(np.interp(target, lr[::-1], np.arange(len(xt))[::-1])).astype(int))
        xt, rank = xt[idx], rank[idx]
    ln_n = np.log(rank)
    def r2(y):
        A = np.vstack([ln_n, np.ones_like(ln_n)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        ss = np.sum((y - y.mean()) ** 2)
        return 1 - np.sum((y - A @ coef) ** 2) / ss, coef
    with np.errstate(all="ignore"):
        res = optimize.minimize_scalar(lambda c: -r2(xt ** c)[0], bounds=(1e-3, 3), method="bounded")
    c = res.x
    r2_se, coef = r2(xt ** c)
    r2_pl, coef_pl = r2(np.log(xt))
    return dict(c=c, r2_se=r2_se, a=-coef[0], b=coef[1], r2_pl=r2_pl, mu=-1 / coef_pl[0],
                x=xt, rank=rank, coef_pl=coef_pl)

# ---------------------------------------------------------------- one-call summary
def analyse(x, w, xmin, discrete):
    xt, wt = tail(x, w, xmin)
    pl = fit_pl(xt, wt, xmin, discrete)
    se = fit_se(xt, wt, xmin, discrete)
    ln = fit_ln(xt, wt, xmin, discrete)
    z_se, p_se = vuong(se, pl, wt)
    z_ln, p_ln = vuong(ln, pl, wt)
    z_sl, p_sl = vuong(se, ln, wt)
    W, pW = wilks_se_vs_pl(se, pl)
    ro = rank_order_fit(x, w, xmin)
    aics = np.array([pl.aic, se.aic, ln.aic]); wts = np.exp(-(aics - aics.min()) / 2); wts /= wts.sum()
    return dict(xmin=xmin, n_tail=int(wt.sum()), alpha_pl=pl.params["alpha"],
                se_b=se.params["b"], se_c=se.params["c"], se_x0=se.params["x0"],
                ln_mu=ln.params["mu"], ln_sigma=ln.params["sigma"],
                ll_pl=pl.ll, ll_se=se.ll, ll_ln=ln.ll,
                vuong_se_pl=z_se, p_se_pl=p_se, vuong_ln_pl=z_ln, p_ln_pl=p_ln,
                vuong_se_ln=z_sl, p_se_ln=p_sl, wilks=W, p_wilks=pW,
                aicw_pl=wts[0], aicw_se=wts[1], aicw_ln=wts[2],
                ro_c=ro["c"], ro_r2_se=ro["r2_se"], ro_r2_pl=ro["r2_pl"]), dict(pl=pl, se=se, ln=ln, ro=ro)
