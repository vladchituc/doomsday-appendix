"""Clauset, Shalizi and Newman (2009) power-law tests, built on tailfit.py.

Recipe (CSN Box 1):
  1. choose xmin by minimising the KS distance between the data above xmin and the power law fitted to it;
  2. goodness of fit: semi-parametric bootstrap (CSN Sec. 4.1). Synthetic data sets follow the fitted power law
     above xmin and resample the observed data below it; each is refitted from scratch, including xmin.
     p = share of synthetic KS distances at least as large as the observed one. p <= 0.1 rules the power law out;
  3. likelihood-ratio tests against alternatives (CSN Sec. 5, App. C): Vuong's normalised ratio for non-nested
     alternatives (lognormal, exponential, stretched exponential); for the nested power law with exponential cutoff,
     2*(LL_cutoff - LL_pl) against a 50:50 mixture of 0 and chi2(1) because lambda = 0 sits on the boundary.

Conventions follow tailfit.py: x, w are sorted unique values and counts; "alpha" is the exponent of the survival
function (CSN's alpha minus 1). Integer data use the binned likelihood P(k) = S(k-1/2) - S(k+1/2), which matches
CSN's own discrete approximation (their Eq. 3.7 and App. D) and is applied to every model alike.

Two practical departures from CSN, both stated wherever results are reported:
  * the xmin search covers tails holding between `lo_tail` items and a share `hi_frac` of all items
    (default 100 items to the top 20%), not every distinct value;
  * candidate cutoffs are spaced evenly in log(number of items in the tail), 100 of them.
"""
import numpy as np
import mpmath as mp
from scipy import optimize, special, stats
import tailfit as tf


# ------------------------------------------------------------------ KS distance and xmin
def ks_distance(x, w, xmin, discrete, alpha):
    """Two-sided KS distance between the empirical tail CDF and the fitted power law, tail = x >= xmin."""
    xt, wt = tf.tail(x, w, xmin)
    u = xmin - 0.5 if discrete else xmin
    n = wt.sum()
    ge = np.cumsum(wt[::-1])[::-1] / n           # P(X >= x)
    gt = ge - wt / n                              # P(X >  x)
    if discrete:
        m_ge, m_gt = np.exp(tf.lnS_pl(xt - 0.5, u, alpha)), np.exp(tf.lnS_pl(xt + 0.5, u, alpha))
    else:
        m_ge = m_gt = np.exp(tf.lnS_pl(xt, u, alpha))
    return float(max(np.max(np.abs(ge - m_ge)), np.max(np.abs(gt - m_gt))))


def fit_alpha(x, w, xmin, discrete):
    xt, wt = tf.tail(x, w, xmin)
    if not discrete:                              # closed form (Hill)
        return wt.sum() / np.dot(wt, np.log(xt / xmin))
    return tf.fit_pl(xt, wt, xmin, True).params["alpha"]


def candidates(x, w, n_total, lo_tail=100, hi_frac=0.2, k=100):
    n_ge = np.cumsum(w[::-1])[::-1]
    hi = max(min(hi_frac * n_total, n_ge[0]), lo_tail + 1)
    targets = np.geomspace(lo_tail, hi, k)
    idx = np.searchsorted(-n_ge, -targets, "right") - 1   # last x with n_ge >= target
    idx = np.clip(idx, 0, len(x) - 1)
    return np.unique(x[idx])


def choose_xmin(x, w, discrete, n_total=None, lo_tail=100, hi_frac=0.2, k=100, curve=False):
    n_total = w.sum() if n_total is None else n_total
    cand = candidates(x, w, n_total, lo_tail, hi_frac, k)
    rows = []
    for c in cand:
        a = fit_alpha(x, w, c, discrete)
        rows.append((c, a, ks_distance(x, w, c, discrete, a)))
    rows = np.array(rows)
    i = int(np.argmin(rows[:, 2]))
    out = dict(xmin=rows[i, 0], alpha=rows[i, 1], D=rows[i, 2])
    if curve:
        n_ge = np.cumsum(w[::-1])[::-1]
        out["curve"] = dict(xmin=rows[:, 0], alpha=rows[:, 1], D=rows[:, 2],
                            n_tail=n_ge[np.searchsorted(x, rows[:, 0])])
    return out


# ------------------------------------------------------------------ synthetic data
def _pl_draws(rng, m, xmin, alpha, discrete):
    r = rng.random(m)
    if discrete:
        return np.floor((xmin - 0.5) * (1 - r) ** (-1 / alpha) + 0.5)
    return xmin * (1 - r) ** (-1 / alpha)


def _as_hist(v, discrete):
    if discrete:
        x, w = np.unique(v, return_counts=True)
        return x, w.astype(float)
    v = np.sort(v)
    return v, np.ones_like(v)


class Bootstrap:
    """Semi-parametric synthetic data sets (CSN Sec. 4.1) for one observed data set.

    Items below a floor that no candidate xmin can reach are only counted, never materialised, which keeps
    million-item data sets fast. The floor sits at the 1.5*hi_frac quantile so synthetic candidates never need it.
    """
    def __init__(self, x, w, discrete, xmin, alpha, lo_tail=100, hi_frac=0.2):
        self.x, self.w, self.disc, self.xmin, self.alpha = x, w, discrete, xmin, alpha
        self.lo_tail, self.hi_frac = lo_tail, hi_frac
        self.n = w.sum()
        n_ge = np.cumsum(w[::-1])[::-1]
        self.n_tail = n_ge[np.searchsorted(x, xmin)]
        floor_i = np.searchsorted(-n_ge, -min(1.5 * hi_frac, 1.0) * self.n, "right") - 1
        self.floor = x[max(floor_i, 0)] if hi_frac < 0.66 else -np.inf
        body = x < xmin
        mid = body & (x >= self.floor)
        self.n_body = w[body].sum()
        self.mid_x, self.mid_p = x[mid], w[mid] / w[mid].sum() if w[mid].sum() > 0 else w[mid]
        self.p_mid = w[mid].sum() / self.n_body if self.n_body > 0 else 0.0

    def draw(self, rng):
        n_t = rng.binomial(int(self.n), self.n_tail / self.n)
        n_mid = rng.binomial(int(self.n) - n_t, self.p_mid) if self.n_body > 0 else 0
        tail = _pl_draws(rng, n_t, self.xmin, self.alpha, self.disc)
        mid = self.mid_x[rng.choice(len(self.mid_x), size=n_mid, p=self.mid_p)] if n_mid else np.array([])
        x, w = _as_hist(np.concatenate([mid, tail]), self.disc)
        return x, w

    def ks_replicate(self, seed):
        rng = np.random.default_rng(seed)
        x, w = self.draw(rng)
        return choose_xmin(x, w, self.disc, n_total=self.n, lo_tail=self.lo_tail, hi_frac=self.hi_frac)["D"]


def fixed_xmin_ks_replicate(args):
    """Parametric bootstrap at a fixed cutoff: draw the tail from the fitted law, refit alpha, return KS."""
    seed, n_t, xmin, alpha, discrete = args
    rng = np.random.default_rng(seed)
    x, w = _as_hist(_pl_draws(rng, n_t, xmin, alpha, discrete), discrete)
    a = fit_alpha(x, w, xmin, discrete)
    return ks_distance(x, w, xmin, discrete, a)


# ------------------------------------------------------------------ alternative models
def lnS_exp(x, u, lam):
    return -lam * (x - u)


def fit_exp(x, w, xmin, discrete):
    u = xmin - 0.5 if discrete else xmin
    def pts(lam):
        return tf._ll_points(lambda t: lnS_exp(t, u, lam), lambda t: np.log(lam) - lam * (t - u), x, u, discrete)
    lam0 = w.sum() / np.dot(w, x - u)
    if discrete:
        r = optimize.minimize_scalar(lambda ll: -np.dot(w, pts(np.exp(ll))), bounds=(np.log(lam0) - 5, np.log(lam0) + 5),
                                     method="bounded")
        lam0 = np.exp(r.x)
    return tf.Fit("EXP", dict(lam=lam0), pts(lam0), w, 1)


def _log_uigamma_mp(a, z):
    """Reference implementation (slow): log Gamma(a, z) via mpmath."""
    return np.array([float(mp.log(mp.gammainc(a, zz))) for zz in np.atleast_1d(z)])


def _log_uigamma(a, z, z_asym=50.0):
    """log of the upper incomplete gamma function Gamma(a, z), any real a < 1 or a > 0, z > 0, vectorised.

    z > z_asym: asymptotic series  Gamma(a,z) ~ z^(a-1) e^(-z) sum_k (a-1)...(a-k) / z^k  (8 terms).
    otherwise:  start from s = a + m in (0, 1] with scipy, recurse down  Gamma(s-1,z) = (Gamma(s,z) - z^(s-1) e^-z)/(s-1).
    Checked against mpmath in csn_selftest.py.
    """
    z = np.atleast_1d(np.asarray(z, float))
    out = np.empty_like(z)
    big = z > z_asym
    if big.any():
        zb = z[big]; term = np.ones_like(zb); tot = np.ones_like(zb)
        for k in range(1, 9):
            term = term * (a - k) / zb
            tot = tot + term
        out[big] = (a - 1) * np.log(zb) - zb + np.log(tot)
    sm = ~big
    if sm.any():
        zs = z[sm]
        if a > 0:
            out[sm] = np.log(special.gammaincc(a, zs)) + special.gammaln(a)
        else:
            m = int(np.ceil(-a)) if abs(a - round(a)) > 1e-12 else int(-round(a))
            s = a + m
            g = special.exp1(zs) if abs(s) < 1e-12 else special.gammaincc(s, zs) * special.gamma(s)
            for _ in range(m):
                g = (g - zs ** (s - 1) * np.exp(-zs)) / (s - 1)
                s -= 1
            out[sm] = np.log(g)
    return out


def fit_tpl(x, w, xmin, discrete, alpha0):
    """Power law with exponential cutoff: density proportional to x^(-d) exp(-lam x), d = density exponent."""
    u = xmin - 0.5 if discrete else xmin
    if discrete:
        lo, hi = x - 0.5, x + 0.5
    def ll_pts(d, lam):
        a = 1 - d
        lnorm = _log_uigamma(a, lam * u)[0]
        if discrete:
            g_lo, g_hi = _log_uigamma(a, lam * lo), _log_uigamma(a, lam * hi)
            return g_lo + np.log1p(-np.exp(np.minimum(g_hi - g_lo, -1e-300))) - lnorm
        return -d * np.log(x) - lam * x + (1 - d) * np.log(lam) - lnorm
    def nll(p):
        d, lam = p[0], np.exp(p[1])
        try:
            v = -np.dot(w, ll_pts(d, lam))
        except (ValueError, ZeroDivisionError):
            return 1e300
        return v if np.isfinite(v) else 1e300
    best = None
    for lam_scale in [1e-3, 1e-2, 1e-1]:
        start = [alpha0 + 1 - 0.2, np.log(lam_scale / xmin)]
        r = optimize.minimize(nll, start, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-6, maxiter=600))
        if best is None or r.fun < best.fun:
            best = r
    d, lam = best.x[0], np.exp(best.x[1])
    return tf.Fit("TPL", dict(d=d, lam=lam, cutoff=1 / lam), ll_pts(d, lam), w, 2)


def nested_lr(f_big, f_small):
    """2*(LL_big - LL_small) with p from a 50:50 mixture of 0 and chi2(1) (parameter on the boundary)."""
    W = max(2 * (f_big.ll - f_small.ll), 0.0)
    return W, 0.5 * stats.chi2.sf(W, 1)


def lr_tests(x, w, xmin, discrete):
    """Likelihood-ratio tests of the power law against each alternative, CSN sign convention:
    positive = power law favoured. Non-nested: normalised Vuong z. Nested (cutoff): raw log-likelihood ratio."""
    xt, wt = tf.tail(x, w, xmin)
    pl = tf.fit_pl(xt, wt, xmin, discrete)
    fits = dict(pl=pl, ln=tf.fit_ln(xt, wt, xmin, discrete), exp=fit_exp(xt, wt, xmin, discrete),
                se=tf.fit_se(xt, wt, xmin, discrete), tpl=fit_tpl(xt, wt, xmin, discrete, pl.params["alpha"]))
    out = {}
    for k in ["ln", "exp", "se"]:
        z, p = tf.vuong(pl, fits[k], wt)
        out[k] = dict(stat=z, p=p)
    W, p = nested_lr(fits["tpl"], pl)
    out["tpl"] = dict(stat=-(fits["tpl"].ll - pl.ll), p=p)   # CSN report LL_pl - LL_cutoff
    Wse, pse = nested_lr(fits["se"], pl)                      # SE contains the power law as c -> 0
    out["se_nested"] = dict(stat=Wse, p=pse)
    return out, fits


def verdict(p_gof, lr):
    """CSN Table 6.2 wording."""
    if p_gof <= 0.1:
        return "none"
    if lr["tpl"]["p"] < 0.1 and lr["tpl"]["stat"] < 0:
        return "with cut-off"
    if all(lr[k]["p"] < 0.1 and lr[k]["stat"] > 0 for k in ["ln", "exp", "se"]):
        return "good"
    return "moderate"
