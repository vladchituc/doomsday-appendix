# Technical appendix

Code, data and derivations for **["We were supposed to all die last Friday (and we're not going to die next year)"](https://vladchituc.substack.com/p/we-were-supposed-to-all-die-last)**, Vlad Chituc, *to scale*, November 2026.

This repository regenerates the post's data figures and numbers from the included data, and writes out the derivation behind the stretched-exponential law. The code and this write-up were produced with Claude (Fable 5.2) under the author's direction. Corrections are welcome.

**Contents**

- 1. Where the stretched-exponential law of motion comes from
- 2. Running it
- A. The population figures (Figures 1 and 2)
- B. The Davidson model and what it asks of each AI researcher (Figure 3)
- C. Are idea sizes Pareto? (Figures 4 and 5)
- D. Singularities, and what replaces them (Figure 7)
- E. Returns to research in the NanoGPT speedrun (Figure 8)
- F. Limits of these analyses
- References

---

## 1. Where the stretched-exponential law of motion comes from

The post replaces one line of the growth model: the law of motion for software, which says how fast software improves for a given amount of research. The standard law is a power law. The post uses a stretched exponential, called curved Jones here. Both come from the same model of research, in which research is a search over ideas of random quality. A Pareto distribution of idea quality gives the power law, and a stretched-exponential (Weibull) distribution gives the law the post uses.

### 1.1 Notation

| Symbol | Meaning |
|---|---|
| S | Technology level: the quality of the best idea found so far. S₀ is today's level. |
| K | Cumulative number of tries (ideas drawn so far) |
| R | Research effort. R₀ is today's. |
| λ | Exponent on research effort; λ < 1 means researchers duplicate each other's work |
| F̄(x) | Share of ideas better than x (the survival function of idea quality) |
| θ(x) | Local tail exponent, −d ln F̄ / d ln x: how fast ideas thin out at quality x. Constant for a Pareto. |
| c | Weibull shape. c → 0 is the Pareto. |
| β(S) | Exponent of the ideas production function, −d ln(Ṡ/S) / d ln S at fixed R: how fast progress slows as the level rises. β₀ is today's value. |
| r | Returns to research, λ/β |
| g₀ | Today's pace of progress, Ṡ/S at S₀ and R₀ |

Jones (2023) writes β for the Weibull shape, which is c here. Page and equation numbers for Kortum are those of the working paper (1994).

### 1.2 Research as a search

**Step 1. Research buys tries, and the technology level is the best try so far.** This is Kortum's model (1994, pp. 14–18), which Jones (2023, p. 1998) adopts. With the exponent λ that the later literature adds:

```
dK/dt = R^λ
```

**Step 2. The frontier sits where one try in K beats it.** Jones's Theorem 1 (2023, p. 1999) shows that for any continuous distribution, K·F̄(S) converges to an exponential random variable with mean 1. On the typical path:

```
K · F̄(S) = 1
```

So the chance that the next try beats the frontier is F̄(S) = 1/K, whatever the distribution.

**Step 3. Differentiate.** Taking logs of K·F̄(S) = 1 and differentiating with respect to time gives θ(S)·Ṡ/S = K̇/K, so

```
Ṡ/S = R^λ · F̄(S) · (1/θ(S))
    = research effort × chance a try beats the frontier × average size of the step when it does
```

The step is measured in logs. For a Pareto its mean is exactly 1/θ, whatever the frontier. For a Weibull it is approximately 1/θ(S), which shrinks as S rises.

The exponent of the ideas production function follows:

```
β(S) = −d ln(Ṡ/S) / d ln S = θ(S) + d ln θ(S) / d ln S
```

β is constant only if θ is constant, which means only if the tail is Pareto. Kortum reaches the same conclusion: "only a Pareto search distribution is consistent with the observation that a constant rate of patenting produces exponential growth of productivity" (1994, p. 14).

### 1.3 Pareto ideas give the standard power law

With F̄(x) = (x/x₀)^(−θ), Step 2 gives S = x₀·K^(1/θ), and Step 3 gives

```
Ṡ/S = (x₀^θ / θ) · R^λ · S^(−θ)
```

This is the standard ideas production function, Ṡ/S = ν·R^λ·S^(−β), with β = θ. It is the form in Bloom, Jones, Van Reenen and Webb (2020, eq. 17), and the software line of Davidson, Halperin, Houlden and Korinek (2026, eq. 49). The term S^(−β) is the survival function of a Pareto distribution evaluated at the frontier.

### 1.4 Weibull ideas give the stretched-exponential law

With F̄(x) = exp[−(x/x₀)^c], the local tail exponent is θ(S) = c·(S/x₀)^c. Write θ₀ for its value at today's level. Step 3 then gives the **exact search law**:

```
Ṡ/S = g₀ · (R/R₀)^λ · (S/S₀)^(−c) · exp( −(θ₀/c) · [(S/S₀)^c − 1] )

β(S) = c + θ₀ · (S/S₀)^c
```

The exponential term is the Weibull survival function, F̄(S)/F̄(S₀). The term (S/S₀)^(−c) is the shrinking step, 1/θ(S).

**The law in Figure 7** drops the shrinking step. Bloom et al. (2020, section I) define ideas "in units so that a constant flow of new ideas leads to constant exponential growth in A". Under that convention the step size in Step 3 is a constant, and the law is research effort × the chance that a try beats the frontier:

```
Ṡ/S = ν · R^λ · F̄(S)

Pareto F̄:     Ṡ/S = ν · R^λ · S^(−β)                                          the standard power law
Weibull F̄:    Ṡ/S = g₀ · (R/R₀)^λ · exp( −(β₀/c) · [(S/S₀)^c − 1] )            curved Jones

β(S) = β₀ · (S/S₀)^c          returns   r(S) = λ/β(S) = r₀ · (S/S₀)^(−c)
```

Curved Jones is therefore the standard law with a Weibull survival function in place of the Pareto one:

- At c = 0 it is Davidson et al.'s law exactly.
- Today's pace g₀ and today's returns r₀ are theirs.
- c is the only new parameter.

### 1.5 How far Figure 7's law is from the exact search law

Holding the step fixed is exact for a Pareto and an approximation for a Weibull. `law_comparison.py` measures it by rerunning the full model (section B.1, headline scenario, 40 years) with each law in the software line, both pinned to the same pace and the same β today. The outcome is Figure 7's: the most software progress made in any 12 months, in years of progress at today's pace.

| r today | c | Curved Jones (Figure 7) | Exact search law | Ratio |
|---|---|---|---|---|
| 0.83 | 0.01 | 1.90 | 1.91 | 1.005 |
| 1.0 | 0.02 | 2.94 | 2.97 | 1.011 |
| 1.0 | 0.05 | 1.89 | 1.93 | 1.020 |

Across all 20 cells (`results/law_comparison.csv`) the exact law gives 0.5% to 14% more progress in the fastest year, and neither law is singular in any cell. `search_simulation.py` also simulates the search draw by draw. The exact law stays close to the simulated median, and curved Jones runs slower late in the path (`results/search_simulation.csv`).

### 1.6 Jones (2023): when Weibull ideas still give a power law

Jones (2023) keeps Steps 1 to 3 and the Weibull distribution, and changes how tries accumulate. Research adds ingredients, and the tries are every recipe that can be made from them:

```
dN/dt = α · R^λ · N^φ        N ingredients; φ < 1                    (his eq. 21)
K = 2^N                       every combination of ingredients       (his eq. 22)
```

Step 2 with a Weibull tail then gives

```
S = x₀ · (N · ln 2)^(1/c)
Ṡ/S = (1/c) · Ṅ/N  ∝  R^λ · S^(−c·(1−φ))
```

That is the standard power law, with a constant β = c·(1−φ). In his model the number of tries grows in proportion to the number already made, which exactly offsets the thinning tail. If each try instead costs research effort, as in Step 1, the stretched-exponential law of 1.4 holds.

Jones raises the question himself (2023, pp. 2019–2020): "Is it possible for researchers to evaluate a combinatorially growing number of recipes to find the best one?" Which case holds for AI software research is not settled here.

---

## 2. Running it

You need Python 3.9 or later.

```bash
pip install -r requirements.txt
```

```bash
bash run_all.sh
```

This takes about two minutes on a laptop. It writes every figure to `figures/` and every table to `results/`. Its last step, `scripts/check_numbers.py`, prints numbers quoted in the post next to the values recomputed here. The Figure 7 sweep is cached in `results/singularity_sweep.npz`; delete that file to recompute it (about a minute).

| Post figure | Script | Output |
|---|---|---|
| 1. What von Foerster's curve asks of mothers | `scripts/fig1_children_per_woman.py` | `figures/fig1_children_per_woman.png`, `results/children_per_woman_required.csv` |
| 2. If the world had halved its birth rate in 1960 | `scripts/fig2_births_halved.py` | `figures/fig2_births_halved.png`, `results/births_halved_1960.csv` |
| 3. What the growth model asks of each AI researcher | `scripts/davidson_model.py`, `scripts/fig3_compute_per_researcher.py` | `figures/fig3_compute_per_researcher.png`, `results/compute_per_researcher.csv` |
| 4. Idea-size tails against a Pareto | `scripts/fig4_tail_fits.py` | `figures/fig4_tail_fits.png`, `results/tail_tests_top1pct.csv` |
| 5. The exponent above each cutoff | `scripts/fig5_rising_beta.py` | `figures/fig5_rising_beta.png`, `results/rising_beta_points.csv` |
| 6. How multiplying factors gives a stretched exponential | `scripts/fig6_mechanism.py` | `figures/fig6_mechanism.png`. The post's version is an illustration; this is a plain chart of the same arithmetic. |
| 7. Software singularity with one assumption changed | `scripts/fig7_singularity_map.py` | `figures/fig7_singularity_map.png`, `results/singularity_sweep.csv` |
| 8. Returns to research, measured on AI itself | `speedrun/fig8_returns_to_research.py` | `figures/fig8_returns_to_research.png` |

`run_all.sh` draws Figure 8 from the committed speedrun estimates. To refit them (about three minutes):

```bash
bash speedrun/run.sh
```

The Clauset–Shalizi–Newman goodness-of-fit bootstraps are optional and take ten to twenty minutes:

```bash
python3 scripts/csn_tests.py 500
```

**Data:**

| File | Source |
|---|---|
| `data/population/*.csv` | UN World Population Prospects 2024, via Our World in Data |
| `data/derived/kpss_value_2000_2019.npy` | Kogan, Papanikolaou, Seru and Stoffman (2017), extended data: real value per patent, 1982 $ million |
| `data/derived/patentsview_examiner_cites_2002_2011.csv` | PatentsView final release (December 2024): histogram of examiner-added citations received |
| `data/derived/openalex_all_2000_2009.csv` | OpenAlex: histogram of citations received by journal articles published 2000–09 |
| `speedrun/data/*.csv` | The modded-nanogpt record table and the GitHub API's lists of that repository's pull requests, issues and commits, fetched 18 November 2026 |

OpenAlex and GitHub counts keep growing, so a fresh fetch will not match exactly.

---

## A. The population figures

Von Foerster, Mora and Amiot (1960) fitted 24 estimates of world population with

```
N(t) = 1.79 × 10¹¹ / (2026.87 − t)^0.99
```

The denominator reaches zero at t = 2026.87, which is 13 November 2026. Growth per person is 0.99/(2026.87 − t), so the growth rate rises roughly in proportion to N itself.

### A.1 Figure 1: children per woman required

The calculation takes the case most favourable to the curve. From 1960 on, nobody dies, so every added person is a birth.

```
births needed per year     B(t) = dN/dt
women aged 15–49           W(t) = ½ × (everyone born between t − 50 and t − 15)
children per woman         TFR(t) = 35 × B(t) / W(t)        (35 childbearing years)
```

Cohorts born before 1960 come from the UN's 1960 world age structure, scaled to the curve's 1960 population. Later cohorts come from the curve itself.

| Year | Population (bn) | Births needed (bn/yr) | Women 15–49 (bn) | Children per woman |
|---|---|---|---|---|
| 1960 | 2.79 | 0.041 | 0.66 | 2.2 |
| 2003 | 7.74 | 0.32 | 1.26 | 8.9 (passes the record: Yemen, 1985, 8.86) |
| 2010 | 10.9 | 0.64 | 1.51 | 14.8 |
| 2018 | 20.6 | 2.30 | 2.29 | 35.2 |

In 2010 the curve's growth rate passes 6% a year, Coale's (1961) ceiling for an immortal population at record fertility.

### A.2 Figure 2: the counterfactual

The UN's world population by five-year age group, 1960–2023, is rerun with one change: births in every year are half what they would have been.

- Births are actual fertility × ½ × the counterfactual's potential mothers.
- Mothers are weighted by a fixed age pattern of childbearing: 9%, 27%, 27%, 19%, 11%, 5% and 2% of births for ages 15–19 through 45–49.
- Survival at every age is kept as observed.

| 2023 | Population | Share aged 65+ |
|---|---|---|
| Actual | 8.09 bn | 10.0% |
| Fertility half its actual value every year (the orange line) | 3.25 bn (40%) | 24.9% (2.5×) |

The scenario is mechanical: it has no feedback from population to mortality, income or fertility.

---

## B. The Davidson model and what it asks of each AI researcher

### B.1 The model

`davidson_model.py` implements equations 47–55 of Davidson, Halperin, Houlden and Korinek (2026, NBER Working Paper 35155, "DHHK") with the calibration of their section 5.3. The state is five logs, each relative to the moment of automation: software s, hardware quality h, general productivity a, goods capital k_Y and compute capital k_H.

The model has four sectors i ∈ {S, H, A, Y}: software research, hardware research, general research and goods production. A share f_i of each sector's tasks is automated, and automated work is done by AI labour, which is compute × software.

```
log effective labour in sector i:   L_i(t) = (1 − f_i)·n_i·t + f_i·(k_H + s)

ds/dt = g_S · exp(λ·L_S − β_S·s)          the ideas production function for software
dh/dt = g_H · exp(λ·L_H − β_H·h)
da/dt = g_A · exp(λ·L_A − β_A·a)
y     = a + (1 − α)·L_Y + α·k_Y            log output
dk_Y/dt = (g_Y0 + δ_Y)·exp(y − k_Y) − δ_Y
dk_H/dt = (g_KH0 + δ_H)·exp(h + y − k_H) − δ_H
```

g_Y0 = g_A/(1 − α) + n_Y and g_KH0 = g_H + g_Y0 are the growth rates of output and compute capital before automation.

| Parameter | Value |
|---|---|
| α | 0.4 |
| λ | 1 |
| Returns to research r_S, r_H, r_A | 1, 5, 0.32 (β_i = 1/r_i) |
| Today's pace of progress g_S, g_H | 1.10, 0.37 log points a year |
| g_A | r_A·n_A |
| Labour growth n_S, n_H, n_A, n_Y | 1.10, 0.074, 0.04, 0.01 |
| Depreciation δ_Y, δ_H | 0.07, 0.20 |

**Headline scenario:** f_S = 1, and f_H = f_A = f_Y = 0.05. Software research is fully automated, and 5% of everything else is too.

DHHK approximate the singularity "by testing when growth in the system exceeds 200% per year". Here that happens at 5.46 years in the headline scenario; the paper reports a singularity "within six years". This is a re-implementation, and the authors' own code was not run.

### B.2 Figure 3: compute per AI researcher

Two assumptions set the units, and nothing else depends on them:

- N₀ = 20,000 AI researchers today.
- C₀ = 10¹⁶ operations a second for each.

The number of AI researchers is N(t) = N₀·e^(k_H + s), so the compute per researcher is

```
C₀ · N₀ · e^(k_H) / N(t) = C₀ · e^(−s)
```

That is today's brain-equivalent divided by the software efficiency gain. A different C₀ shifts every point by the same factor.

| World output growth | Year | Software gain | AI researchers (chips alone) | Ops/s each | Reference machine (within about 2×) |
|---|---|---|---|---|---|
| 30% | 3.8 | 8 × 10⁴ | 1.1 × 10¹⁰ (1.3 × 10⁵) | 1.2 × 10¹¹ | Xbox 360 (2005), about 2.4 × 10¹¹ |
| 50% | 4.5 | 8 × 10⁷ | 1.9 × 10¹³ (2.4 × 10⁵) | 1.3 × 10⁸ | Pentium PC running Windows 95, about 10⁸ |
| 100% | 5.1 | 9 × 10¹² | 4.6 × 10¹⁸ (4.9 × 10⁵) | 1.1 × 10³ | Manchester Baby (1948), about 1.1 × 10³ |
| 200% (threshold) | 5.5 | 10¹⁹ | 1.0 × 10²⁵ (1.0 × 10⁶) | 1.0 × 10⁻³ | one operation every 16 minutes |

The reference machine speeds are round figures.

---

## C. Are idea sizes Pareto?

### C.1 Data

Idea quality is not observed, so the standard proxies are used. For citation data, "top 1%" means the top 1% of items cited at least once.

| Proxy | Items | Top 1% begins at |
|---|---|---|
| Patent market value, US patents issued 2000–19 | 1,812,340 | $126 million (1982 dollars) |
| Examiner citations received, utility patents granted 2002–11 | 1,348,259 cited | 31 citations |
| Citations received, journal articles published 2000–09 | 14,979,398 cited | 418 citations |

### C.2 The two models

Both models are fitted conditional on x ≥ u, the start of the top 1%:

```
Pareto:                   ln S(x) = −θ · ln(x/u)
stretched exponential:    ln S(x) = −(b/c) · [(x/u)^c − 1],        i.e. S(x) ∝ exp[−(x/x₀)^c]
```

Here S(x) is the share of the tail at least as large as x. As c → 0 the second model becomes the first, with θ = b, so the Pareto is a special case. The local exponent θ(x) = −d ln S / d ln x is constant for the Pareto and equals b·(x/u)^c for the stretched exponential.

Citation counts are whole numbers, so they use a binned likelihood: Pr(X = k) = S(k − ½) − S(k + ½). Patent values are continuous and use the density.

### C.3 The tests in Figure 4

- **Nested likelihood-ratio test.** LR = 2·(lnL_SE − lnL_Pareto). The value c = 0 sits on the edge of the allowed range, so LR is asymptotically a 50:50 mixture of 0 and χ²(1), and p = ½·Pr(χ²₁ > LR) (Self and Liang 1987; Malevergne, Pisarenko and Sornette 2005).
- **Vuong's normalised likelihood ratio,** as used by Clauset, Shalizi and Newman (2009, "CSN"). Write ℓᵢ for each item's log-likelihood difference, stretched minus Pareto. Then z = Σℓᵢ / √(n·Var ℓ), with a two-sided normal p-value; a positive z favours the stretched exponential.
- **CSN's goodness-of-fit bootstrap** for the Pareto, with the threshold held at the top 1% (`csn_tests.py`).

| Top 1% | n | c | LR | p (nested) | Vuong z | p (Vuong) | CSN goodness of fit |
|---|---|---|---|---|---|---|---|
| Patent values | 18,124 | 0.32 | 338.8 | < .001 | +10.41 | < .001 | p ≤ .002 |
| Patent citations | 12,755 | 0.17 | 38.0 | < .001 | +3.35 | < .001 | p ≤ .002 |
| Paper citations | 149,560 | 0.04 | 67.0 | < .001 | +4.17 | < .001 | p ≤ .002 |

The R² values in the figure's legend are computed on log share against fitted log share. They describe how closely each curve follows the points; they are not a test.

### C.4 Figure 5: the exponent above each cutoff

Each dot fits a power law to all items above one cutoff, using CSN's estimators: Hill's estimator for continuous data, θ̂ = n / Σ ln(xᵢ/u), and the discrete maximum-likelihood estimator for counts. Cutoffs run from the top 1% down to the point where 100 items remain. The band is ±1.96 standard errors. The dots are not independent, because each tail contains the next.

| | Exponent at the top 1% | With 100 items left |
|---|---|---|
| Patent values | 2.00 | 3.50 |
| Patent citations | 2.90 | 3.86 |
| Paper citations | 1.79 | 2.01 |

**This exponent is θ, not the growth model's β.** θ describes a cross-section: how fast ideas thin out as you look at bigger ones. β describes a time path: how fast progress slows as technology improves. The search model of section 1 connects them: as the frontier moves up, β takes the value of the tail exponent further out, β(S) = θ(S) under curved Jones. So "the further out you look, the higher θ" becomes "as technology improves, β rises". The link holds only if tries cost research effort (1.6).

**The proxies are not in the model's units.** If true quality is a power of the proxy, q = x^k, both the Weibull shape and the Pareto exponent rescale by 1/k. Zero stays zero. So whether the exponent rises carries over to the model, but how fast does not: Figure 4's values of c are not Figure 7's.

This holds only for power transforms. If quality were instead exponential in the proxy, q = exp(x^k), a stretched-exponential tail in the proxy would be a Pareto tail in quality. The analysis assumes dollars and citation counts are level measures of an idea's size and not log measures. Stockfish patch gains are left out for this reason: Elo is a log scale, so a rising exponent in Elo units is what a constant β predicts.

### C.5 What the tests support

- **Supported:** the observed tails bend away from a Pareto at the top 1%.
- **Not established: that the stretched exponential is the right alternative.** A lognormal does about as well against the Pareto. The result in section D needs β to keep rising without limit (D.2). Any tail thinner than every power law gives that. A tail whose exponent rises and then levels off does not.
- **Not established: the far tail.** The tests depend on where the tail is taken to start. `results/c_by_tail_depth.csv` refits c at 18 depths from the top 10% to the top 0.05%, each with a 95% profile-likelihood interval (`figures/figA1_c_by_tail_depth.png`). The interval excludes 0 at every depth down to the top 0.26%. Deeper than that the samples are small, and most intervals from the top 0.1% down reach 0. For paper citations c falls steadily with depth, from 0.15 at the top 10% to about 0.01–0.02 at the top 0.1%, so that tail is close to Pareto at the very top. A stretched exponential has the same c at every depth, so a falling c is what an exponent that levels off would look like. Paper citations are the longest of the three tails, and the frontier in the search model sits in the far tail.

---

## D. Singularities, and what replaces them

### D.1 The thresholds under the power law

**Software alone.** With compute held fixed and f_S = 1, research effort rises one-for-one with software. With λ = 1 and β = 1/r:

```
ds/dt = g₀ · exp((1 − 1/r) · s)
```

This reaches infinity in finite time if and only if r > 1.

**The full model.** DHHK's explosion condition is

```
f_Y + f_S·r_S + f_H·r_H + f_A·r_A / (1 − α) > 1
```

In the headline scenario this is 0.05 + r_S + 0.25 + 0.027 > 1. So the full model is singular for every r_S above 0.673, not 1: hardware and general-productivity feedback make up the rest. The hatched strip in Figure 7 is drawn from this threshold.

### D.2 Curved Jones: no singularity for any c > 0

With the curved Jones law from section 1.4, in the software-only loop:

```
ds/dt = g₀ · exp(φ(s)),        φ(s) = s − (1/r) · (e^(c·s) − 1) / c
```

The time to reach level s is T(s) = (1/g₀)·∫₀^s e^(−φ(u)) du. Because e^(c·s) eventually outgrows s, φ(u) → −∞, so the integrand grows without bound and T(∞) = ∞. This holds for every c > 0 and every r.

If r > 1, the pace first rises and then peaks:

```
peak at s* = ln(r)/c                     (software has grown by a factor r^(1/c))
peak pace = exp([ln r − 1 + 1/r] / c) × today's pace
```

The same argument covers the exact search law, where φ(s) = s − c·s − (1/r − c)·(e^(c·s) − 1)/c, as long as c < 1/r. Every cell of Figure 7's grid satisfies that.

**What the result needs.** The argument uses only that β(S) rises without limit, so it holds for any idea distribution thinner than every power law (a lognormal, for example). It does not hold for every departure from a pure power law. If β rises and then levels off at β∞, the law is a power law again at high S, and the software loop is singular whenever λ/β∞ > 1.

**The full model.** The proof above is for the software loop alone. For the full model the result is numerical (D.3). The reason it carries over: once r(S) = r₀·(S/S₀)^(−c) falls below the threshold of D.1 (0.673 in the headline scenario), the explosion condition fails.

### D.3 The sweep in Figure 7

- **Model:** B.1, with the software law replaced by curved Jones. Everything else is unchanged.
- **Grid:** r from 0.50 to 1.50 in steps of 0.01 (101 values) × c from 0.01 to 0.1 on a log scale (48 values). The power law is run at every r. In total that is 4,949 runs of 40 years.
- **Colour:** the most software progress made in any 12 months, measured in years of progress at today's pace (1.10 log points a year). A value of 3 means the fastest year packs in three years of progress at today's rate.
- **Integration:** runs with c > 0 are integrated in a stretched clock, dσ = (1 + ṡ)·dt, so a burst of any size can be passed through and measured.

**Result:** under the power law, every run with r ≥ 0.72 diverges within the 40 years; between 0.673 and 0.72 the divergence takes longer than that. None of the 4,848 runs with c > 0 is singular within the 40 years.

| r today | c = 0 | 0.01 | 0.02 | 0.03 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|
| 0.83 (chess engine) | singular | 1.90 | 1.42 | 1.29 | 1.18 | 1.09 |
| 1.0 (DHHK) | singular | 4.57 | 2.94 | 2.38 | 1.89 | 1.48 |

At r = 1, the fastest year falls below 3 years of progress at c = 0.019. At r = 0.9 it runs from 2.61 years of progress (c = 0.01) down to 1.22 (c = 0.1).

### D.4 The range of c

There is no estimate of c for AI software research, so Figure 7's axis, 0.01 to 0.1, is set by how much bend a power law could hide. Under curved Jones β(S) = β₀·(S/S₀)^c, so β doubles each time software improves by log₁₀2/c ≈ 0.3/c orders of magnitude:

| c | Software improvement over which β doubles |
|---|---|
| 0.01 | 30 orders of magnitude |
| 0.03 | 10 orders of magnitude |
| 0.1 | 3 orders of magnitude |

**The floor, c = 0.01, is a very weak bend.** Thirty orders of magnitude is the ratio of the width of the observable universe (about 10²⁷ m) to a grain of sand (about 10⁻³ m). At c = 0.01, a law followed across that whole span would see its exponent rise only from β₀ to 2β₀:

| Span | β at the far end | R² of a straight line fitted to the log survival curve |
|---|---|---|
| 3 orders of magnitude | 1.07 × β₀ | 0.9999 |
| 5 orders of magnitude | 1.12 × β₀ | 0.9998 |
| 20 orders of magnitude | 1.58 × β₀ | 0.996 |
| 30 orders of magnitude | 2.00 × β₀ | 0.992 |

So over the three to five orders of magnitude that a data set typically covers, c = 0.01 cannot be told from a power law, and even over all thirty a straight line fits with R² above 0.99. For comparison, the Pareto fits that Figure 4 rejects have R² of 0.956 to 0.995 over less than two orders of magnitude. A smaller c would require the exponent to stay closer to constant than this over a span no empirical law has been observed across, so smaller values are not run.

**The ceiling, c = 0.1,** doubles β within three orders of magnitude. Over that span it is exactly as straight as c = 0.01 is over thirty (R² 0.992), which is about the most bend that could go unnoticed in a series of that length.

c = 0 is the power law itself, shown in the strip on the left of Figure 7. The headline run of the model covers 19 orders of magnitude of software progress (B.2).

---

## E. Returns to research in the NanoGPT speedrun (Figure 8)

Of the published estimates of r for software, only Erdil, Besiroglu and Ho's (2024) Stockfish estimate is fitted to a long series of dated efficiency gains with a measure of research input alongside. The NanoGPT speedrun supplies a second series of that kind, and its subject is the training of a language model. The code and data are in `speedrun/`.

**The task.** The modded-nanogpt repository (Jordan and contributors) keeps a public record table for one task: train a language model to a validation loss of 3.28 on the FineWeb dataset, on eight NVIDIA H100 GPUs, in as little wall-clock time as possible. The baseline is Karpathy's reproduction of GPT-2 small, which takes 45 minutes.

**Efficiency, A,** is the speedup over the 45-minute baseline.

- The table lists 92 records from 28 May 2024 to 30 August 2026, falling from 45 minutes to 0.665 minutes.
- The timing rules changed after record 21, which was re-timed under the new rules (2.933 minutes became 3.014). Later records are compared with the re-timed figure, so the series is chain-linked. The total speedup is 69.5×.
- Records set on the same day are merged, as Erdil et al. do for Stockfish.
- The fit starts at record 3 (4 October 2024, 24.9 minutes), when pull requests start arriving regularly: one was opened in June 2024 and the next on 5 October. From there A rises from 1.81 to 69.5 over 1.90 years, in 83 steps.

**Research input, I.** Stockfish has a direct count of experiments. The speedrun does not, so two proxies are built from the repository's public activity, each measured daily:

- **Pull requests opened per day,** averaged over the trailing 30 days. A pull request is a submitted attempt at a record, whether or not it was accepted. There are 258 in the window. Six days with none in the trailing 30 are given half the smallest positive value.
- **Active contributors:** distinct accounts that opened a pull request or an issue, or authored a commit, in the trailing 30 days. It averages 12 and ranges from 1 to 45.

**The model** is equation 26 of Erdil et al. (their section 4.3.3), the one they fit to Stockfish. For two successive observations A₁ and A₂:

```
(A₂^β − A₁^β) / β  ~  Stable(α, skew 1, location μ·J, scale σ·A₁^(β − β/α)·J^(1/α))
J = ∫ I(t)^λ dt   over the interval between the two observations
```

Without noise this is the ideas production function integrated between the two dates, A₂^β − A₁^β = β·μ·J, with returns r = λ/β. The five parameters (β, λ, α, μ, σ) are fitted by maximum likelihood. Intervals are the 2.5th and 97.5th percentiles over 500 bootstrap resamples of the 83 steps.

**Results.** The Stockfish column is Table 4 and section 5.2 of Erdil et al.

| | Stockfish (Erdil et al.) | Speedrun, pull requests | Speedrun, active contributors |
|---|---|---|---|
| Steps | 255 | 83 | 83 |
| β | 0.476 (SE 0.066) | 1.59 (95%: 0.96 to 2.92) | 1.75 (0.92 to 2.50) |
| λ | 0.392 (SE 0.079) | 0.77 (0.09 to 1.26) | 1.25 (0.36 to 1.84) |
| **r = λ/β** | **0.825 (SE 0.15)** | **0.48 (SE 0.16)** | **0.72 (SE 0.19)** |
| 95% interval for r | not reported | 0.06 to 0.69 | 0.30 to 1.03 |
| Resamples with r ≥ 1 | not reported | 0 of 500 | 18 of 500 |
| Test of λ = 0 | p = .004 | LR = 13.2, p < .001 | LR = 19.9, p < .001 |
| Out-of-sample gain over λ = 0 | +2.79 nats | +3.09 nats | −1.21 nats |
| Naive estimate of r | 2.4 | 2.04 | 2.83 |

- **Out of sample:** both models are fitted to the first 80% of steps and scored on the last 17. Pull requests predict the held-out steps better than a model with no input; active contributors predict them worse.
- **The two estimates are compatible.** Their 95% intervals overlap, and 0.48 lies inside the contributor interval. 0.72 lies just above the pull-request interval, which ends at 0.69.
- **Against the thresholds of D.1:** the pull-request estimate is below both 1 and 0.673. The contributor estimate is below 1 and above 0.673.
- **The naive estimate** is the growth rate of A divided by the growth rate of I. As Erdil et al. find for Stockfish, it overstates r several times over.
- **The final record** cut the time from 1.126 to 0.665 minutes, the largest single step. Dropping it gives r = 0.47 and 0.69.

**Limits:**

- **The input is a proxy.** Neither series counts experiments.
- **The series is short:** a third as many steps as Stockfish, over 1.9 years against ten.
- **Input may respond to progress.** A promising stretch can attract contributors, and nothing here corrects for that.

The four other estimates in Figure 8 are from Erdil et al.'s Table 8 (posterior medians and 90% intervals), where input is the number of unique paper authors and each domain contributes a single data point.

---

## F. Limits of these analyses

- **Rejecting a Pareto tail does not establish a stretched exponential** (C.5). The model result needs β to keep rising without limit; a tail whose exponent levels off keeps the singularity (D.2). The paper-citation tail is close to Pareto at the very top.
- **The proxies are treated as level measures of idea size.** If quality is exponential in the proxy, a stretched-exponential tail in the proxy is a Pareto tail in quality (C.4).
- **No singularity is proved for the software loop alone.** For the full model it is the result of 4,848 runs of 40 years (D.2, D.3).
- **There is no measured value of c for AI software** (D.4). Figure 4's values are in proxy units (C.4).
- **The link from idea sizes to β assumes each try costs research effort.** Jones's combinatorial model is the main way this argument could fail (1.6).
- **Figure 7's law holds the size of a step fixed.** The exact search law gives 0.5% to 14% more progress in the fastest year (1.5).
- **Only the software law is changed, and only in DHHK's headline scenario.**
- **The growth model is a re-implementation,** checked against the paper's reported dates and not against the authors' code.
- **Figures 1 and 2 are mechanical.** They have no behavioural feedback.
- **The speedrun estimates of r rest on proxies for research input,** over 1.9 years of data (E).

---

## References

- Bloom, N., Jones, C. I., Van Reenen, J. and Webb, M. (2020). Are ideas getting harder to find? *American Economic Review* 110(4), 1104–1144.
- Clauset, A., Shalizi, C. R. and Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review* 51(4), 661–703.
- Coale, A. J. (1961). Population density and growth (letter). *Science* 133.
- Davidson, T., Halperin, B., Houlden, T. and Korinek, A. (2026). When does automating AI research produce explosive growth? Feedback loops in innovation networks. NBER Working Paper 35155.
- Erdil, E., Besiroglu, T. and Ho, A. (2024). Estimating idea production: a methodological survey. arXiv:2405.10494.
- Jordan, K. and contributors (2024–26). modded-nanogpt: NanoGPT speedrun record history. github.com/KellerJordan/modded-nanogpt.
- Jones, C. I. (2023). Recipes and economic growth: a combinatorial march down an exponential tail. *Journal of Political Economy* 131(8), 1994–2031.
- Kogan, L., Papanikolaou, D., Seru, A. and Stoffman, N. (2017). Technological innovation, resource allocation, and growth. *Quarterly Journal of Economics* 132(2), 665–712.
- Kortum, S. (1994). A model of research, patenting, and productivity growth. NBER Working Paper 4646. Published as Kortum (1997), *Econometrica* 65(6), 1389–1419.
- Malevergne, Y., Pisarenko, V. and Sornette, D. (2005). Empirical distributions of stock returns: between the stretched exponential and the power law? *Quantitative Finance* 5(4), 379–401.
- Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *Journal of the American Statistical Association* 82(398), 605–610.
- von Foerster, H., Mora, P. M. and Amiot, L. W. (1960). Doomsday: Friday, 13 November, A.D. 2026. *Science* 132, 1291–1295.
