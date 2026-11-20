# NanoGPT speedrun: returns to research effort

Code and data for section E of the [technical appendix](../README.md#e-returns-to-research-in-the-nanogpt-speedrun-figure-8), which has the method, results and limits. This folder fits the stochastic ideas production function of Erdil, Besiroglu and Ho (2024, section 4.3.3) to the record history of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

## Running it

From the repository root, with the dependencies in `requirements.txt` installed:

```bash
bash speedrun/run.sh
```

This takes about three minutes on a laptop and uses every core. It prints the estimates and rewrites `results/`. An optional argument sets the number of bootstrap resamples (500 by default).

To refresh the data from GitHub first:

```bash
python3 speedrun/fetch_github.py
```

```bash
python3 speedrun/build_series.py
```

The fetch makes about 30 requests to the public GitHub API. If it is refused for exceeding the unauthenticated rate limit, set `GITHUB_TOKEN`. Counts keep growing, so the rebuilt series will differ from the committed ones.

## Files

| File | What it is |
|---|---|
| `fetch_github.py` | Downloads the record table, and the lists of pull requests, issues and commits, into `data/raw/` (not committed). |
| `build_series.py` | Builds the three files in `data/` from `data/raw/`. |
| `jones_fit.py` | The model, its likelihood and the optimiser. |
| `estimate_r.py` | Fits the model for each input series, and runs the tests and the bootstrap. |
| `fig8_returns_to_research.py` | Draws the post's Figure 8 from `results/r_estimates.csv` and the published estimates of Erdil et al. |
| `stable_table.npz` | Log-density of the stable distribution on a grid, built by `python3 speedrun/jones_fit.py` (about a minute). The fit interpolates it; `estimate_r.py` also prints the likelihood computed with scipy directly. |

| Data file | Rows | Columns |
|---|---|---|
| `data/records.csv` | One per row of the published record table (92 records and 2 re-timings) | record number, date, minutes, description, whether the row is a re-timing, chain-linked speedup |
| `data/efficiency.csv` | One per day with a record (86) | date, latest record number, minutes, A (speedup over the 45-minute baseline) |
| `data/inputs_daily.csv` | One per day from 1 May 2024 | pull requests opened that day, their trailing 30-day mean, active contributors in the trailing 30 days |

| Results file | Contents |
|---|---|
| `results/r_estimates.csv` | One row per input series: β, λ, r and α with bootstrap intervals, the likelihood-ratio test of λ = 0, the out-of-sample comparison, the naive estimate, and r without the final record. |
| `results/bootstrap_<input>.csv` | β, λ and r for each of the 500 resamples. |
