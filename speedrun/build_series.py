"""Turn data/raw/ into the three committed files the estimate uses.

  data/records.csv       the world-record table as published (track 1: 3.28 validation loss on 8 H100s)
  data/efficiency.csv    A, the speedup over the 45-minute baseline, one row per day with a record
  data/inputs_daily.csv  the two daily research-input series
"""
import os, re
import numpy as np, pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ROW = re.compile(r"^\|?\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(.*?)\s*\|\s*(\d\d/\d\d/\d\d)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|?\s*$")

# ---- records: the first table in the README is track 1 ----
rows, table = [], 0
for line in open(os.path.join(D, "raw", "README.md"), encoding="utf-8").read().split("\n"):
    if line.startswith("| # | Record time"):
        table += 1
        continue
    m = ROW.match(line)
    if m and table == 1:
        num, time_raw, desc, date, _, contrib = m.groups()
        rows.append(dict(record=int(num), date=pd.to_datetime(date, format="%m/%d/%y"),
                         minutes=float(re.search(r"[\d.]+", time_raw).group()),
                         description=re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", desc),
                         retiming="not a new record" in contrib))
rec = pd.DataFrame(rows)

# Chain-link. After record 21 the timing rules changed and record 21 was re-timed twice (2.933 -> 2.997 -> 3.014
# minutes). A re-timing leaves efficiency unchanged; the next record is compared with the re-timed figure.
log_a = [0.0]
for i in range(1, len(rec)):
    log_a.append(log_a[-1] if rec.retiming[i] else log_a[-1] + np.log(rec.minutes[i - 1] / rec.minutes[i]))
rec["speedup"] = np.exp(log_a)
rec.to_csv(os.path.join(D, "records.csv"), index=False, date_format="%Y-%m-%d")

eff = (rec[~rec.retiming].groupby("date", as_index=False)          # one observation per day, as in Erdil et al.
       .agg(record=("record", "max"), minutes=("minutes", "last"), A=("speedup", "last")))
eff.to_csv(os.path.join(D, "efficiency.csv"), index=False, date_format="%Y-%m-%d")

# ---- inputs ----
day = lambda s: pd.to_datetime(s, utc=True).dt.tz_localize(None).dt.normalize()
iss = pd.read_csv(os.path.join(D, "raw", "issues.tsv"), sep="\t", header=None, names=["number", "created", "who", "kind"], quoting=3)
com = pd.read_csv(os.path.join(D, "raw", "commits.tsv"), sep="\t", header=None, names=["sha", "created", "who"], quoting=3)
iss["d"], com["d"] = day(iss.created), day(com.created)
idx = pd.date_range("2024-05-01", max(iss.d.max(), com.d.max(), eff.date.max()), freq="D")

prs = iss[iss.kind == "pr"].groupby("d").size().reindex(idx, fill_value=0)
events = pd.concat([iss[["d", "who"]], com[["d", "who"]]])
out = pd.DataFrame({
    "date": idx,
    "prs_opened": prs.values,
    # pull requests opened per day, mean over the trailing 30 days
    "prs_per_day_30d": prs.rolling(30, min_periods=1).mean().values,
    # people who opened a pull request or issue, or authored a commit, in the trailing 30 days
    "contributors_30d": [events.who[(events.d <= d) & (events.d > d - pd.Timedelta(days=30))].nunique() for d in idx],
})
out.to_csv(os.path.join(D, "inputs_daily.csv"), index=False, date_format="%Y-%m-%d")

print(f"{(~rec.retiming).sum()} records, {len(eff)} days with a record, total speedup {eff.A.iloc[-1]:.1f}x")
print(f"{(iss.kind == 'pr').sum()} pull requests, {len(com)} commits, {events.who.nunique()} distinct contributors")
