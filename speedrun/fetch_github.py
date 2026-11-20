"""Download the raw inputs from GitHub into data/raw/ (not committed; the derived series in data/ are).

  README.md    the modded-nanogpt README, which holds the world-record table
  issues.tsv   every issue and pull request: number, created_at, login, kind (pr / issue)
  commits.tsv  every commit on the default branch: sha, author date, author

Uses the public GitHub API. Unauthenticated use allows 60 requests an hour, which is enough for one run;
set GITHUB_TOKEN to raise the limit. Counts keep growing, so a fresh fetch will not match the committed series exactly.
"""
import json, os, urllib.request

REPO = "KellerJordan/modded-nanogpt"
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-fetch/1.0", "Accept": "application/vnd.github+json"})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["GITHUB_TOKEN"])
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def pages(path):
    page = 1
    while True:
        sep = "&" if "?" in path else "?"
        batch = json.loads(get(f"https://api.github.com/repos/{REPO}/{path}{sep}per_page=100&page={page}"))
        if not batch:
            return
        for item in batch:
            yield item
        page += 1


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, "README.md"), "w") as f:
        f.write(get(f"https://raw.githubusercontent.com/{REPO}/master/README.md"))
    n = 0
    with open(os.path.join(RAW, "issues.tsv"), "w") as f:
        for it in pages("issues?state=all"):
            f.write("\t".join([str(it["number"]), it["created_at"], (it.get("user") or {}).get("login") or "",
                               "pr" if "pull_request" in it else "issue"]) + "\n"); n += 1
    print(n, "issues and pull requests")
    n = 0
    with open(os.path.join(RAW, "commits.tsv"), "w") as f:
        for c in pages("commits"):
            who = (c.get("author") or {}).get("login") or c["commit"]["author"]["name"]
            f.write("\t".join([c["sha"], c["commit"]["author"]["date"], who.replace("\t", " ")]) + "\n"); n += 1
    print(n, "commits")
