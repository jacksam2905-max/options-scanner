#!/usr/bin/env python3
"""Weekly review + GATED auto-tune (Saturday job).

Reads the daily backtest journal, dedupes to DISTINCT cohorts, summarizes the
trailing window, and tallies recurring proposals. For a proposal that has
persisted across >= PERSIST_MIN distinct cohorts, it validates the matching
candidate against backtest_lab.py. If BOTH gates pass:
    (1) persistence  : >= PERSIST_MIN distinct cohorts
    (2) validation   : the variant beats base avg R by >= MIN_DR
                       in >= WIN_FRAC of the windows it traded
it nudges ONE bounded tuning parameter one step, SMOKE-TESTS the change
(py_compile + a real scan), and — only if the smoke test passes — commits to
main (Render autodeploys). Any failure -> revert, no commit.

SAFETY RAILS (non-negotiable for a cron job that touches production):
  - only bounded numeric params in tuning.json are ever changed (never code)
  - one change per run; 14-day cooldown after any change
  - smoke test must pass or the change is reverted
  - every change appended to tuning_changelog.md and the commit message
  - kill switch: WEEKLY_AUTOAPPLY=0 makes this report-only

Writes weekly_review.md (committed) every run.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
JOURNAL = os.path.join(ROOT, "backtest_journal.md")
REVIEW = os.path.join(ROOT, "weekly_review.md")
CHANGELOG = os.path.join(ROOT, "tuning_changelog.md")
TUNE_PATH = os.path.join(ROOT, "tuning.json")

PERSIST_MIN = int(os.environ.get("WEEKLY_PERSIST_MIN", "5"))   # distinct cohorts
MIN_DR = float(os.environ.get("WEEKLY_MIN_DR", "0.15"))        # min avgR improvement
WIN_FRAC = float(os.environ.get("WEEKLY_WIN_FRAC", "0.6"))     # windows the variant must beat base
COOLDOWN_DAYS = int(os.environ.get("WEEKLY_COOLDOWN_DAYS", "14"))
TRAIL = int(os.environ.get("WEEKLY_TRAIL", "10"))              # distinct cohorts in the summary
AUTOAPPLY = os.environ.get("WEEKLY_AUTOAPPLY", "1") != "0"

TUNE_BOUNDS = {"chase_max_pct": (2.0, 8.0), "vol_confirm_pct": (10.0, 40.0)}

# Proposal -> the bounded param it maps to + the backtest_lab variant that must
# validate it. (vol-confirm is already applied, so only the entry-proximity lever
# is wired for auto-tuning today; add rows here to extend.)
PROPOSALS = [
    {"key": "never_triggered", "needle": "never triggered",
     "param": "chase_max_pct", "delta": -1.0, "variant": "V2 near-pivot",
     "why": "entries sit too far above pivot; require closer-to-pivot (lower chase_max_pct)"},
]


def parse_journal() -> list[dict]:
    rows = []
    try:
        with open(JOURNAL) as fh:
            for ln in fh:
                if not ln.startswith("| 20"):
                    continue
                c = [x.strip() for x in ln.strip().strip("|").split("|")]
                if len(c) < 8:
                    continue
                rows.append({"run": c[0], "cohort": c[1], "sel": c[2], "trig": c[3],
                             "win": c[4], "avgr": c[5], "net": c[6], "notes": c[7]})
    except FileNotFoundError:
        pass
    # dedupe by distinct cohort date — keep the latest run for each cohort
    by_cohort = {}
    for r in rows:
        by_cohort[r["cohort"]] = r
    return sorted(by_cohort.values(), key=lambda r: r["cohort"])


def days_since_last_change() -> float:
    try:
        with open(CHANGELOG) as fh:
            dates = re.findall(r"^\| (\d{4}-\d{2}-\d{2})", fh.read(), re.M)
        if dates:
            last = dt.date.fromisoformat(dates[-1])
            return (dt.date.today() - last).days
    except FileNotFoundError:
        pass
    return 1e9


def run_backtest_lab() -> dict:
    """Run backtest_lab.py and parse the variant table -> {name: (avgR, better, total)}."""
    try:
        p = subprocess.run([PY, "backtest_lab.py"], cwd=ROOT, capture_output=True,
                           text=True, timeout=1200)
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}
    out = {}
    for ln in (p.stdout or "").splitlines():
        m = re.search(r"([+-]\d+\.\d+)\s+(\d+)/(\d+)\s*$", ln)
        if m:
            out[ln.strip()] = (float(m.group(1)), int(m.group(2)), int(m.group(3)))
    return out


def lab_stat(lab: dict, needle: str):
    for line, vals in lab.items():
        if needle.lower() in line.lower():
            return vals
    return None


def smoke_test() -> tuple[bool, str]:
    """Compile + a real (fast) scan must succeed, else the tuning change is bad."""
    c = subprocess.run([PY, "-m", "py_compile", "vcp_tracker.py"], cwd=ROOT,
                       capture_output=True, text=True)
    if c.returncode != 0:
        return False, "py_compile failed: " + (c.stderr or "")[-300:]
    smoke = os.path.join(ROOT, "reports", "smoke_data.js")
    s = subprocess.run([PY, "vcp_tracker.py", "--json", smoke, "--no-sentiment",
                        "--top-options", "0"], cwd=ROOT, capture_output=True,
                       text=True, timeout=600)
    if s.returncode != 0:
        return False, "scan failed: " + (s.stderr or "")[-300:]
    try:
        with open(smoke) as fh:
            txt = fh.read()
        d = json.loads(txt[txt.index("{"):txt.rstrip().rstrip(";").rindex("}") + 1])
        if not d.get("tickers"):
            return False, "scan produced no tickers"
    except Exception as exc:  # noqa: BLE001
        return False, f"smoke output unreadable: {exc}"
    return True, "ok"


def git(*args) -> tuple[int, str]:
    if os.environ.get("WEEKLY_NO_GIT") == "1":            # dry-run / testing
        return 0, "(git skipped: WEEKLY_NO_GIT=1)"
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> int:
    today = dt.date.today().isoformat()
    rows = parse_journal()
    trail = rows[-TRAIL:]
    lines = [f"## Weekly review — {today}", ""]
    if not trail:
        lines.append("No journal rows yet — nothing to review.")
        _write_review(lines)
        return 0

    # trailing summary
    def favg(key):
        vals = []
        for r in trail:
            try:
                vals.append(float(r[key].replace("%", "").replace("R", "").replace("+", "")))
            except ValueError:
                pass
        return sum(vals) / len(vals) if vals else 0.0
    lines += [
        f"- Distinct cohorts reviewed: **{len(trail)}** "
        f"({trail[0]['cohort']} .. {trail[-1]['cohort']})",
        f"- Avg win%: **{favg('win'):.0f}%** · avg R/cohort: **{favg('avgr'):+.2f}** · "
        f"avg est net: **{favg('net'):+.1f}%**",
        "",
    ]

    # persistence tally
    applied = None
    candidates = []
    for prop in PROPOSALS:
        n = sum(1 for r in trail if prop["needle"] in r["notes"].lower())
        line = f"- Proposal `{prop['key']}` — seen in **{n}/{len(trail)}** distinct cohorts"
        if n >= PERSIST_MIN:
            line += f"  ✅ persistence gate met (>= {PERSIST_MIN})"
            candidates.append(prop)
        else:
            line += f"  ⏳ below persistence gate ({PERSIST_MIN})"
        lines.append(line)
    lines.append("")

    if not candidates:
        lines.append("**Verdict: no change warranted** — no proposal crossed the persistence gate. "
                     "(Most weeks should land here.)")
        _write_review(lines)
        _commit_report()
        print("\n".join(lines))
        return 0

    # validation via backtest_lab (one run, reused for all candidates)
    lines.append("### Validation (backtest_lab.py)")
    lab = run_backtest_lab()
    if "_error" in lab:
        lines.append(f"- backtest_lab failed: {lab['_error']} — holding, no change.")
        _write_review(lines); _commit_report(); return 0
    base = lab_stat(lab, "base (current)")
    base_r = base[0] if base else None
    cooldown = days_since_last_change()

    for prop in candidates:
        v = lab_stat(lab, prop["variant"])
        if not v or base_r is None:
            lines.append(f"- `{prop['variant']}` not found in lab output — can't validate; holding.")
            continue
        avgr, better, total = v
        dr = avgr - base_r
        frac = better / total if total else 0.0
        ok = dr >= MIN_DR and frac >= WIN_FRAC
        lines.append(f"- `{prop['variant']}`: avgR {avgr:+.2f} vs base {base_r:+.2f} "
                     f"(Δ{dr:+.2f}R, need ≥{MIN_DR:+.2f}); windows {better}/{total} "
                     f"({frac*100:.0f}%, need ≥{WIN_FRAC*100:.0f}%) -> "
                     + ("✅ VALIDATED" if ok else "❌ not validated"))
        if ok and applied is None:
            if cooldown < COOLDOWN_DAYS:
                lines.append(f"  - validated, but in cooldown ({cooldown:.0f}d < {COOLDOWN_DAYS}d) "
                             "— not applying this week.")
            elif not AUTOAPPLY:
                lines.append("  - validated, but WEEKLY_AUTOAPPLY=0 (report-only) — not applying.")
            else:
                applied = _apply(prop, lines)

    _write_review(lines)
    if applied:
        _commit_change(applied)
    else:
        _commit_report()
    print("\n".join(lines))
    return 0


def _apply(prop: dict, lines: list) -> dict | None:
    """Nudge one bounded param, smoke-test, revert on failure."""
    try:
        with open(TUNE_PATH) as fh:
            tune = json.load(fh)
    except Exception:  # noqa: BLE001
        tune = {}
    param = prop["param"]
    lo, hi = TUNE_BOUNDS[param]
    cur = float(tune.get(param, lo))
    new = min(max(cur + prop["delta"], lo), hi)
    if new == cur:
        lines.append(f"  - `{param}` already at bound ({cur}); nothing to change.")
        return None
    backup = dict(tune)
    tune[param] = new
    with open(TUNE_PATH, "w") as fh:
        json.dump(tune, fh, indent=2)
    ok, msg = smoke_test()
    if not ok:
        with open(TUNE_PATH, "w") as fh:                  # REVERT
            json.dump(backup, fh, indent=2)
        lines.append(f"  - ⚠ smoke test FAILED ({msg}) — reverted `{param}` to {cur}. No change.")
        return None
    # log it
    today = dt.date.today().isoformat()
    row = (f"| {today} | {param} | {cur} → {new} | {prop['variant']} validated | "
           f"{prop['why']} |")
    new_log = not os.path.exists(CHANGELOG)
    with open(CHANGELOG, "a") as fh:
        if new_log:
            fh.write("# Tuning changelog (auto-applied by weekly_review.py)\n\n"
                     "| Date | Param | Change | Trigger | Rationale |\n"
                     "|---|---|---|---|---|\n")
        fh.write(row + "\n")
    lines.append(f"  - ✅ APPLIED: `{param}` {cur} → {new} (smoke test passed). "
                 f"Revert with: edit tuning.json or `git revert`.")
    return {"param": param, "cur": cur, "new": new, "variant": prop["variant"]}


def _write_review(lines: list):
    header = ("# Weekly review log\n\nAuto-generated each Saturday by weekly_review.py. "
              "Newest at the bottom.\n\n") if not os.path.exists(REVIEW) else ""
    with open(REVIEW, "a") as fh:
        fh.write(header + "\n".join(lines) + "\n\n---\n\n")


def _commit_report():
    git("add", "weekly_review.md")
    git("commit", "-m", f"weekly review {dt.date.today().isoformat()} (no change)")


def _commit_change(applied: dict):
    git("add", "tuning.json", "tuning_changelog.md", "weekly_review.md", "dashboard/data.js",
        "dashboard/data.json")
    msg = (f"auto-tune: {applied['param']} {applied['cur']} -> {applied['new']} "
           f"({applied['variant']} validated, smoke-tested)\n\n"
           "Applied by weekly_review.py after persistence + backtest_lab gates. "
           "Reversible via git revert / tuning.json.")
    git("commit", "-m", msg)
    rc, out = git("push", "origin", "main")
    print(f"  push: rc={rc} {out[-200:]}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
