# Backtest journal — daily swing self-assessment

One line per run. Grades the cohort of A/A+ recommendations the scanner would have taken ~25 trading days earlier (portfolio-constrained: <=5 open, <=2/sector), simulated trigger->3R vs ATR stop. Est net % = sum(R) x 1.5% risk/slot. Proposals live in the run logs, NOT here; adopt a change only after it holds across MULTIPLE rows.

**How & when to use this — it's a drift detector, NOT a trade signal.** Each row grades trades from ~5 weeks ago, so nothing here is time-sensitive; don't act on it like a signal.
- **Review weekly / every ~10 rows**, not daily.
- **Act on persistence, never one row.** A single bad row is noise (the sample is 0-5 trades).
- **If an observation/proposal recurs across 5+ rows** (or Win%/Avg R trends down for weeks), turn it into ONE specific tweak, validate it with `backtest_lab.py`, and adopt only if it holds in >=5/7 windows. Never change logic on one day's evidence.
- **Empty cohorts**: in a weak tape = correct (cash); in a strong tape = investigate the A/A+ gate.
- Full per-run analysis + proposals are in `reports/assess_YYYY-MM-DD.log`.

| Run (local) | Cohort as-of | Sel | Trig | Win% | Avg R | Est net % | Notes |
|---|---|---|---|---|---|---|---|
| 2026-06-16 20:45 | 2026-05-12 | 3 | 3 | 100% | +1.10 | +4.9% | 3/3 triggered; best KLAC +2.0R / worst INVH -0.4R |
| 2026-06-17 15:41 | 2026-05-13 | 4 | 3 | 50% | +0.88 | +4.0% | 3/4 triggered; best NTAP +2.0R / worst AMAT -1.0R; 1 never triggered |
| 2026-06-18 15:16 | 2026-05-14 | 5 | 2 | 50% | +0.50 | +1.5% | 2/5 triggered; best NTAP +2.0R / worst AVGO -1.0R; 3 never triggered |
| 2026-06-19 15:16 | 2026-05-14 | 5 | 2 | 50% | +0.50 | +1.5% | 2/5 triggered; best NTAP +2.0R / worst AVGO -1.0R; 3 never triggered |
| 2026-06-22 15:37 | 2026-05-15 | 4 | 2 | 100% | +1.50 | +4.5% | 2/4 triggered; best NTAP +2.0R / worst JBHT +1.0R; 2 never triggered |
| 2026-06-23 15:44 | 2026-05-18 | 4 | 3 | 67% | +1.00 | +4.5% | 3/4 triggered; best AMAT +2.0R / worst AMZN -1.0R; 1 never triggered |
| 2026-06-24 16:22 | 2026-05-19 | 4 | 1 | 100% | +2.00 | +3.0% | 1/4 triggered; best NTAP +2.0R / worst NTAP +2.0R; 3 never triggered |
| 2026-06-25 15:21 | 2026-05-20 | 5 | 4 | 67% | +1.08 | +6.5% | 4/5 triggered; best NTAP +2.0R / worst ADI -1.0R; 1 never triggered |
| 2026-06-26 15:19 | 2026-05-21 | 5 | 4 | 67% | +1.10 | +6.6% | 4/5 triggered; best KLAC +2.0R / worst CAT -1.0R; 1 never triggered |
| 2026-06-29 15:31 | 2026-05-22 | 5 | 1 | 0% | -1.00 | -1.5% | 1/5 triggered; best AMZN -1.0R / worst AMZN -1.0R; 4 never triggered |
| 2026-06-30 15:20 | 2026-05-26 | 5 | 2 | 100% | +2.00 | +6.0% | 2/5 triggered; best AMAT +2.0R / worst AMAT +2.0R; 3 never triggered |
