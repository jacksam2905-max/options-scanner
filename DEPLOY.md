# Deploy the VCP Scanner to the cloud (free, mobile-accessible)

Goal: a password-protected dashboard on an HTTPS URL you (and 2–3 trusted
viewers) can open from any phone, independent of your Mac.

Host used here: **Render.com** (free tier). The free instance sleeps after
~15 min idle; when you open the URL it wakes (~30–60 s cold start) and the page
auto-runs a fresh scan on load, so you see current data within ~15 s.

--------------------------------------------------------------------
## What's already prepared in this repo
- `Dockerfile`              — builds the app image
- `requirements.txt`        — Python deps
- `render.yaml`             — one-click Render blueprint
- `.gitignore`              — keeps your Tradier token OUT of git
- `dashboard/serve.py`      — reads HOST/PORT, basic-auth (DASH_USER/DASH_PASS),
                              shared live-quote cache (QUOTE_TTL), optional
                              server-side scan loop (SCAN_INTERVAL)

⚠️ Your token lives in `tradier_creds.sh`, which is **git-ignored** — it will NOT
be pushed. On the cloud you set the token as an env var instead (step 3).

--------------------------------------------------------------------
## Step 1 — push this folder to a GitHub repo
```
cd "/Users/jacob/Code/VCP tracker"
git init                       # if not already a repo
git add .
git commit -m "VCP scanner — cloud deploy"
# create an EMPTY private repo on github.com first, then:
git remote add origin https://github.com/<you>/vcp-scanner.git
git branch -M main
git push -u origin main
```
Confirm `tradier_creds.sh` is NOT listed by `git status` (it's ignored).

## Step 2 — create the service on Render
1. Go to https://render.com → sign up (free) → **New +** → **Blueprint**.
2. Connect your GitHub and pick the `vcp-scanner` repo. Render reads `render.yaml`.
   (Or: **New + → Web Service → Docker**, point at the repo.)

## Step 3 — set the secret env vars (Render → your service → Environment)
| Key | Value |
|-----|-------|
| `TRADIER_TOKEN` | your Tradier access token |
| `DASH_USER`     | a username, e.g. `jacob` |
| `DASH_PASS`     | a strong password (this is what you share with your viewers) |
| `SCAN_ACCOUNT`  | account size for sizing, e.g. `50000` (optional) |

(`HOST`, `OPTIONS_SOURCE`, `OHLCV_SOURCE`, `TRADIER_BASE`, `QUOTE_TTL` are already
set by the blueprint.)

## Step 4 — deploy & open
- Render builds and gives you `https://vcp-scanner-xxxx.onrender.com`.
- Open it on your phone → enter `DASH_USER` / `DASH_PASS` → you're in.
- Add it to your home screen for an app-like icon.

--------------------------------------------------------------------
## Notes
- **Sharing:** give the URL + `DASH_PASS` to 2–3 people. They all see the same
  scan; the live-quote cache (`QUOTE_TTL=2s`) keeps your Tradier usage flat no
  matter how many watch.
- **Keep it always-fresh even when nobody's looking** (optional): set
  `SCAN_INTERVAL=300`. On the *free* tier the box sleeps when idle so this only
  runs while someone has it open; on a paid/always-on box it keeps data fresh 24/5.
- **Cost:** free tier is fine for you + a couple of viewers. If the cold-start
  delay annoys you, the cheapest always-on upgrade is ~$7/mo (Render Starter).
- **Buying (Phase 2):** Buy is still a dry-run. Do NOT connect Robinhood while
  the password is shared — that would let viewers hit your brokerage account.
- **Update the deployed app:** `git push` → Render auto-redeploys.
