#!/usr/bin/env bash
#
# Refresh NRL predictor data and push to GitHub so the deployed Streamlit app
# picks up the new data. Intended to run weekly via cron (Thursday 9am).
#
#   1. Scrape fixtures/odds + regenerate predictions   (08_update_db.py)
#   2. Fetch the 2026 draw + results                   (10_fetch_draw_2026.py)
#   3. Re-simulate the season                          (11_simulate_season.py)
#   4. Commit data/nrl.db (only if it changed) and push to origin/master
#
set -euo pipefail

# cron runs with a minimal PATH — ensure git and friends are found.
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

REPO_DIR="/Users/nathanbennett/Development/nrl-predictor"
PYTHON="/usr/local/opt/python@3.13/bin/python3"

cd "$REPO_DIR"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') refresh_and_push start ====="

"$PYTHON" 08_update_db.py
"$PYTHON" 10_fetch_draw_2026.py
"$PYTHON" 11_simulate_season.py

if git diff --quiet -- data/nrl.db; then
  echo "No change to data/nrl.db — nothing to commit."
else
  git add data/nrl.db
  git commit -m "chore: regenerate predictions $(date '+%Y-%m-%d')

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  git push origin master
  echo "Pushed updated data/nrl.db."
fi

echo "===== $(date '+%Y-%m-%d %H:%M:%S') refresh_and_push done ====="
