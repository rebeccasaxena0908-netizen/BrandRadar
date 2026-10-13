#!/usr/bin/env bash
# Start Brand Radar: the collector in the background and the dashboard in the foreground.
# Ctrl+C stops both. If a collector is already running, its lock makes the new one exit and only
# the dashboard starts. Collector output goes to logs/collector.log.
#
#   scripts/start.sh            # real data
#   BRAND_RADAR_DEMO=1 scripts/start.sh   # demo data: dashboard only (the collector never runs on demo data)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x venv/bin/python ]]; then
  echo "No venv found. Run: python3.11 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
source venv/bin/activate
mkdir -p logs

collector_pid=""
stop() {
  trap - INT TERM EXIT
  if [[ -n "$collector_pid" ]] && kill -0 "$collector_pid" 2>/dev/null; then
    echo "Stopping collector (pid $collector_pid)…"
    # SIGTERM, not SIGINT: a script's background jobs ignore SIGINT (collector.py treats TERM as Ctrl+C)
    kill -TERM "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
}
trap stop INT TERM EXIT

if [[ "${BRAND_RADAR_DEMO:-}" =~ ^(1|true|yes|on)$ ]]; then
  echo "Demo mode: dashboard only, reading the demo database."
else
  python collector.py >> logs/collector.log 2>&1 &
  collector_pid=$!
  echo "Collector started (pid $collector_pid), logging to logs/collector.log"
fi

echo "Dashboard: http://localhost:8501  (Ctrl+C stops everything)"
streamlit run app.py
