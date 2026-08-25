#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: ./deploy-canary-user.sh [--port 8792] [--broker-id local-canary] [--no-start]" >&2
}

canary_source="$(cd "$(dirname "$0")" && pwd)"
canary_state="$HOME/.local/share/go7-spark-broker-canary"
canary_config="$HOME/.config/go7-spark-broker-canary"
canary_unit_dir="$HOME/.config/systemd/user"
canary_port=8792
canary_id="local-canary"
start_service=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ && "$2" -ge 1024 && "$2" -le 65535 ]] || { usage; exit 2; }
      canary_port="$2"
      shift 2
      ;;
    --broker-id)
      [[ $# -ge 2 && "$2" =~ ^[a-z][a-z0-9]*([._-][a-z0-9]+){0,7}$ ]] || { usage; exit 2; }
      canary_id="$2"
      shift 2
      ;;
    --no-start)
      start_service=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

for command in python3 openssl systemctl; do
  command -v "$command" >/dev/null || { echo "$command is required" >&2; exit 1; }
done

release_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
release_dir="$canary_state/releases/$release_id"
previous_target=""
previous_env="$release_dir/previous.env"
previous_unit="$release_dir/previous.service"
backup_path="$canary_state/backups/broker-$release_id.sqlite3"
previous_was_active=0
previous_was_enabled=0
if [[ -L "$canary_state/current" ]]; then
  previous_target="$(readlink "$canary_state/current")"
fi
if systemctl --user is-active --quiet go7-spark-broker-canary.service; then
  previous_was_active=1
fi
if systemctl --user is-enabled --quiet go7-spark-broker-canary.service; then
  previous_was_enabled=1
fi

mkdir -p "$release_dir" "$canary_state/data" "$canary_state/backups" "$canary_config" "$canary_unit_dir"
python3 -m venv "$release_dir/venv"
"$release_dir/venv/bin/python" -m pip install --disable-pip-version-check --no-deps "$canary_source"
candidate_version="$("$release_dir/venv/bin/python" -c 'from spark_broker import BROKER_VERSION; print(BROKER_VERSION)')"

if [[ "$start_service" -eq 0 ]]; then
  echo "Staged canary release: $release_id"
  echo "Canary version: $candidate_version"
  echo "No config, unit, current pointer, database, or running service was changed."
  exit 0
fi

if [[ ! -s "$canary_config/token" ]]; then
  umask 077
  openssl rand -hex 32 >"$canary_config/token"
fi
chmod 600 "$canary_config/token"

if [[ -f "$canary_config/env" ]]; then
  install -m 600 "$canary_config/env" "$previous_env"
fi
if [[ -f "$canary_unit_dir/go7-spark-broker-canary.service" ]]; then
  install -m 644 "$canary_unit_dir/go7-spark-broker-canary.service" "$previous_unit"
fi

if [[ -f "$canary_state/data/broker.sqlite3" ]]; then
  CANARY_DB="$canary_state/data/broker.sqlite3" CANARY_BACKUP="$backup_path" \
    python3 - <<'PY'
import os
import sqlite3

source = sqlite3.connect(os.environ["CANARY_DB"])
target = sqlite3.connect(os.environ["CANARY_BACKUP"])
try:
    source.backup(target)
finally:
    target.close()
    source.close()
PY
fi

rollback() {
  trap - ERR
  set +e
  systemctl --user stop go7-spark-broker-canary.service || true
  if [[ -f "$previous_env" ]]; then
    install -m 600 "$previous_env" "$canary_config/env"
  else
    rm -f "$canary_config/env"
  fi
  if [[ -f "$previous_unit" ]]; then
    install -m 644 "$previous_unit" "$canary_unit_dir/go7-spark-broker-canary.service"
  else
    rm -f "$canary_unit_dir/go7-spark-broker-canary.service"
  fi
  # Preserve the live database across binary rollback. The online backup is
  # retained for explicit operator recovery and is never applied implicitly.
  if [[ -n "$previous_target" ]]; then
    ln -sfn "$previous_target" "$canary_state/current.rollback"
    mv -Tf "$canary_state/current.rollback" "$canary_state/current"
  else
    rm -f "$canary_state/current"
  fi
  systemctl --user daemon-reload || true
  if [[ "$previous_was_enabled" -eq 1 && -f "$canary_unit_dir/go7-spark-broker-canary.service" ]]; then
    systemctl --user enable go7-spark-broker-canary.service || true
  else
    systemctl --user disable go7-spark-broker-canary.service || true
  fi
  if [[ "$previous_was_active" -eq 1 && -n "$previous_target" ]]; then
    systemctl --user restart go7-spark-broker-canary.service || true
  fi
}

on_error() {
  status=$?
  echo "canary deployment failed; restoring the previous canary release" >&2
  rollback
  exit "$status"
}
trap on_error ERR

install -m 600 /dev/stdin "$canary_config/env.next" <<ENV
SPARK_BROKER_ID=$canary_id
SPARK_BROKER_BIND=127.0.0.1
SPARK_BROKER_PORT=$canary_port
SPARK_BROKER_TOKEN_FILE=$canary_config/token
SPARK_BROKER_DATA=$canary_state/data
ENV
mv -f "$canary_config/env.next" "$canary_config/env"

ln -sfn "releases/$release_id" "$canary_state/current.next"
mv -Tf "$canary_state/current.next" "$canary_state/current"
install -m 644 "$canary_source/systemd/go7-spark-broker-canary.user.service" \
  "$canary_unit_dir/go7-spark-broker-canary.service"
systemctl --user daemon-reload

if [[ "$start_service" -eq 1 ]]; then
  systemctl --user enable go7-spark-broker-canary.service
  systemctl --user restart go7-spark-broker-canary.service
  ready=0
  for _attempt in $(seq 1 30); do
    if CANARY_URL="http://127.0.0.1:$canary_port" CANARY_TOKEN_FILE="$canary_config/token" \
      EXPECTED_VERSION="$candidate_version" python3 - <<'PY'
import json
import os
import urllib.request

with open(os.environ["CANARY_TOKEN_FILE"], encoding="utf-8") as stream:
    token = stream.read().strip()
with urllib.request.urlopen(os.environ["CANARY_URL"] + "/health/live", timeout=1) as response:
    live = json.load(response)
if live.get("version") != os.environ["EXPECTED_VERSION"]:
    raise SystemExit(1)
request = urllib.request.Request(
    os.environ["CANARY_URL"] + "/health/ready",
    headers={"Authorization": f"Bearer {token}"},
)
with urllib.request.urlopen(request, timeout=1) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
    then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    false
  fi
fi

trap - ERR
echo "Canary release: $release_id"
echo "Canary version: $candidate_version"
echo "Canary config: $canary_config/env"
echo "Canary token file: $canary_config/token"
echo "Canary URL: http://127.0.0.1:$canary_port"
if [[ -f "$backup_path" ]]; then
  echo "Pre-deploy database backup: $backup_path"
fi
