#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: ./deploy-user.sh [--config /absolute/path/to/broker.env] [--no-start]" >&2
}

broker_source="$(cd "$(dirname "$0")" && pwd)"
broker_state="$HOME/.local/share/go7-spark-broker"
broker_config="$HOME/.config/go7-spark-broker"
broker_unit_dir="$HOME/.config/systemd/user"
config_source=""
start_service=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      config_source="$2"
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

if [[ -n "$config_source" ]]; then
  [[ "$config_source" = /* && -f "$config_source" && ! -L "$config_source" ]] || {
    echo "--config must name an absolute regular file" >&2
    exit 2
  }
  if grep -Eq '^[[:space:]]*(SPARK_BROKER_TOKEN|SPARK_OPENAI_API_KEY|SPARK_TEXT_API_KEY)[[:space:]]*=' "$config_source"; then
    echo "config files must reference credential files; inline secrets are refused" >&2
    exit 2
  fi
fi

release_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
release_dir="$broker_state/releases/$release_id"
previous_target=""
previous_env="$release_dir/previous.env"
previous_unit="$release_dir/previous.service"
backup_path="$broker_state/backups/broker-$release_id.sqlite3"
previous_was_active=0
previous_was_enabled=0
if [[ -L "$broker_state/current" ]]; then
  previous_target="$(readlink "$broker_state/current")"
fi
if systemctl --user is-active --quiet go7-spark-broker.service; then
  previous_was_active=1
fi
if systemctl --user is-enabled --quiet go7-spark-broker.service; then
  previous_was_enabled=1
fi

mkdir -p "$release_dir" "$broker_state/data" "$broker_state/backups" "$broker_config" "$broker_unit_dir"
python3 -m venv "$release_dir/venv"
"$release_dir/venv/bin/python" -m pip install --disable-pip-version-check --no-deps "$broker_source"
candidate_version="$("$release_dir/venv/bin/python" -c 'from spark_broker import BROKER_VERSION; print(BROKER_VERSION)')"

if [[ "$start_service" -eq 0 ]]; then
  echo "Staged broker release: $release_id"
  echo "Broker version: $candidate_version"
  echo "No config, unit, current pointer, database, or running service was changed."
  exit 0
fi

if [[ ! -s "$broker_config/token" ]]; then
  umask 077
  openssl rand -hex 32 >"$broker_config/token"
fi
chmod 600 "$broker_config/token"

if [[ -f "$broker_config/env" ]]; then
  install -m 600 "$broker_config/env" "$previous_env"
fi
if [[ -f "$broker_unit_dir/go7-spark-broker.service" ]]; then
  install -m 644 "$broker_unit_dir/go7-spark-broker.service" "$previous_unit"
fi
if [[ -f "$broker_state/data/broker.sqlite3" ]]; then
  BROKER_DB="$broker_state/data/broker.sqlite3" BROKER_BACKUP="$backup_path" \
    python3 - <<'PY'
import os
import sqlite3

source = sqlite3.connect(os.environ["BROKER_DB"])
target = sqlite3.connect(os.environ["BROKER_BACKUP"])
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
  systemctl --user stop go7-spark-broker.service
  if [[ -f "$previous_env" ]]; then
    install -m 600 "$previous_env" "$broker_config/env"
  fi
  if [[ -f "$previous_unit" ]]; then
    install -m 644 "$previous_unit" "$broker_unit_dir/go7-spark-broker.service"
  else
    rm -f "$broker_unit_dir/go7-spark-broker.service"
  fi
  # The online backup is an operator recovery artifact, not an automatic
  # rollback source. Rewinding a live database can discard jobs and artifact
  # registrations created after the backup and must never happen implicitly.
  if [[ -n "$previous_target" ]]; then
    ln -sfn "$previous_target" "$broker_state/current.rollback"
    mv -Tf "$broker_state/current.rollback" "$broker_state/current"
  else
    rm -f "$broker_state/current"
  fi
  systemctl --user daemon-reload
  if [[ "$previous_was_enabled" -eq 1 && -f "$broker_unit_dir/go7-spark-broker.service" ]]; then
    systemctl --user enable go7-spark-broker.service
  else
    systemctl --user disable go7-spark-broker.service
  fi
  if [[ "$previous_was_active" -eq 1 && -f "$broker_unit_dir/go7-spark-broker.service" ]]; then
    systemctl --user restart go7-spark-broker.service
  fi
}

on_error() {
  status=$?
  echo "broker deployment failed; restoring the previous release" >&2
  rollback
  exit "$status"
}
trap on_error ERR

if [[ -n "$config_source" ]]; then
  install -m 600 "$config_source" "$broker_config/env"
elif [[ ! -f "$broker_config/env" ]]; then
  install -m 600 /dev/stdin "$broker_config/env" <<ENV
SPARK_BROKER_ID=local-capability-host
SPARK_BROKER_BIND=127.0.0.1
SPARK_BROKER_PORT=8790
SPARK_BROKER_TOKEN_FILE=$broker_config/token
SPARK_BROKER_DATA=$broker_state/data
ENV
fi

if ! grep -q '^SPARK_BROKER_TOKEN_FILE=' "$broker_config/env"; then
  echo "SPARK_BROKER_TOKEN_FILE=$broker_config/token" >>"$broker_config/env"
fi
if ! grep -q '^SPARK_BROKER_DATA=' "$broker_config/env"; then
  echo "SPARK_BROKER_DATA=$broker_state/data" >>"$broker_config/env"
fi
chmod 600 "$broker_config/env"

ln -sfn "releases/$release_id" "$broker_state/current.next"
mv -Tf "$broker_state/current.next" "$broker_state/current"
install -m 644 "$broker_source/systemd/go7-spark-broker.user.service" "$broker_unit_dir/go7-spark-broker.service"
systemctl --user daemon-reload

if [[ "$start_service" -eq 1 ]]; then
  broker_port="$(sed -n 's/^SPARK_BROKER_PORT=//p' "$broker_config/env" | tail -n 1)"
  broker_port="${broker_port:-8790}"
  systemctl --user enable go7-spark-broker.service
  systemctl --user restart go7-spark-broker.service
  ready=0
  for _attempt in $(seq 1 30); do
    if BROKER_URL="http://127.0.0.1:$broker_port" BROKER_TOKEN_FILE="$broker_config/token" \
      EXPECTED_VERSION="$candidate_version" python3 - <<'PY'
import json
import os
import urllib.request

with open(os.environ["BROKER_TOKEN_FILE"], encoding="utf-8") as stream:
    token = stream.read().strip()
with urllib.request.urlopen(os.environ["BROKER_URL"] + "/health/live", timeout=1) as response:
    live = json.load(response)
if live.get("version") != os.environ["EXPECTED_VERSION"]:
    raise SystemExit(1)
request = urllib.request.Request(
    os.environ["BROKER_URL"] + "/health/ready",
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
echo "Broker release: $release_id"
echo "Broker version: $candidate_version"
echo "Broker config: $broker_config/env"
echo "Broker token file: $broker_config/token"
if [[ -f "$backup_path" ]]; then
  echo "Pre-deploy database backup: $backup_path"
fi
