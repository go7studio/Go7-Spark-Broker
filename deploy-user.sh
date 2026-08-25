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

mkdir -p "$broker_state/data" "$broker_config" "$broker_unit_dir"
python3 -m venv "$broker_state/venv"
"$broker_state/venv/bin/python" -m pip install --disable-pip-version-check --no-deps --upgrade "$broker_source"

if [[ ! -s "$broker_config/token" ]]; then
  umask 077
  openssl rand -hex 32 >"$broker_config/token"
fi
chmod 600 "$broker_config/token"

if [[ -n "$config_source" ]]; then
  [[ "$config_source" = /* && -f "$config_source" && ! -L "$config_source" ]] || {
    echo "--config must name an absolute regular file" >&2
    exit 2
  }
  if grep -Eq '^[[:space:]]*(SPARK_BROKER_TOKEN|SPARK_OPENAI_API_KEY|SPARK_TEXT_API_KEY)[[:space:]]*=' "$config_source"; then
    echo "config files must reference credential files; inline secrets are refused" >&2
    exit 2
  fi
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

install -m 644 "$broker_source/systemd/go7-spark-broker.user.service" "$broker_unit_dir/go7-spark-broker.service"
systemctl --user daemon-reload
if [[ "$start_service" -eq 1 ]]; then
  systemctl --user enable go7-spark-broker.service
  systemctl --user restart go7-spark-broker.service
  systemctl --user --no-pager status go7-spark-broker.service
fi

echo "Broker config: $broker_config/env"
echo "Broker token file: $broker_config/token"
echo "Broker URL: http://127.0.0.1:8790"
