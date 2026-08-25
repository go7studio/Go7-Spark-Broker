# Go7 Spark Capability Broker

Go7 Spark Capability Broker is a small execution service for NVIDIA DGX Spark
and other user-owned inference hosts. It gives Workhorse, MCP clients, and the
`sparkctl` CLI one authenticated, versioned job and artifact protocol.

This is not a Workhorse installation and it is not a model server. Workhorse is
the control plane that chooses and authorizes work. This broker is the runtime
plane that advertises installed capabilities, rotates mutually exclusive GPU
profiles, journals jobs, and returns verified artifacts. Model servers and 3D
engines remain separate third-party workloads behind the broker.

## Ownership boundary

| Layer | Owns |
| --- | --- |
| Go7 Workhorse | caller grants, host selection, task intent, visible workers, and continuation approval |
| Go7 Spark Capability Broker | authentication, capability discovery, inference routing, fenced resource admission, queues, profile activation, cancellation, restart recovery, and artifact integrity |
| Runtime adapters | translating a typed capability into a local OpenAI-compatible endpoint or an explicitly installed 3D workload |
| Third-party runtimes | model loading, inference, CUDA execution, and model licenses |

The broker does not download models, accept shell commands, or infer local
paths from a request. An administrator installs workloads and names them in a
mode-restricted environment file. See [ARCHITECTURE.md](ARCHITECTURE.md).

## What ships

- Protocol 1.0 asynchronous jobs with trace routes and idempotency keys.
- Typed artifact upload, metadata, SHA-256 verification, byte-range download,
  atomic staging, and bounded storage operations.
- An OS-locked, epoch-fenced host coordinator with durable GPU leases,
  fail-closed restart reconciliation, unified-memory admission hooks, and
  acknowledged lifecycle controls.
- Multi-profile inference routing by advertised model identity, service class,
  administrator priority, measured health, residency, or memory preference.
- A scheduling control loop that continues observing higher-priority work while
  a cooperative long-running adapter executes.
- Dynamic capability discovery. Unconfigured adapters are absent.
- A generic OpenAI-compatible text adapter with a configurable model, profile
  identity, memory estimate, endpoint, and optional Docker container.
- An optional Hunyuan3D adapter with typed GLB/report output and an optional
  Blender preparation continuation. The broker reports the continuation but
  never launches Blender.
- `sparkctl`, an optional standalone `spark-mcp` server, systemd units, and a
  user-level installer.

The default installation exposes only `system.echo`. No model family,
container, workload path, hostname, or remote network is assumed.

Managed resumable training is intentionally not advertised in this release.
Protocol 1.0 cannot truthfully represent a yielded training run, so the broker
fails such a yield instead of reporting false completion. The resource
coordinator can already ask an administrator-installed governor to throttle or
checkpoint-and-release before an inference call. A first-class training
capability requires the protocol and checkpoint work described in
[Training integration](docs/TRAINING-INTEGRATION.md).

## Requirements

- Linux with Python 3.11 or newer.
- `systemd --user`, OpenSSL, and a user session for the recommended installer.
- Docker only for adapters configured to manage Docker workloads.
- A separately installed and licensed model runtime for inference capabilities.

## Install

Clone the repository on the inference host and run:

```bash
./deploy-user.sh
systemctl --user status go7-spark-broker
```

The installer:

- creates an isolated virtual environment under
  `~/.local/share/go7-spark-broker`;
- generates a 256-bit bearer token with mode `0600`;
- creates a minimal configuration only when none exists;
- preserves an existing configuration during upgrades; and
- installs the user service without embedding the repository checkout path.

To install an explicit configuration:

```bash
cp systemd/go7-spark-broker.env.example /tmp/broker.env
# Edit /tmp/broker.env with literal absolute paths and local runtime values.
./deploy-user.sh --config /tmp/broker.env
```

The installer refuses inline broker or model-server secrets in imported
configuration. Use credential files. Run `./deploy-user.sh --no-start` when a
service must be reviewed before first launch.

The system-wide template `systemd/go7-spark-broker@.service` is available for
managed hosts. Its administrator must provide `/etc/go7-spark-broker/USER.env`,
the installation under `/opt/go7-spark-broker`, data-directory ownership, and
any workload-specific systemd path permissions.

## Configure capabilities

Safe core configuration:

```dotenv
SPARK_BROKER_ID=local-capability-host
SPARK_BROKER_BIND=127.0.0.1
SPARK_BROKER_PORT=8790
SPARK_BROKER_TOKEN_FILE=/absolute/path/to/broker-token
SPARK_BROKER_DATA=/absolute/path/to/broker-data
```

`deploy-user.sh` supplies the token and data paths when they are omitted from
an imported configuration. With the shipped user service, keep data beneath
`~/.local/share/go7-spark-broker/` and writable workloads beneath that directory
or `~/workloads/`, unless a narrow systemd `ReadWritePaths` drop-in grants
another location.

Optional single OpenAI-compatible text runtime:

```dotenv
SPARK_OPENAI_ENDPOINT=http://127.0.0.1:8000
SPARK_OPENAI_API_KEY_FILE=/absolute/path/to/runtime-client-key
SPARK_OPENAI_MODEL=your-model-id
SPARK_OPENAI_CONTAINER=your-container-name
SPARK_OPENAI_PROFILE_ID=gpu.local-text
SPARK_OPENAI_DESCRIPTION=Local OpenAI-compatible text generation
SPARK_OPENAI_ESTIMATED_MEMORY_GB=48
```

`SPARK_OPENAI_CONTAINER` is optional. When present, the broker checks and starts
that exact allowlisted Docker container before waiting on the loopback runtime's
`/readyz` endpoint. Legacy `SPARK_TEXT_*` values remain accepted for upgrades.

For multiple inference profiles, copy
[`examples/inference-routes.example.json`](examples/inference-routes.example.json)
outside the checkout and configure:

```dotenv
SPARK_OPENAI_ROUTES_FILE=/absolute/path/to/inference-routes.json
```

The routes file is administrator-owned. It contains loopback model endpoints,
literal container names, model identities, resource envelopes, and
credential-file paths. Callers may select an advertised model identity or a
service class; they can never supply an endpoint, container, executable, or
filesystem path. `/v1/capabilities` publishes safe route metadata so Workhorse,
MCP, and CLI clients can discover what is installed.

To connect a host resource probe or an existing training/inference governor,
copy [`examples/resource-policy.example.json`](examples/resource-policy.example.json)
outside the checkout and configure:

```dotenv
SPARK_RESOURCE_POLICY_FILE=/absolute/path/to/resource-policy.json
```

The governor contract uses a broker lease ID, broker epoch fence, and monotonic
control generation. The broker re-reads the resource snapshot after a control
acknowledgement and refuses the job if an incompatible profile remains. See
[Resource governor protocol](docs/RESOURCE-GOVERNOR.md).

Optional Hunyuan3D adapter:

```dotenv
SPARK_HUNYUAN_ROOT=/absolute/path/to/installed-hunyuan3d-workload
```

The workload itself is not distributed by this repository. Cross-profile
container stop lists are rejected; lifecycle changes belong to the fenced
resource coordinator. Keep writable
directories beneath `~/.local/share/go7-spark-broker/` or `~/workloads/` when
using the shipped user service, or add narrowly scoped `ReadWritePaths` in a
systemd drop-in.

## Connect Workhorse

1. Keep the broker bound to loopback.
2. For a different machine, publish that loopback service through an
   authenticated HTTPS reverse proxy or private overlay network.
3. Place a copy of the broker bearer token in a mode-restricted file on the
   Workhorse machine. Do not put the token in Workhorse state or MCP JSON.
4. In Workhorse, open **Settings → LLMs → Local Compute**, add the HTTPS URL and
   token-file path, then test the host.
5. Grant caller roles, capabilities, and continuation capability/tool pairs.

Workhorse reads `/v1/capabilities` and exposes only capabilities that are both
healthy and granted. It can then upload inputs, submit jobs, poll/cancel them,
receive artifacts, and dispatch separately approved continuations.

## CLI

```bash
export SPARK_BROKER_URL=https://your-private-host.example/broker
export SPARK_BROKER_TOKEN_FILE=/absolute/path/to/broker-token-copy

sparkctl capabilities
sparkctl status
sparkctl chat 'Return exactly ROUTER_OK' --wait --print-output
sparkctl chat 'Summarize this' --model your-small-model-id --service-class interactive --wait
sparkctl upload source.png --kind image --role source_image --media-type image/png
sparkctl generate-3d source.png --target-engine godot --max-faces 100000 --approve-blender --wait
sparkctl job JOB_ID
sparkctl events JOB_ID
sparkctl artifact ARTIFACT_ID
sparkctl download ARTIFACT_ID ./model.glb
sparkctl cancel JOB_ID
```

`sparkctl submit request.json` accepts a complete versioned request, so new
capabilities do not require a new CLI command.

## MCP

`spark-mcp` is an optional direct client adapter. It uses the same URL and
token-file environment and exposes discovery, generic submit, convenience
text/3D calls, model and service-class routing, status, events, cancellation,
and chunked artifact transfer. Full downloads are verified against the
registered artifact digest.
Workhorse does not require `spark-mcp`; Workhorse talks to the broker protocol
directly.

## Development

```bash
python3 -m unittest discover -s tests -v
python3 -m pytest -q
bash -n deploy-user.sh
```

The tests use temporary registries, fake model endpoints, injected executors,
fake governor acknowledgements, and synthetic host pressure. They do not read
a developer's home directory, Docker installation, credentials, or devices.

## Security and publication

See [SECURITY.md](SECURITY.md) before exposing a broker beyond loopback. The
bearer token defines one administrator trust domain; use one broker/service
account per trust boundary. Files
under `~/.config/go7-spark-broker`, runtime data, model checkpoints, generated
artifacts, and machine-specific systemd overrides never belong in this
repository. The checked-in configuration contains placeholders only.

Licensed under the [MIT License](LICENSE).
