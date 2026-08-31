# Read-only Spark resource probe

`spark-resource-probe` is the broker's optional, generic GPU inventory service.
It exposes one authenticated loopback endpoint:

```text
GET /v1/resource-snapshot
```

It has no process-control, container-control, training-control, or model-load
endpoint. The broker and separately reviewed controllers own lifecycle
mutations. The probe only observes installed runtimes and fails closed when it
cannot prove who owns a CUDA consumer.

## Safety boundary

The probe:

- unions `nvidia-smi` compute-process and `pmon` process inventories so both
  compute and graphics contexts are classified, retaining per-process memory
  when the driver exposes it;
- reconciles the summed per-process memory against `memory.used` when the
  driver exposes a device total; a residual beyond the documented 64 MiB
  rounding/driver tolerance becomes a synthetic unknown consumer;
- binds Docker consumers through their Linux cgroup container ID and verifies
  the running container's immutable image ID;
- binds systemd user-service consumers through their control-group path and
  verifies both the unit's main process and every matched GPU process against
  the pinned SHA-256 executable identity behind `/proc/PID/exe`;
- reports every unmatched, ambiguous, vanished, or unobservable CUDA PID in
  `unknownConsumers`;
- reports missing GPU enumeration as a synthetic unknown consumer;
- reports typed DGX unified-memory and memory-PSI measurements; and
- optionally measures the bytes available to a new short-lived CUDA process in a
  short-lived child process, so the observer does not retain a GPU context; and
- reads controller-published, mode-restricted mutation-state files so the
  broker can causally bind acknowledgements to observed inventory; and
- advances an atomically persisted, host-local snapshot generation for every
  response.

An unhealthy or identity-mismatched installed runtime is never advertised as
an active profile. A positive `unknownConsumers`, degraded health, missing UMA
metrics, or missing PSI closes broker GPU admission. The probe never kills or
pauses an unknown process.

When `cudaMemoryProbe` is enabled, a failed CUDA Driver API measurement also
degrades the snapshot. The measurement answers a different question from
Linux `MemAvailable`: it reports what a new CUDA context can allocate at that
moment after driver reservations and current contexts are accounted for. On a
unified-memory host both gates are required for a new profile. Reclaimable
Linux cache can make `MemAvailable` materially larger than immediately
CUDA-allocatable memory.

DGX Spark exposes unified CPU/GPU memory and may report device-level
`memory.used` as `N/A`. In that case the snapshot reports reconciliation as
unavailable instead of inventing a total. Safety then comes from the union of
both NVIDIA process inventories, exact PID-to-runtime binding, per-process
usage where available, host `MemAvailable`/PSI admission, the optional fresh-
context CUDA envelope, and the separately required gateway drain. A host whose
driver does expose device totals receives the additional residual-memory check
automatically.

## Install the inventory

Copy [`resource-probe.example.json`](../examples/resource-probe.example.json)
to `/etc/go7-spark-resource-probe/SERVICE_ACCOUNT.json` and replace the sample
digests with the installed immutable identities. Docker exposes its image ID
with:

```bash
docker inspect --format '{{.Image}}' INSTALLED_CONTAINER
```

For an active systemd user unit, hash the executable inode that is actually
running:

```bash
pid="$(systemctl --user show --property=MainPID --value INSTALLED_UNIT.service)"
sha256sum "/proc/${pid}/exe"
```

The JSON schema is closed: unknown fields, duplicate mappings, mutable tags,
and non-SHA-256 identities are rejected. Container and unit names are literal
administrator configuration; requests cannot supply them.

Set `cudaMemoryProbe` to `true` for UMA admission. The helper uses the installed
CUDA Driver API, creates its context in a short-lived child, records
`cuMemGetInfo`, releases the primary-context reference, and exits before NVIDIA
process inventory is sampled. A missing driver or invalid result fails closed.

When resource controllers are installed, add each controller's durable state
file to `controllerStateFiles`. The controller must atomically replace this
mode-`0600` file after applying a fenced mutation and before returning its HTTP
acknowledgement. The probe never writes it. The broker requires the subsequent
snapshot to contain the exact random mutation ID, lease, fence, epoch, control
generation, mode, safe-boundary result, and checkpoint proof.
[`controller-state.example.json`](../examples/controller-state.example.json)
shows the closed state-file schema.

Provision the config and a random bearer credential as regular, non-symlink
files owned by the probe service account and mode `0400` or `0600`. Keep their
parent directory administrator-controlled and not writable by that account.
Provision the generation directory for the service account with mode `0700`.
The generation file and its lock must remain on durable local storage, not a
temporary or per-session directory.

The included [`go7-spark-resource-probe@.service`](../systemd/go7-spark-resource-probe@.service)
runs the probe as the same dedicated account whose systemd user units it
observes. Install the repository under `/opt/go7-spark-broker`, provision the
files above, then enable exactly one instance for the GPU host:

```bash
sudo systemctl enable --now go7-spark-resource-probe@SERVICE_ACCOUNT.service
```

If Docker is rootless, the probe uses the service account's fixed
`/run/user/UID/docker.sock` when present. For a system Docker socket, grant the
dedicated account only the access already required by its installed runtime.
Docker socket access is effectively privileged and should not be granted to
untrusted callers.

The shipped system unit exposes the service account's home and runtime
directory read-only rather than hiding them: `systemctl --user` needs the user
manager bus under `/run/user/UID`, and rootless Docker uses the same runtime
tree. Broker/probe state remains confined to the explicit `/var/lib` path.

Connect the broker's resource policy to the service using a loopback endpoint
and a separate mode-restricted copy of the bearer credential:

```json
{
  "probe": {
    "endpoint": "http://127.0.0.1:8791",
    "tokenFile": "/absolute/path/to/probe-client-token",
    "required": true
  }
}
```

## Snapshot contract

A healthy response follows the strict resource-governor contract:

```json
{
  "health": "healthy",
  "generation": 42,
  "unknownConsumers": 0,
  "activeProfiles": ["gpu.local-text"],
  "profiles": {
    "gpu.local-text": {
      "health": "healthy",
      "identityVerified": true,
      "runtimeIdentity": "oci-image:sha256:opaque;container:opaque",
      "ownerId": "spark.probe",
      "gpuMemoryBytes": 4294967296
    }
  },
  "controllerStates": {},
  "gpuMemory": {
    "reportedUsedBytes": null,
    "attributedBytes": 4294967296,
    "residualBytes": null,
    "reconciled": false,
    "toleranceBytes": 67108864
  },
  "metrics": {
    "umaTotalBytes": 137438953472,
    "umaAvailableBytes": 107374182400,
    "swapFreeBytes": 0,
    "memoryPressureSomeAvg10": 0.0,
    "cudaAllocatableBytes": 85899345920,
    "cudaAddressSpaceTotalBytes": 137438953472,
    "sampledAtMonotonic": 12345.5
  },
  "observabilityErrors": []
}
```

`runtimeIdentity` values are diagnostic opaque strings. The security decision
comes from comparing observed immutable identities to the installed config and
binding every GPU PID to exactly one configured cgroup. The bearer credential
authenticates the reader; bind remains numeric loopback and responses are
marked `Cache-Control: no-store`.

The probe's generation is durable and monotonic for observations. Causality
comes from a newer generation carrying the exact controller-published random
mutation identity, not from comparing counters alone. The controller-state
contract is described in [`RESOURCE-GOVERNOR.md`](RESOURCE-GOVERNOR.md); this
read-only service reports the state but never claims or causes a control
operation to complete.
