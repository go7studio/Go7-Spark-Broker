# Staged rollout and rollback

Broker upgrades must be staged beside the active service. Never install a new
wheel into the virtual environment of a running broker, and never use an older
broker that ignores epochs or fences as the rollback target after controllers
have adopted the fenced governor protocol.

## Release identities

Build one wheel from a reviewed commit and record all of these before copying
it to a host:

- Git commit SHA;
- wheel filename and SHA-256 digest;
- broker version;
- routing configuration revision, when inference routes are installed; and
- resource policy and probe inventory revisions maintained by the operator.

Install the wheel into a new versioned virtual environment. Do not mutate the
active environment. Give each parallel instance a distinct loopback port,
token, data directory, broker ID, environment file, and systemd unit. GPU
instances must nevertheless share the one host lock, durable epoch, probe, and
controller generation domain.

## CPU-only canary

The first canary exposes only `system.echo`. Omit every GPU variable, including
OpenAI routes, single-model configuration, 3D workload roots, resource policy,
host lock, and epoch. This makes it possible to validate packaging and the
protocol without touching a model, container, gateway, trainer, or GPU lease.

Required canary gates:

1. The active production unit and its public loopback health endpoint remain
   continuously available.
2. The canary reports the expected broker version and only the expected
   capabilities.
3. Authentication rejects missing and incorrect bearer credentials.
4. `system.echo` completes through the durable queue and event state machine.
5. CLI upload/download preserves bytes and SHA-256.
6. MCP upload and chunked download preserves a GLB artifact and SHA-256.
7. Restarting the canary preserves completed jobs and artifacts and does not
   create active or unknown GPU leases.
8. Stopping and removing the canary leaves production unchanged.

The canary is not evidence that GPU switching is safe. It proves only the
package, service, authentication, job, CLI, MCP, and artifact paths.

## GPU promotion gate

Do not enable a GPU capability until every item below is true:

- the read-only resource probe identifies every CUDA PID and binds it to an
  administrator-installed immutable runtime identity;
- no legacy or second coordinator can mutate the same GPU outside the shared
  host lock, epoch, lease, fence, mutation-observation, and probe-generation domain;
- an inference gateway closes admission and drains accepted in-flight calls
  before its model can unload;
- every displaced training controller can publish and verify an immutable
  checkpoint at a safe boundary, acknowledge the causal generation, release
  CUDA, and resume from the exact next step;
- transition, crash, stale-epoch, wrong-fence, delayed-response, disk-full,
  unknown-consumer, and restart-takeover tests pass;
- a bounded inference window and minimum training quantum prevent thrashing
  and starvation (`maximumInferenceWindowSeconds` and every training
  controller's `minimumNormalSeconds` are mandatory); and
- the rollback release is a previously proven fenced build, never `0.1.x`.

An acknowledgement is not sufficient by itself. The broker must observe a
newer resource snapshot containing the exact random mutation identity and
fenced controller state, then verify the expected profile presence or absence
before advancing the lease state.

## Promotion and rollback

Keep the previous release directory intact. Promotion changes only the service
unit's versioned executable target or an atomic `current` symlink, then restarts
the service and checks liveness, readiness, authenticated status, capability
revision, queue state, and resource state. Back up SQLite with its online
backup API before promotion; do not copy a live database file directly.

Binary rollback preserves the live database. The pre-deploy SQLite backup is
an explicit operator recovery artifact and is never restored automatically:
rewinding it could discard jobs and artifact registrations created after the
backup. Schema changes must therefore remain backward-compatible with the
previous fenced release or provide a separately reviewed forward repair.

If a gate fails before any controller mutation, restore the previous fenced
release target and restart it. If a mutation outcome is unknown, do not roll
back into a process that might guess ownership. Preserve the lease journal and
throttle state, quarantine GPU admission, keep diagnostic reads available, and
recover through the documented new-epoch takeover protocol.

Record the result as an operator-owned deployment artifact. Hostnames, IP
addresses, usernames, tokens, local paths, checkpoint identifiers, and model
licenses do not belong in this public repository.
