# Resource governor protocol

GPU capabilities fail closed unless the broker has all three of these:

- an administrator-provisioned host-wide lock path;
- a durable host-wide epoch file outside `/tmp`, `/var/tmp`, and per-user runtime directories; and
- a required, healthy resource probe with unified-memory admission enabled.

The operating-system lock prevents two live broker processes using the same
path from mutating GPU 0 concurrently. It is not durable after process death.
The persistent epoch, fenced controller protocol, and observed resource probe
provide the cross-restart safety boundary. Every account with GPU access must
use the same paths and governor. The supported managed deployment uses one
dedicated service account.

Every controller mutation is write-ahead journaled with its compensation
before the network call. Controller and probe bearer credentials must be
service-owned files with mode `0600` or stricter. All endpoints are HTTP
loopback URLs; proxy environment variables and redirects are ignored.

## Shared causal generation

The resource probe and every controller response use one host-wide,
monotonically increasing `snapshotGeneration` domain. A controller increments
that generation only after its requested state is observable in the resource
inventory. The broker rejects a post-control snapshot whose `generation` is
older than the acknowledgement. Generations must survive controller restarts.

## Resource snapshot

The broker reads authenticated `GET /v1/resource-snapshot`:

```json
{
  "health": "healthy",
  "generation": 42,
  "unknownConsumers": 0,
  "activeProfiles": ["gpu.training-main"],
  "profiles": {
    "gpu.training-main": {
      "health": "healthy",
      "latencyMs": 180,
      "availableConcurrency": 1
    }
  }
}
```

`unknownConsumers` counts GPU/CUDA consumers the governor cannot bind to a
known administrator-installed profile. Any positive value closes admission;
the broker never kills an unknown process. A selected profile explicitly
reported by `profiles` must have `health: healthy`. Latency and concurrency are
routing hints only.

## Apply a target mode

For each affected controller, the broker sends authenticated
`POST /v1/resource-mode`:

```json
{
  "protocolVersion": "1.0",
  "leaseId": "lease_opaque",
  "fencingToken": "fence_opaque",
  "brokerEpoch": 43,
  "controlGeneration": 1,
  "targetMode": "checkpoint-release",
  "reason": "admit:interactive:job_opaque"
}
```

A successful response is:

```json
{
  "leaseId": "lease_opaque",
  "fencingToken": "fence_opaque",
  "brokerEpoch": 43,
  "acknowledgedGeneration": 1,
  "effectiveMode": "checkpoint-release",
  "health": "healthy",
  "appliedAtSafeBoundary": true,
  "snapshotGeneration": 44,
  "checkpoint": {
    "runId": "run_opaque",
    "checkpointId": "checkpoint_opaque",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }
}
```

The `checkpoint` object is mandatory when a controller declares
`workloadKind: training` and the target is its throttled/release mode. It is not
required while restoring `normalMode`. The manifest digest identifies an
immutable durable checkpoint collection; it is not a local directory claim.

Controllers must persist the greatest accepted broker epoch, active lease
fence, greatest control generation for that fence, effective mode, and shared
snapshot generation before replying. They must reject an older epoch, a
different active lease at the same epoch, or a lower generation. Repeating the
same generation and payload is idempotent. `health: degraded`, a missing safe
boundary, a mismatched echo, or a stale snapshot generation is failure.

## Restart takeover

An old operational fence is never reused after a broker restart. Once the new
broker owns the host lock and has first proved the leased inference profile and
unknown consumers absent, it may send authenticated
`POST /v1/resource-takeover`:

```json
{
  "protocolVersion": "1.0",
  "previousLeaseId": "lease_old",
  "previousFencingToken": "fence_old",
  "recoveryFencingToken": "fence_44_recovery_opaque",
  "brokerEpoch": 44,
  "controlGeneration": 1000001,
  "targetMode": "normal",
  "reason": "restart-recovery:lease_old"
}
```

The controller atomically compares the previous lease and fence with its
persisted state, verifies that `brokerEpoch` is newer, adopts the recovery
fence, applies the target mode, advances the shared snapshot generation, and
returns:

```json
{
  "previousLeaseId": "lease_old",
  "previousFencingToken": "fence_old",
  "recoveryFencingToken": "fence_44_recovery_opaque",
  "brokerEpoch": 44,
  "acknowledgedGeneration": 1000001,
  "effectiveMode": "normal",
  "health": "healthy",
  "appliedAtSafeBoundary": true,
  "snapshotGeneration": 45
}
```

Any comparison failure is a conflict, not a best-effort restore. The broker
keeps the stale lease `unknown` and the host quarantined.

## Admission and release order

Admission order is: validate request, select route, sample resources, journal
lease and control intent, apply controller mode, verify acknowledgement,
re-sample the causal generation, check memory and compatibility, activate the
selected runtime, and re-sample activation.

Release order is deliberately reversed: complete or cancel work, unload the
selected broker-managed runtime, prove its profile absent when another
controller was displaced, restore controllers to normal, re-sample, then mark
the lease released. A failed unload never resumes training. The lease becomes
`unknown`, the host is quarantined, readiness becomes unavailable, and new jobs
are rejected. Status and artifact reads remain available for diagnosis.

An unthrottled inference permit may intentionally leave its one profile
resident. It cannot rotate to a different profile until the probe and a
configured lifecycle controller prove the old profile absent.

## Co-location policy in protocol 1.0

Reducing concurrency, batch size, CPU/IO weight, or training activity does not
prove CUDA or unified-memory isolation. This release therefore permits
checkpoint-and-release rotation but not live training/inference co-residency.
`sharedCertifications` must remain empty until a digest-bound certification
format and crash/fault suite ship. This is the safe path for a training
controller today: checkpoint, release GPU ownership, run the routed call,
unload the model, then resume from the verified checkpoint.

## Failure responses

Controllers should return non-2xx responses for invalid schema, authentication,
stale epoch, fence conflict, generation conflict, unsafe-boundary timeout, or
checkpoint failure. The broker treats transport errors, redirects, malformed
JSON, bodies over 1 MiB, non-healthy responses, and echo mismatches as failed
admission or failed release. It never guesses that a mutation succeeded.
