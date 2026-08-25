# Resource governor protocol

The broker is the only component allowed to decide that a GPU task may start.
An existing training or inference governor can remain responsible for applying
framework-specific limits, but it operates as a fenced adapter behind the
broker rather than as a second scheduler.

## Resource snapshot

When configured, the broker reads `GET /v1/resource-snapshot` over loopback with
the controller bearer credential. A successful response is a JSON object:

```json
{
  "health": "healthy",
  "generation": 42,
  "unknownConsumers": 0,
  "activeProfiles": ["gpu.background-workload"],
  "profiles": {
    "gpu.background-workload": {
      "health": "healthy",
      "latencyMs": 180,
      "availableConcurrency": 1
    }
  }
}
```

`unknownConsumers` counts GPU/CUDA consumers that the governor cannot bind to a
known administrator-installed profile. Any positive value closes admission;
the broker never kills an unknown process. Profile latency and concurrency are
routing hints only. Admission is always re-sampled after lifecycle controls.

## Apply a target mode

For every configured controller affected by a request, the broker sends
`POST /v1/resource-mode`:

```json
{
  "protocolVersion": "1.0",
  "leaseId": "lease_opaque",
  "fencingToken": "fence_epoch_opaque",
  "controlGeneration": 1,
  "targetMode": "interactive-boost",
  "reason": "admit:interactive:job_opaque"
}
```

The response must echo the lease and fence and acknowledge the exact generation
and effective mode:

```json
{
  "leaseId": "lease_opaque",
  "fencingToken": "fence_epoch_opaque",
  "acknowledgedGeneration": 1,
  "effectiveMode": "interactive-boost",
  "health": "healthy",
  "appliedAtSafeBoundary": true
}
```

Controllers must:

- reject a fence from an older broker epoch or a different active lease;
- make an identical generation idempotent;
- reject a lower generation after a higher generation was applied;
- apply training changes only at a framework-safe boundary;
- report `healthy` only when the requested effective state is observable; and
- preserve the last verified checkpoint when a target mode requests release.

The broker restores each controller to its administrator-configured normal mode
before releasing the job lease. A missing or mismatched acknowledgement leaves
the lease `unknown` and quarantines further GPU admission.

## What throttling means in this release

Reducing request concurrency, batch size, CPU/IO weight, or training activity is
useful, but none proves that unified memory or a CUDA context was released.
After the acknowledgement, the broker reads a new snapshot:

- an exclusive job is admitted only after every incompatible profile is gone;
- an inference permit may use an already-resident copy of its own profile;
- another resident profile remains incompatible in this release; and
- `sharedCertifications` must remain empty until the repository ships a
  digest-bound certification format and fault suite.

This deliberately makes a governor's checkpoint-and-release mode useful today
while preventing an unmeasured live throttle from being mistaken for GPU or
memory isolation.

## Failure and restart rules

Every lease is stored before the first controller mutation. On restart, the
broker acquires the host lock, increments its epoch, marks old leases unknown,
and attempts an exact fenced restore. It releases a stale lease only when the
resource probe is healthy, reports no unknown consumers, and no longer reports
the leased profile as active. Otherwise the host stays quarantined for operator
reconciliation.

