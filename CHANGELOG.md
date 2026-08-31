# Changelog

## 0.2.0

- Add short-lived CUDA process memory telemetry and a required second admission
  envelope for GPU-enabled brokers, preventing reclaimable Linux memory from
  being mistaken for immediately CUDA-allocatable capacity on unified-memory
  hosts. CPU-only brokers may leave the probe disabled.
- Add an authenticated, fenced systemd-user training lifecycle controller with
  crash recovery, immutable checkpoint receipt verification, checkpoint
  advancement enforcement, and exact-evidence resume checks.
- Add a pure, deterministic routing compiler/simulator with canonical config
  revisions shared by production route selection and offline validation.
- Add an authenticated read-only resource probe with immutable Docker/systemd
  identity checks, CUDA consumer classification, opportunistic total-versus-
  attributed GPU-memory reconciliation, UMA/PSI metrics, and durable monotonic
  generations.
- Fail closed on incomplete, regressed, or causally stale resource snapshots
  and verify probe-observed random controller mutation identities before
  admitting or releasing a lease.
- Consolidate owner-only no-symlink configuration and credential reads, and
  reject provider responses whose model identity differs from the selected
  installed route.
- Document CPU-only canary, GPU promotion, and fenced rollback gates.
- Add durable broker epochs, fenced resource leases, host singleton locking,
  restart reconciliation, UMA admission hooks, and acknowledged governor modes.
- Publish model-specific MCP tools only when their typed capabilities are
  installed, and let deferred jobs yield immediately to other runnable work.
- Stage production upgrades in immutable release directories with authenticated
  health/version promotion, live-database-preserving binary rollback, and
  recoverable quarantine for unregistered artifact bytes.
- Add administrator-configured multi-profile OpenAI-compatible inference
  routing with discoverable safe route metadata for HTTP, CLI, and MCP users.
- Validate capability requests before runtime activation and keep a separate
  internal cooperative-preemption hook for future resumable protocol adapters;
  no training capability is advertised in 0.2.
- Enforce a durable minimum normal-mode quantum for displaced training
  controllers and a bounded inference window before restoration.
- Reject unsafe cross-profile container stop lists and namespace 3D cleanup by
  broker ownership.
- Fail protocol-1.0 training yields instead of reporting false completion.
- Harden credential files, redirects, aggregate storage, queue depth, MCP input,
  upload concurrency, request timeouts, CI pins, and public-history hygiene.
- Require observable GPU admission, host-wide live locking plus durable epochs,
  causal governor generations, verified training checkpoints, managed unload
  before training resume, and new-epoch recovery takeover.

## 0.1.0

- Added the protocol 1.0 job, event, artifact, and continuation contracts.
- Added durable SQLite scheduling, restart reconciliation, cancellation, and
  idempotency enforcement.
- Added generic OpenAI-compatible text and optional Hunyuan3D runtime adapters.
- Added CLI, optional MCP access, systemd deployment, public documentation,
  repository safety tests, and clean-room packaging verification.
