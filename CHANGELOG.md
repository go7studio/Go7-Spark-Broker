# Changelog

## 0.2.0

- Add durable broker epochs, fenced resource leases, host singleton locking,
  restart reconciliation, UMA admission hooks, and acknowledged governor modes.
- Add administrator-configured multi-profile OpenAI-compatible inference
  routing with discoverable safe route metadata for HTTP, CLI, and MCP users.
- Validate capability requests before runtime activation and keep a separate
  control loop observing priority while cooperative long work runs.
- Reject unsafe cross-profile container stop lists and namespace 3D cleanup by
  broker ownership.
- Fail protocol-1.0 training yields instead of reporting false completion.
- Harden credential files, redirects, aggregate storage, queue depth, MCP input,
  upload concurrency, request timeouts, CI pins, and public-history hygiene.

## 0.1.0

- Added the protocol 1.0 job, event, artifact, and continuation contracts.
- Added durable SQLite scheduling, restart reconciliation, cancellation, and
  idempotency enforcement.
- Added generic OpenAI-compatible text and optional Hunyuan3D runtime adapters.
- Added CLI, optional MCP access, systemd deployment, public documentation,
  repository safety tests, and clean-room packaging verification.
