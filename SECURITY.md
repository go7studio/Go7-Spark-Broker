# Security

## Supported deployment boundary

The broker binds to loopback by default. Keep it there and expose it only
through an authenticated HTTPS reverse proxy or private overlay network. The
escape hatch for a direct non-loopback bind is intended for administrators who
provide an equivalent protected network boundary; plain public HTTP is unsafe.

Every protocol route except `/health/live` requires a bearer token of at least
32 characters. Store tokens and model-runtime keys in files readable only by
the service account with mode `0600` or stricter. The broker rejects inline
secrets by default, validates credential ownership/mode, and accepts inline
values only behind the explicit development escape hatch
`SPARK_ALLOW_INLINE_SECRETS=1`. Do not use that escape hatch in a service.

The bearer token defines one administrator trust domain. It is not a per-user
authorization system: any holder can discover, submit, inspect, cancel, and
download within that broker. Use a separate broker/service account and token
for a different trust boundary. Workhorse grants remain the user-facing control
plane; do not distribute the raw broker token to untrusted callers.

## Trust model

- Callers are trusted only for the capability contract, never for commands,
  endpoints, Docker names, or filesystem paths.
- Runtime adapters and their local workload installations are administrator
  code and are inside the host trust boundary.
- Model output and uploaded artifacts are untrusted data.
- A continuation is data until a separate control-plane authorization and
  installed adapter dispatch it.
- Docker access is privileged. Give it only to the dedicated service account
  and configure literal container allowlists.
- Resource-governor and model-server endpoints must be loopback HTTP. Remote
  broker clients require HTTPS; redirects are not followed with bearer tokens.
- A throttle acknowledgement is not proof of memory release. Fresh resource
  observations and lease fencing decide admission.

## Denial-of-service boundaries

The broker enforces per-artifact size, aggregate artifact storage, pending-job,
JSON-body, request read-timeout, and concurrent-upload limits. MCP rejects an
oversized encoded line before base64 decoding. Operators must still provide
log rotation, disk monitoring, retention policy, and upstream connection/rate
limits appropriate to their host.

The artifact API records a digest for the whole artifact. Range chunks report a
transport-time chunk digest, but that digest is not a pre-published Merkle proof.
Clients that need integrity must verify the full downloaded artifact against the
registered SHA-256 value, as `sparkctl download` does.

## Public repository hygiene

Never commit:

- bearer tokens, API keys, credential files, or environment files containing
  secrets;
- hostnames, IP addresses, usernames, SSH aliases, private overlay-network
  names, or personal home-directory paths;
- model checkpoints, generated artifacts, broker databases, logs, or crash
  staging directories; or
- third-party workload source or weights unless their licenses explicitly
  permit redistribution.

Examples use placeholders and loopback addresses only. Tests use temporary
directories and fake credentials.

## Reporting a vulnerability

Use the repository's private GitHub security-advisory channel when available.
Do not open a public issue containing an exploit, token, private endpoint, or
device identity. Include the affected version, deployment boundary, minimal
reproduction, and whether credentials or artifact integrity are involved.
