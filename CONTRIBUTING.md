# Contributing

Changes should preserve the broker's small, fail-closed execution boundary.

1. Create a branch and keep machine configuration outside the repository.
2. Add or update tests before changing a public capability contract.
3. Run `python3 -m unittest discover -s tests -v`, `python3 -m pytest -q`, and
   `bash -n deploy-user.sh`.
4. Confirm examples contain no real hosts, usernames, paths, container names,
   model credentials, or generated artifacts.
5. Describe compatibility effects for protocol, capability, and environment
   variable changes.

Runtime contributions must expose a typed descriptor and an allowlisted adapter.
They must not accept caller-supplied commands, executable paths, Docker names,
or network endpoints. Optional third-party runtimes remain separately installed
and licensed; do not vendor model weights into this repository.

Inference routes may add model identities and profile envelopes, but never
publish their endpoint, credential, container, or host path through capability
discovery. Lifecycle controllers must echo the exact lease fence and generation,
be idempotent, and include failure/restart tests. Executors never stop another
profile directly.

Managed training is gated on a protocol that can represent a resumable yield,
immutable checkpoint collections, exact-resume tests, and verified GPU release.
Do not add a protocol-1.0 training adapter that reports a checkpoint yield as
successful completion.
