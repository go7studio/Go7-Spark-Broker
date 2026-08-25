# Repository instructions

- Keep the broker independent from Workhorse UI, chats, vendors, and local
  device identity. It is a typed execution host.
- Do not commit credentials, environment files, runtime databases, generated
  artifacts, model weights, machine paths, or private network names.
- Caller requests never select commands, executables, Docker containers,
  endpoints, or filesystem paths. Those remain administrator configuration.
- New runtime adapters require complete invocation descriptors, bounded input
  and output validation, idempotent activation, cancellation, cleanup ownership,
  restart behavior, and tests.
- Preserve protocol 1.0 compatibility or document and test an explicit version
  transition.
- Before committing, run `python3 -m unittest discover -s tests -v`,
  `python3 -m pytest -q`, and `bash -n deploy-user.sh`.
