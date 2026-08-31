# Training integration

This repository coordinates inference today and ships a fenced,
checkpoint-aware lifecycle controller for an independently installed systemd
user training service. It does not yet advertise a managed training
capability.

## Current safe path

An administrator may connect an existing training controller through
`SPARK_RESOURCE_POLICY_FILE`. When interactive inference arrives, the broker:

1. validates the request and selects an installed inference profile;
2. records a fenced GPU permit;
3. asks the training controller to checkpoint and release its GPU profile;
4. requires a safe-boundary acknowledgement and immutable checkpoint digest;
5. re-samples unified-memory and runtime ownership;
6. admits inference only if the incompatible training profile has released; and
7. unloads and verifies the inference profile absent; and
8. uses a new fenced acknowledgement to restore the controller's normal mode.

If the controller merely reduces compute while training remains resident, the
inference request stays queued. Live co-residency is intentionally disabled
until the exact runtime pair passes a digest-bound shared-mode certification.

## Shipped lifecycle controller

`spark-training-controller` exposes only the two authenticated loopback
operations required by the resource governor. Its administrator-owned config
fixes one systemd user unit, controller/profile identity, mode pair, checkpoint
root, receipt, and state paths. A caller cannot select commands, units,
endpoints, or filesystem paths.

Copy `examples/training-controller.example.json` outside the checkout and make
the config and bearer credential mode `0600`. The system template
`systemd/go7-spark-training-controller@.service` runs the controller as the
same dedicated account that owns the training user unit. The controller:

- persists the request before changing the unit;
- recovers an interrupted transition on restart;
- enforces broker epoch, lease, fence, and control-generation ordering;
- stops the fixed unit and verifies it inactive before releasing the profile;
- hashes every receipt member and rejects symlinks, traversal, missing files,
  size mismatches, or digest changes;
- persists the receipt identity observed immediately before stopping and
  requires a same-run, newer checkpoint identity on every release, including
  first controller use and crash recovery, and refuses an ID-only bump whose
  file manifest evidence did not change; and
- revalidates the exact prior checkpoint before resuming the unit.

The trainer must publish a new owner-only receipt matching
`examples/checkpoint-receipt.example.json` as part of its graceful stop path.
Every listed path is relative to the configured checkpoint root. The receipt
is only accepted after every file is durable and immutable. A generic
`SIGINT`, an unchanged checkpoint pointer, or an inactive process without a
new verified receipt fails closed.

Do not set `allowFreshStart` in a production integration unless the installed
trainer separately proves that starting without a checkpoint is intentional.
The controller does not interpret framework-specific checkpoint formats, so an
exact next-step resume test remains an administrator certification gate.

### Always-resident inference constraint

Protocol 1.0 cannot combine this rotation controller with an inference process
that is required to remain GPU-resident. Before the broker restores training,
it must unload the selected inference profile and prove it absent. Broker
startup therefore rejects a controller-backed configuration whose inference
profile has no managed unload lifecycle. If a container lifecycle is supplied,
the broker stops that container after the bounded inference window.

Consequently, do not attach this controller to an always-resident Qwen service.
Keep the controller disarmed while Qwen remains resident. A smaller baseline
model may be evaluated for certified coexistence, but CUDA headroom alone is
not certification and `sharedCertifications` remains rejected in protocol 1.0.

## Required managed-training contract

A future `model.training.run` adapter must be administrator-installed and must
not accept a command, executable, image, container name, endpoint, or host path
from a caller. Its immutable profile revision must bind:

- recipe and image digests;
- base-model and dataset-manifest digests;
- maximum unified-memory, CPU, IO, and checkpoint-storage envelopes;
- framework readiness and heartbeat schemas;
- checkpoint and graceful-yield controls;
- exact resume compatibility; and
- runtime ownership labels carrying broker epoch, lease, and run identity.

Training state must distinguish a job from a long-lived training run. Required
durable resources are a training run, runtime instance, GPU lease, checkpoint
collection, and last-known-good checkpoint pointer.

## Protocol gate

Protocol 1.0 terminal states cannot distinguish successful model completion
from a verified checkpoint yield. The current scheduler therefore returns
`yield_protocol_unsupported` if a cooperative adapter yields; it never reports
that slice as completed.

Managed training may ship only with a compatible protocol that exposes a
non-successful resumable outcome to Workhorse, CLI, and MCP clients. A resume
must bind to one immutable checkpoint manifest rather than a local directory.

## Checkpoint acceptance

A checkpoint is complete only after all model, optimizer, scheduler, scaler,
random-number, sampler/data-position, global-step, recipe, base-model, and
framework-version records are durable. Publication must use a temporary
collection, incomplete marker, per-member size and digest manifest, durable
flush, atomic publication, and an optional load probe. Never overwrite the last
known-good checkpoint.

GPU release is a separate check. A valid checkpoint does not prove that the
process, CUDA context, or unified-memory allocation disappeared.

## Tests required before advertising training

- crash injection around every checkpoint and lease transition;
- exact next-step and next-sample resume comparison;
- storage-full behavior that preserves the previous checkpoint;
- higher-priority inference causing bounded checkpoint-and-release;
- controller timeout and mismatched fence behavior;
- broker restart with both surviving and dead runtimes;
- unknown GPU consumer quarantine without termination;
- continuous interactive traffic and explicit training-starvation policy; and
- digest/runtime changes invalidating every shared-mode certification.
