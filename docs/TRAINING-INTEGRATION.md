# Training integration

This repository coordinates inference today and provides the fenced governor
boundary required to keep independent training from racing it. It does not yet
advertise a managed training capability.

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
