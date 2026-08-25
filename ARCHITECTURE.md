# Architecture

## Two independent planes

```text
Workhorse / CLI / MCP client
          │
          │ HTTPS + bearer token + protocol 1.0
          ▼
Go7 Spark Capability Broker
  ├─ capability registry
  ├─ durable job and event store
  ├─ multi-profile inference router
  ├─ host singleton, epoch, and fenced resource coordinator
  ├─ durable GPU leases and acknowledged lifecycle controls
  ├─ artifact registry and integrity checks
  └─ allowlisted runtime adapters
          │
          ├─ local OpenAI-compatible model server
          └─ explicitly installed 3D workload
```

Workhorse owns product intent and caller policy. The broker does not know about
Workhorse chats, projects, vendors, or users. It accepts only the typed request
contract and returns typed jobs and artifacts.

The broker owns runtime truth. A caller cannot claim that a model is loaded,
that an artifact is valid, or that a job completed. The broker activates the
selected profile, records transitions, validates outputs, and reports an
interrupted state after uncertain restart work.

## Relationship to existing inference routers

This broker does not replace an inference engine or a large-scale inference
router:

- [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) and
  SGLang serve model APIs. They can sit behind the OpenAI-compatible adapter.
- [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/) coordinates generative-AI
  frontends, workers, KV-aware routing, and scaling. A Dynamo frontend can sit
  behind an adapter when that deployment is appropriate.
- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
  serves model repositories through per-model schedulers and backends. A Triton
  deployment can likewise become an adapter target.

Go7's layer exists for a different boundary: durable cross-runtime jobs,
Workhorse-compatible typed capability discovery, model/profile activation on a
constrained local host, artifact transfer and integrity, restart truth, and
separately authorized continuations. Existing serving stacks remain useful
runtime components rather than competing copies inside this repository.

## Generic core and explicit adapters

The scheduler, store, HTTP protocol, artifact registry, CLI client, and MCP
adapter are capability-family neutral. Runtime-specific behavior lives behind
the `Executor` interface in `spark_broker/executors.py`.

An executor must provide:

- a stable capability ID and runtime profile ID;
- a complete invocation descriptor with typed inputs, outputs, and constraints;
- bounded validation before touching the runtime;
- an idempotent `activate()` operation;
- cancellation and timeout handling;
- atomic artifact registration with hashes; and
- cleanup limited to resources explicitly owned by that executor.

Runtime executors are code-reviewed adapters, not dynamically supplied shell
commands. This is intentional: a generic arbitrary-command plugin would turn
the broker protocol into remote code execution.

## Profile rotation

The scheduler still executes one capability job at a time, but a separate
control loop continues observing higher-priority work while a cooperative
long-running adapter executes. Pure executor validation and inference route
selection happen before any resource mutation.

The resource coordinator owns an OS lock, a monotonic broker epoch, and durable
leases. It samples host memory pressure and an optional administrator-installed
resource probe, applies fenced controller generations, re-samples after those
controls, and only then activates the selected profile. Executors never stop
one another. The 3D adapter removes crash leftovers only when they carry the
exact broker ownership label; cross-profile stop lists are rejected.

An OpenAI-compatible inference profile uses a per-call `permit` because the
model service may remain resident after the call. Batch/3D work uses an
exclusive lease. A resident different profile prevents admission unless it was
verified absent after the governor action. This release does not accept
shared-mode certifications.

## Persistence

SQLite stores requests, jobs, events, artifact metadata, idempotency keys, and
route history. Artifact bytes live in the configured data directory and enter
the registry by atomic rename after size and digest verification.

On process start, the coordinator first acquires the host lock, advances its
epoch, reconciles stale fences and resource observations, and only then marks
unattached active jobs `interrupted`. It never guesses that GPU work completed
or that a database lease proves physical release. A repeated idempotency key
returns the durable record for the original request. A changed payload with the
same key conflicts.

## Inference routing

One logical `text.chat.generate` capability may contain several
administrator-installed routes. Each route owns its model identity, profile,
loopback endpoint, optional container, service classes, resource envelope, and
priority. The caller may name an advertised model or request a service class
and policy preference. It cannot send endpoint, container, command, or path
data.

The router may use observed health, residency, latency, concurrency, memory,
and administrator priority. The chosen route and reason are stored in the
execution plan and returned with the job result. Routing hints never bypass the
coordinator's fresh admission sample.

## Training boundary

The coordinator contract can govern an external training runtime today, but a
managed training executor is not shipped. Protocol 1.0 cannot represent a
verified resumable yield without false completion, so the scheduler fails a
yield with `yield_protocol_unsupported`. See
[`docs/TRAINING-INTEGRATION.md`](docs/TRAINING-INTEGRATION.md).

## Continuations

A continuation is typed output data describing a separately callable next
capability. It is not a command. The originating request must approve its
capability, and the control plane must grant and dispatch the exact
capability/tool pair. The broker never launches desktop tools such as Blender.

## Adding a runtime

Add a new executor only when the runtime cannot be represented by an existing
adapter. Keep configuration in environment variables or mode-restricted files,
register the executor only when its required settings are present, publish a
complete descriptor, and add contract, activation, cancellation, restart, and
artifact tests. Do not add a caller-supplied executable or path field.
