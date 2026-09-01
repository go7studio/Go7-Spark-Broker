# DGX Spark baseline-model and training-coexistence handoff

Date: 2026-08-31

Status: reversible Nemotron live canary; real-training overlap rejected and held

This document records the measured operating profile that follows the CUDA
admission and fenced-training work in commit `3382f2f`. Nemotron is currently
serving through the authenticated broker as a reversible canary. The retained
Qwen container is the rollback target. Bloom V40 was admitted only for bounded
coexistence experiments and is now held because it failed the interactive
latency gate.

## Decision

Keep NVIDIA Nemotron 3 Nano 4B FP8 as the leading always-on canary baseline.
Keep all MCP visibility, schemas, permissions, attestations, replay control,
dependency ordering, child limits, credentials, and execution outside the
model. The model is only a bounded planner and call proposer. Do not run Bloom
V40 continuously beside it under the tested scheduler profiles.

The tested SGLang shadow profile was:

- 8,192-token context.
- Static memory fraction `0.070`.
- Two running requests and ten Mamba cache slots.
- FlashInfer autotuning disabled for reproducibility.
- Prefill CUDA graphs disabled.
- Decode CUDA graphs captured only for batch sizes one and two.
- One authoritative generation stream per request and at most one independent
  child stream in this resource profile.

The initial two-request attempt was not truly parallel: SGLang silently capped
it to one because nine Mamba slots could not satisfy two five-slot requests.
Ten slots made both streams real.

## Current live and rollback state

The current machine state was revalidated after the experiments:

- `nemotron3-sglang-resident` is running on loopback port 30000 with the exact
  profile above.
- The authenticated inference gateway and `go7-spark-broker` are listening on
  loopback ports 8788 and 8790.
- A fresh broker job completed with the exact response
  `NEMOTRON_BROKER_RECHECK_OK`.
- The MCP server completed `initialize`, published its capability-filtered
  typed tool catalog, and returned the live `gpu.nemotron3.nano4b` capability
  through `spark_capabilities`.
- `qwen38-sglang-fast` is stopped, not deleted. It had been cold-started and
  served before the final Nemotron swap, proving that the retained image and
  container remain a usable rollback path.
- `bloom-v40-500m.service` is inactive, the operator hold exists, and the
  fail-closed autoyield service is active. A direct start while held was denied.

This is a live canary, not a permanent replacement of Qwen and not an approval
for unrestricted MCP authority.

## Measured inference result

The counterbalanced benchmark used three blocks, 20 requests per cell, frozen
prompts, mandatory cache flushes, synchronized batches, and strict non-empty
JSON validation.

| Profile | One-stream completion tok/s | Two-stream completion tok/s | Two-stream p95 | Valid |
|---|---:|---:|---:|---:|
| Eager, one effective server slot | 41.11 | 41.02 | 2.93 s | 120/120 |
| Eager, two Mamba-backed slots | 40.95 | 83.19 | 1.49 s | 120/120 |
| Decode graphs for batches 1 and 2 | 41.77 | 85.59 | 1.45 s | 120/120 |
| Decode graphs plus 10% training duty | not measured | 80.80 | 1.57 s | 60/60 |
| Decode graphs plus 25% training duty | 35.66 | 73.23 | 1.72 s | 120/120 |

Decode graphs added 2.9% two-stream throughput over eager two-stream serving.
The larger gain came from correctly allocating the second Mamba state slot:
two-stream throughput doubled without doubling per-request latency.

## Planner and MCP result

The optimized profile repeated an 18-case deterministic gated corpus three
times. It produced 27/54 exact decisions, 88.89% schema-valid output, and zero
unsafe proposals. That is not accurate enough for an authoritative agent or an
unrestricted tool catalog. It is sufficient only for shadowing a proposer whose
output is rejected unless the deterministic broker independently validates it.

The same corpus was rerun against the live pre-training canary and reproduced
27/54 exact decisions, 88.89% schema-valid output, zero unsafe proposals, and a
1.132-second median. The stable result confirms the tool parser and bounded
proposal behavior; it does not improve the promotion verdict.

The experimental NVIDIA Qwen3 8B NVFP4 arm was rejected on the tested SGLang
runtime. It left materially less allocator-visible reserve, produced weak draft
results, and emitted unmapped KV-scale warnings. There is no official Qwen3.8
8B checkpoint in the researched family, so it must not be represented as a
smaller form of the production Qwen3.8 model.

## Real Bloom V40 training result

The real 499,946,816-parameter Bloom V40 job built and verified its cached
corpus: 10,253 documents, 965,544,928 tokens, 2,519,497 training windows, and
51,025 validation windows. Its canary used a 15% process CUDA-memory fraction
and either a 10% duty-cycle ceiling or, for the separate MPS arm, a 10% MPS
active-thread limit.

Four checkpointed runs demonstrated exact continuation across steps 3, 4, 5,
and 17. Every stop used SIGINT, completed the current optimizer boundary,
published `latest.pt`, and exited cleanly. The final retained checkpoint is a
hard link to `step_000017.pt`, is 6,060,553,263 bytes, records 208,896 tokens and
sampler position 544, and hashes to
`4f16f8a972fe8fff670722486ecfe16f5db4bcf5f7de569ca86251efc718f107`.

The real duty-cycle overlap retained 34,766,872,576 bytes minimum
allocator-visible CUDA free and 90,046,420 KiB minimum host `MemAvailable`.
Memory admission therefore passed. Interactive latency did not: the worst
one-stream p95 rose from 1.469 seconds at baseline to 3.274 seconds, and the
two-stream overlap had a 2.325-second p95 block. All 120 outputs remained valid.

MPS did not solve the collision. With a 10% active-thread limit, minimum CUDA
free fell to 10,421,829,632 bytes, minimum host `MemAvailable` to 84,497,604
KiB, one-stream p95 stayed around 3.30 seconds, and two-stream p95 reached 5.08
seconds. All 120 outputs were valid, but the latency and reserve regressions are
release-gate failures.

This certifies the real trainer's checkpoint/yield/resume mechanics under the
tested canary. It does not certify continuous background coexistence. The next
training experiment needs request-aware pause/yield scheduling or smaller
microbatches; fixed duty cycling and MPS are both rejected for the live route.

## Tested operating profile

1. Leave Nemotron resident with the exact 8K/two-request/ten-Mamba-slot/decode-
   graph profile recorded above.
2. Keep one authoritative generation stream per request. The second server slot
   is for another independent request or one bounded independent child, never a
   second authority for the same request.
3. Keep Bloom V40 held by default. Admission requires healthy Nemotron, stopped
   Qwen, known GPU consumers, at least 32 GiB host `MemAvailable`, live CUDA
   reserve, and checkpoint readiness.
4. For the next experiment, retain the 15% CUDA-memory ceiling and start no
   higher than 10% compute duty, but add request-aware yielding before any
   inference burst. Fixed duty alone is not approved.
5. Re-admit only from a fresh resource sample and a verified checkpoint. Abort
   on p95/queue pressure, telemetry uncertainty, an unknown GPU consumer, or a
   failed checkpoint.
6. Do not enable MPS for this pair. Do not remove the Qwen rollback container.

Long-context, unique-prefix, request-aware burst, and long-soak tiers remain
required before production promotion.

## Evidence integrity

Raw receipts are kept in the private operator evidence store because they
contain runtime and machine-specific paths. The final live-canary receipt
hashes are:

- Agent/MCP benchmark: `a6cd46eaad4329e6688f205e7ed70c34a6dc11b2eaff0dc018bcdf390234bcc4`
- Inference-only concurrency: `9bb7497b3ab40d457bb6162363018603c94868a6bf29e2c09384bea7a4c8011f`
- Real V40 overlap concurrency: `4d7e51c48481495c4ca2b35b0d4a309a66a450b7f1155e6947ccc5abed2f795d`
- MPS overlap concurrency: `e971ef4dda22fad135ac9c568180185d293385ad5fa9dd8e610ea1d85ff03937`
- Baseline resource series: `31d9a49e4c09b0b55dc649cda63114c0be5ce8bc70137a7e3dbae8d3d2db1be6`
- Real-overlap resource series: `b5129259bbe595c89813f49144f5436fd5ebce91ee74760988435c6bc419258b`
- MPS resource series: `a271011d215e4e3930d5f677ec3ec84506dbaa3e58da238febb40df9c72b2ed8`
- Final real checkpoint: `4f16f8a972fe8fff670722486ecfe16f5db4bcf5f7de569ca86251efc718f107`

The model, runtime revision, prompt/corpus hashes, launch profile, resource
series, output validity, and checkpoint hashes are recorded in those receipts.
