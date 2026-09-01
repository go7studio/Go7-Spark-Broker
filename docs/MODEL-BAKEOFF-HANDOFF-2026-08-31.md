# DGX Spark baseline-model and training-coexistence handoff

Date: 2026-08-31

Status: shadow-tested; not promoted

This document records the measured operating hypothesis that follows the CUDA
admission and fenced-training work in commit `3382f2f`. It is evidence for a
future staged rollout, not authorization to replace the production inference
route or arm a trainer.

## Decision

Use NVIDIA Nemotron 3 Nano 4B FP8 as the leading always-on shadow baseline. Keep
all MCP visibility, schemas, permissions, attestations, replay control,
dependency ordering, child limits, credentials, and execution outside the
model. The model is only a bounded planner and call proposer.

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

The experimental NVIDIA Qwen3 8B NVFP4 arm was rejected on the tested SGLang
runtime. It left materially less allocator-visible reserve, produced weak draft
results, and emitted unmapped KV-scale warnings. There is no official Qwen3.8
8B checkpoint in the researched family, so it must not be represented as a
smaller form of the production Qwen3.8 model.

## Checkpointable training result

A disposable BF16 forward/backward/optimizer probe was used before any real
trainer. It supported a duty-cycle ceiling and checkpointed on SIGTERM.

At 25% duty, a synchronized 161-second run retained 46.34 GiB minimum
allocator-visible CUDA free and 103.23 GiB minimum host `MemAvailable`. It
reduced two-stream inference throughput 14.4% and increased p95 latency 18.4%.
At 10% duty, throughput fell 5.6% and p95 rose 8.3%. The lower-duty run reached
a slightly lower memory minimum, confirming that admission must use live probes
rather than infer reserve from duty percentage.

The forced yield checkpointed at step 1,658, exited successfully, and produced
a 268,438,483-byte checkpoint whose SHA-256 matched its manifest. A fresh
process verified the manifest, resumed from exactly step 1,658, and advanced to
step 1,668.

This certifies the disposable probe and validates the controller contract. It
does not certify any production trainer. A real trainer remains ineligible until
it produces the same immutable, hash-verified stop-time evidence.

## Provisional rollout envelope

For the first real-trainer canary:

1. Start at no more than the measured 10% training duty ceiling.
2. Admit only when host memory, short-lived CUDA allocatable memory, known
   consumers, inference queue/latency, and checkpoint readiness pass together.
3. Yield at a verified checkpoint boundary when p95 or queue pressure crosses
   the selected interactive SLO; do not wait for an OOM threshold.
4. Re-admit from a fresh sample generation and re-verify the checkpoint before
   resume.
5. Keep the 25% result as a tested comparison point, not the default.

Long-context, unique-prefix, and long-soak tiers remain required before
production promotion.

## Evidence integrity

Raw receipts are kept in the private operator evidence store because they
contain runtime and machine-specific paths. Key local harness hashes were:

- Agent/MCP benchmark: `6143d9ee05f6ea9148c9e427017169f84fe36cee760e40e70a191150bd53a736`
- Concurrency benchmark: `d0fd95550bc6ac59c71847178e03600cd8e8e898f5fe868ba6f0f2690f0669ef`
- Resource sampler: `8a3235a1926c203702e42d3d5b8c309f5135c837aab74811452537a976a31c4a`
- Checkpointable training probe: `bf57fa49a4ab5cb914a044d51fd504f7885371838c2d3ffd10b3723bc8085f75`

The model, runtime revision, prompt/corpus hashes, launch profile, resource
series, output validity, and checkpoint hashes are recorded in those receipts.
