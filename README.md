# Speculative Decoding on a Single Consumer GPU: When Does It Actually Pay Off?

A measurement study of vLLM's `draft_model` speculative decoding on a single RTX 5090, across concurrency levels 1–32, compared against both a matched baseline and true default vLLM.

**Headline finding:** at every concurrency level we tested, speculative decoding did not reliably beat true default vLLM. It shows a real, reproducible throughput gain against a *matched* baseline that shares its execution-path limitations — but that baseline itself carries a "runtime tax" relative to default vLLM, and the tax consistently outweighs (or roughly cancels) the gain.

---

## 1. Problem

Speculative decoding (SD) is generally expected to speed up autoregressive LLM decoding by using otherwise-idle GPU compute to verify several candidate tokens per step instead of one. Most published evaluations run on datacenter GPUs (A100/H100). We wanted a narrower, practically-relevant question:

> On a single consumer-grade GPU (RTX 5090), serving a fixed, modest-sized model pair, does turning on `draft_model` speculative decoding in vLLM actually increase throughput as concurrency scales — compared to what a user would get by just running default vLLM?

That last clause matters. It is easy to answer "does SD help" by comparing against a strawman baseline. We instead insist on comparing against **true default vLLM**, and we show why that distinction changes the answer.

## 2. Setup

| | Target model | Draft model | GPU |
|---|---|---|---|
| Model pair | Qwen3-14B-AWQ | Qwen3-0.6B | 1x RTX 5090 |

Speculative config: `method=draft_model`, `num_speculative_tokens=5` (K=5). `max_num_seqs=16`, `max_model_len=8192`, `gpu_memory_utilization=0.85`, prefix caching disabled, greedy decoding, fixed short prompt (~77 output tokens per response).

We define three server configurations, launched from `scripts/start_*.sh`:

- **A — default vLLM.** No speculative decoding, no manual overrides. This is what a user gets out of the box.
- **S — SpecDec.** Default vLLM + `draft_model` speculation, K=5.
- **B — matched baseline.** *Not* a real deployment configuration. It exists purely as a measurement control.

Why B exists: enabling `draft_model` speculation forces vLLM to fall back to an older execution path — the V1 model runner, with async scheduling disabled (`VLLM_USE_V2_MODEL_RUNNER=0`, `--no-async-scheduling`). This is an implementation side effect of this vLLM version, unrelated to the speculative-decoding algorithm itself. If we compared S directly to A, we would be conflating two effects: "does speculation help" and "does being forced onto an older execution path hurt." B isolates the second effect by manually replicating the same downgrade *without* turning speculation on, so **S vs B** isolates speculation's own contribution while **B vs A** isolates the downgrade's cost.

## 3. Measurement protocol

For each concurrency `c`, `run_sweep.py` orchestrates: local prewarm (outside the measured window) → start remote telemetry sampling (`scripts/telemetry.py`, ~0.2s interval, Prometheus `/metrics` scrape) → run the measured benchmark (`benchmark.py`, continuous-refill `ThreadPoolExecutor` with `c` workers, not wave-barrier) → stop telemetry. Each condition/concurrency was measured with `min-runs=64` (`c` ≤ 16) or more.

**Timing definitions** (computed per-request in `benchmark.py`):
- **TTFT**: wall-clock from just before the HTTP request opens to the first streamed token. The client sits in Korea, the server in mainland China, connected through an SSH tunnel — TTFT therefore includes cross-border round-trip and tunnel-setup latency, not just server-side time-to-first-token. Absolute TTFT/E2E values should not be read as server-internal latency; all conclusions here are relative comparisons within the same network path.
- **E2E**: wall-clock from request start to the final token.
- **TPOT**: `(E2E − TTFT) / (completion_tokens − 1)`. This is an *approximation* of average decode-step latency, not true inter-token latency (ITL) — it's a single average over the whole decode phase, not a per-token measurement. Guarded to return `None` (not 0) when `completion_tokens ≤ 1`, since the first token is prefill work, not decode.

**Speculative-decoding metrics** (Prometheus counters, `after − before` deltas around each measured window):
- `acceptance_pct = 100 × accepted_tokens / draft_tokens`
- `mean_acceptance_length = 1 + accepted_tokens / drafts` (the "+1" is the bonus token vLLM always emits per verification pass, matched or not)
- `K_observed = draft_tokens / drafts` (sanity check against the configured K=5)

**Queueing metric**: `Wait% = 100 × (# active samples with vllm:num_requests_waiting > 0) / (# active samples)`, where an "active" sample is one where `running > 0 or waiting > 0`. This is a **time-occupancy** statistic — "in what fraction of sampled instants was at least one request waiting" — **not** "what fraction of requests experienced queueing." The two are easy to conflate and mean different things.

**Independent reruns**: B and S were each independently re-launched (fresh server process) and re-measured twice more at c=4 and c=16 (`rerun1`, `rerun2`), in addition to the original sweep across c=1,2,4,8,16,32. A was originally measured only at c=1,4,16, then — after auditing the analysis code turned up a methodological gap (below) — independently re-launched and re-measured twice more at c=4 and c=16 to match B/S's repetition count.

**Known dates**: B/S original sweep: 2026-08-17. A original + B/S rerun1/rerun2: 2026-08-22. A rerun1/rerun2: 2026-08-24. No two conditions were ever measured in the same session/day — see Limitations §1.

## 4. Main result

**No concurrency level we tested shows speculative decoding reliably beating true default vLLM.**

We separate the effect into three layers.

### Layer ① — S vs B (execution path held fixed)

Isolates speculation's own contribution, since both conditions share the same forced downgrade.

| c | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| S vs B | -0.3% | +7.6% | **+13.2%** | +6.6%¹ | **-3.7%** | -0.8% |

Positive across c=2–8, peaking at c=4. Reversed to negative at c=16. Reproduced across three independent reruns:
- c=4: +13.2% / +5.5% / +13.5% (3/3 positive)
- c=16: -3.7% / -7.1% / -18.2% (3/3 negative)

¹ c=8 has only a single measurement, no rerun.

### Layer ② — B vs A (cost of the forced downgrade alone)

| c | 4 (3 reruns) | 16 (3 reruns) |
|---|---|---|
| B vs A | -16.3% / -4.5% / -23.7% | -23.9% / -11.1% / -13.6% |

B trails A by roughly 4–24%, in every single one of the six paired measurements. This is a pure cost of falling back to the older execution path — it has nothing to do with whether speculation is actually turned on.

### Layer ③ — S vs A (the number that actually matters to a deployer)

Computed by pairing each independent measurement of S against the *same-numbered* independent measurement of A (not by dividing every S value by one fixed A value — see the correction note below).

| c | Result |
|---|---|
| 1 | -19.7% (A measured once here; no pairing possible) |
| 4 | -5.2% / **+0.75%** / -13.4% — best case is roughly break-even, not a real win; 2 of 3 pairs are clearly negative |
| 16 | -26.8% / -17.4% / -29.3% — 3/3 negative, no exception |

**Correction note (2026-08-24):** an earlier version of this analysis reported "c=4 ties, within noise" by dividing all three S reruns by A's single original measurement. After independently re-measuring A twice more at c=4/16, we found A's own run-to-run spread (11–14%) is the same order of magnitude as B's (17–26%) — A was never as stable as assumed. Re-pairing S against the matching independent A measurement removes that bias, and the "ties" framing does not survive it. The corrected, more conservative conclusion is the one stated above.

![Runtime tax vs. SpecDec's own contribution](results/charts/chart2_relative_to_A.png)

*0 → B is the runtime tax. B → S is speculation's own contribution. 0 → S is the net effect. Restricted to c=4/16, the only concurrencies where A, B, and S were each independently repeated three times.*

![Throughput vs. concurrency](results/charts/chart1_tps_absolute.png)

*Absolute throughput for all three conditions, showing each condition's own run-to-run spread. A's line is dashed between c=4 and c=16 because c=8 was never measured for A — that segment is interpolated, not observed.*

![Speedup percentage, two panels](results/charts/chart3_speedup_pct.png)

*Left: S vs B across the full concurrency sweep (c=8 is a single measurement, marked separately, no rerun). Right: S vs A, restricted to c=4/16. Same-rerun points are connected to show the positive-to-negative crossover as a trajectory.*

## 5. Mechanism

Two lines of evidence explain *why* the sign flips with concurrency, and rule out the most obvious alternative explanation.

**Not a drafting-quality problem.** `acceptance_pct` (33.3%) and `mean_acceptance_length` (2.67) are constant across every concurrency level and every independent rerun, with zero exceptions. If higher concurrency degraded the draft model's hit rate, these numbers would drop as `c` increases. They don't — the draft-then-verify mechanism itself performs identically regardless of load.

**It's a compute-contention problem, and GPU utilization shows it directly.** At c=4, average GPU utilization is 47–70% across conditions — there is idle compute. At c=16, it climbs to 78–90%, close to saturated. Autoregressive decoding at low concurrency is typically memory-bandwidth-bound, not compute-bound: the GPU spends most of its time waiting on weight transfers, with compute units mostly idle. Speculative decoding's entire value proposition is to spend that otherwise-idle compute verifying multiple candidate tokens per step at near-zero marginal cost. Once concurrency is high enough to saturate compute, that idle capacity disappears, and the extra draft-model forward passes now compete directly with real batched work for the same limited compute — flipping the technique from free to costly.

This matches the general expectation in the speculative-decoding literature that gains concentrate at low batch sizes, and gives a concrete, measured crossover point for this specific consumer-GPU deployment — one point that most SD evaluations, run on higher-memory-bandwidth datacenter GPUs, don't report.

**Queueing detail:** `Wait%` is 0 at every c ≤ 16 sample across both B and S and all reruns — c=16's reversal is pure compute contention, no queueing involved. At c=32, `Wait%` rises sharply and `vllm_waiting_capacity` averages 8–9 (with `vllm_waiting_deferred` at 0 throughout) — this queueing is entirely due to hitting the `max_num_seqs=16` batch-size ceiling, a different mechanism than c=16's reversal, not a continuation of it.

## 6. Scope and Limitations

1. **Cross-date drift, and no two conditions were ever measured in the same session.** B/S original sweep: 2026-08-17. A original, B/S rerun1/rerun2: 2026-08-22. A rerun1/rerun2: 2026-08-24. Even the tag-matched pairing used in Layer ③ (A-rerun1 against S-rerun1, etc.) aligns measurements by *ordinal position*, not by *simultaneous measurement* — the "rerun1" label does not mean the same day across conditions. All three independent values are reported separately throughout this document; we do not collapse them into a mean or median.

2. **Run-to-run variance is real and comparable across A, B, and S** — this is itself a finding, not just noise to average away. B@c16: 964.0 / 1125.5 / 1217.9 tokens/sec (~26% spread). A@c16: 1267.1 / 1265.3 / 1409.4 (~11% spread). A@c4: 361.0 / 371.3 / 412.0 (~14% spread). A's stability was initially assumed and not measured; once measured, it turned out not to be materially more stable than B. Comparisons are made only within matched windows (same rerun tag), never by mixing absolute values across runs. (We separately observed "higher B correlates with more negative S-vs-B delta" across the three c=16 reruns — noted as an observation from 3 data points, not asserted as a general pattern.)

3. **The cause of this run-to-run variance is unknown.** We have not attributed it to GPU clocking/thermal behavior, host-level noise, or something else. This is flagged as future work rather than papered over with more repeats (see §7).

4. **Single workload.** Fixed short prompt, ~77 output tokens, TTFT is ~39% of E2E — decode accounts for a relatively small share of total request time. Speedup from decode-time optimization may be systematically diluted by a proportionally large, fixed prefill cost. See §7 for the output-length sweep that would test this directly.

5. **Cross-border network path.** Client in Korea, vLLM server in mainland China, connection through an SSH tunnel; the TTFT clock starts before the connection is established. Absolute TTFT/E2E therefore include cross-border round-trip and tunnel overhead and should not be read as server-internal latency. All comparisons in this report are relative, within the same fixed network path, which is held constant across A/B/S.

**Also note:** the KV-cache capacity reduction associated with enabling speculative decoding (observed capacity roughly halves, ~101,520 → ~54,192 token-slots, from reserving space for draft/verify buffers) is a separate, long-context deployment cost. Measured peak KV utilization in our workload never exceeded ~3% (B) / ~2.6% (S) even at c=32 — nowhere near this capacity limit — so it did not affect any result in this report. It is a real cost for a different (long-context) deployment scenario, and we call it out explicitly so it isn't confused with anything in §4–§5.

## 7. Future work

- **K sweep** (K=1/3/7 × c=1/4/16). K directly controls the tradeoff between the runtime tax and speculation's own contribution. Open question: does the optimal K shift with concurrency, and can a smaller K push S ahead of A at c=4 rather than merely tying it?
- **Output-length sweep.** Current decode share of total latency is low (§6.4). Hypothesis: longer outputs amortize the fixed downgrade tax over more decode steps, potentially flipping S vs A positive at low concurrency even though it doesn't here. If confirmed, the conclusion sharpens from "not worth it on this hardware" to "not worth it for short-output workloads specifically."
- **Run-to-run variance attribution.** A/B/S all show 10–26% run-to-run throughput variance with an unknown cause. Rather than averaging it away with more repeats, the higher-value next step is instrumenting GPU clock/temperature during measurement to see whether it correlates.
- **Kernel-level mechanism verification.** The GPU-utilization explanation in §5 is built from exclusion (ruling out drafting-quality degradation) plus an aggregate utilization metric, not a kernel-level profiling trace through vLLM's scheduler and model runner. A profiler-based causal chain (scheduler decision → execution behavior → GPU metric → crossover) would confirm it directly.
- Additional draft-model sizes, task diversity (chat / summarization / code), adaptive-K strategies.

## 8. Known issues found during code audit

- **KV-cache column name bug** in `archive/summarize_specdec_formal.py` and `archive/compare_b_vs_specdec_formal.py` (superseded, kept in `archive/` for the record): both read a telemetry column named `vllm_kv_cache_usage_perc`, but `telemetry.py` actually writes `vllm_kv_cache_usage` (no `_perc` suffix). The lookup silently returns `None` on the missing key, so `kv_max_pct` has been empty in every summary CSV those two scripts ever produced. **Not fixed in those two files** — `summarize_crossover.py` (used for everything in this report) uses the correct column name. Anyone re-running the older scripts should be aware `kv_max_pct` from them is not trustworthy.

## 9. Repository layout

```
scripts/start_default.sh          # A: true default vLLM
scripts/start_matched_baseline.sh # B: matched baseline (V1 runner, no async)
scripts/start_specdec.sh          # S: default + draft_model speculation, K=5
scripts/telemetry.py              # remote GPU/vLLM metrics sampler
scripts/telemetry_controller.py   # start/stop wrapper invoked by run_sweep.py

benchmark.py        # client-side request/timing harness
run_sweep.py         # orchestrates prewarm -> telemetry -> benchmark per concurrency
summarize_crossover.py  # aggregates results/{summary,telemetry} -> results/crossover_summary.csv
plot_charts.py       # generates results/charts/chart{1,2,3}_*.png from crossover_summary.csv

results/summary/     # per-run benchmark JSON summaries
results/telemetry/   # per-run Prometheus before/after snapshots + sampled CSV
results/charts/       # the three charts referenced in this document
results/crossover_summary.csv  # single flat table backing all numbers and charts above

archive/              # superseded scripts, kept for the record (not used by the current pipeline)
```

To regenerate everything from raw results: `python summarize_crossover.py && python plot_charts.py`.
