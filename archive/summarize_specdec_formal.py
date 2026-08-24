#!/usr/bin/env python3
import csv
import glob
import json
import re
import statistics
from pathlib import Path

LABEL = "specdec_k5_formal"
CONCURRENCIES = [1, 2, 4, 8, 16, 32]
SUMMARY_DIR = Path("results/summary")
TELEMETRY_DIR = Path("results/telemetry")
OUT = Path("results/specdec_k5_formal_summary.csv")

PROM_METRICS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}

def latest_summary(c):
    files = sorted(SUMMARY_DIR.glob(f"{LABEL}_c{c}_*.json"))
    if not files:
        raise FileNotFoundError(f"missing benchmark summary for c={c}")
    return files[-1]

def read_prom(path):
    vals = {}
    text = Path(path).read_text()
    for key, metric in PROM_METRICS.items():
        value = None
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                value = float(line.rsplit(" ", 1)[1])
                break
        if value is None:
            raise RuntimeError(f"{metric} missing from {path}")
        vals[key] = value
    return vals

def q95(values):
    if not values:
        return None
    xs = sorted(values)
    idx = max(0, min(len(xs)-1, int((0.95 * len(xs) + 0.999999999) - 1)))
    return xs[idx]

def read_telemetry(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def num(row, key):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    active = []
    for r in rows:
        running = num(r, "vllm_running") or 0.0
        waiting = num(r, "vllm_waiting") or 0.0
        if running > 0 or waiting > 0:
            active.append(r)

    if not active:
        return {
            "gpu_avg": None, "gpu_p95": None,
            "running_avg": None, "running_max": None,
            "waiting_avg": None, "waiting_max": None,
            "wait_active_pct": None, "kv_max_pct": None,
            "power_avg_w": None, "active_samples": 0,
        }

    def vals(key):
        return [v for r in active if (v := num(r, key)) is not None]

    gpu = vals("gpu_util_pct")
    running = vals("vllm_running")
    waiting = vals("vllm_waiting")
    kv = vals("vllm_kv_cache_usage_perc")
    power = vals("power_w")

    # tolerate alternate NVML column names if telemetry.py used them
    if not gpu:
        gpu = vals("gpu_util")
    if not power:
        power = vals("gpu_power_w")

    return {
        "gpu_avg": statistics.mean(gpu) if gpu else None,
        "gpu_p95": q95(gpu),
        "running_avg": statistics.mean(running) if running else None,
        "running_max": max(running) if running else None,
        "waiting_avg": statistics.mean(waiting) if waiting else None,
        "waiting_max": max(waiting) if waiting else None,
        "wait_active_pct": (100.0 * sum(1 for x in waiting if x > 0) / len(waiting)) if waiting else None,
        "kv_max_pct": (100.0 * max(kv)) if kv and max(kv) <= 1.5 else (max(kv) if kv else None),
        "power_avg_w": statistics.mean(power) if power else None,
        "active_samples": len(active),
    }

rows = []

for c in CONCURRENCIES:
    sp = latest_summary(c)
    x = json.loads(sp.read_text())

    before = read_prom(TELEMETRY_DIR / f"{LABEL}_c{c}_before.prom")
    after = read_prom(TELEMETRY_DIR / f"{LABEL}_c{c}_after.prom")
    delta = {k: after[k] - before[k] for k in PROM_METRICS}

    drafts = delta["drafts"]
    draft_tokens = delta["draft_tokens"]
    accepted = delta["accepted_tokens"]

    tel = read_telemetry(TELEMETRY_DIR / f"{LABEL}_c{c}.csv")

    row = {
        "concurrency": c,
        "successful_runs": x["successful_runs"],
        "failed_runs": x["failed_runs"],
        "ttft_p50_ms": x["ttft_s"]["median_p50"] * 1000,
        "ttft_p95_ms": x["ttft_s"]["p95"] * 1000,
        "e2e_p50_ms": x["e2e_s"]["median_p50"] * 1000,
        "e2e_p95_ms": x["e2e_s"]["p95"] * 1000,
        "tpot_p50_ms": x["tpot_s"]["median_p50"] * 1000,
        "tpot_p95_ms": x["tpot_s"]["p95"] * 1000,
        "rps": x["request_throughput_rps"],
        "output_tps": x["output_token_throughput_tps"],
        "draft_rounds": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "K_observed": (draft_tokens / drafts) if drafts else None,
        "acceptance_rate_pct": (100.0 * accepted / draft_tokens) if draft_tokens else None,
        "mean_acceptance_length": (1.0 + accepted / drafts) if drafts else None,
        **tel,
        "benchmark_summary_file": str(sp),
    }
    rows.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nWrote: {OUT}\n")
print(
    f"{'C':>3} {'TPS':>9} {'TTFT50':>9} {'E2E50':>9} {'TPOT50':>9} "
    f"{'Acc%':>7} {'MeanLen':>8} {'GPUavg':>8} {'Wait%':>7}"
)
for r in rows:
    def fmt(v, nd=1):
        return "-" if v is None else f"{v:.{nd}f}"
    print(
        f"{r['concurrency']:>3} "
        f"{r['output_tps']:>9.1f} "
        f"{r['ttft_p50_ms']:>9.1f} "
        f"{r['e2e_p50_ms']:>9.1f} "
        f"{r['tpot_p50_ms']:>9.2f} "
        f"{fmt(r['acceptance_rate_pct']):>7} "
        f"{fmt(r['mean_acceptance_length'], 2):>8} "
        f"{fmt(r['gpu_avg']):>8} "
        f"{fmt(r['wait_active_pct']):>7}"
    )
