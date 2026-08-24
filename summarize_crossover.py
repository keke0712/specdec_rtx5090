#!/usr/bin/env python3
"""
Summarize A (default vLLM) / B (matched baseline) / S (SpecDec K=5) across
multiple concurrencies and multiple independent server reruns.

Fixes a bug found in summarize_specdec_formal.py / compare_b_vs_specdec_formal.py:
those files read telemetry column "vllm_kv_cache_usage_perc", but telemetry.py
actually writes the column as "vllm_kv_cache_usage" (no "_perc" suffix). Because
the lookup silently returns None on a missing key, kv_max_pct has always been
empty in every previously produced summary CSV. This script uses the correct
column name.

Also surfaces two telemetry columns that were being recorded but never read
anywhere: vllm_waiting_capacity / vllm_waiting_deferred (vLLM's own breakdown
of *why* a request is waiting: queue-capacity-limited vs deferred for other
scheduling reasons).
"""
import csv
import json
import math
import statistics
from pathlib import Path

SUMMARY_DIR = Path("results/summary")
TELEMETRY_DIR = Path("results/telemetry")
OUT = Path("results/crossover_summary.csv")

PROM_METRICS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}

# (condition, label, concurrencies to include, has speculative decoding)
RUNS = [
    ("A", "default_vllm_formal", [1, 4, 16], False),
    ("A", "default_vllm_formal_rerun1", [4, 16], False),
    ("A", "default_vllm_formal_rerun2", [4, 16], False),
    ("B", "matched_baseline_formal", [1, 2, 4, 8, 16, 32], False),
    ("B", "matched_baseline_formal_rerun1", [4, 16], False),
    ("B", "matched_baseline_formal_rerun2", [4, 16], False),
    ("S", "specdec_k5_formal", [1, 2, 4, 8, 16, 32], True),
    ("S", "specdec_k5_formal_rerun1", [4, 16], True),
    ("S", "specdec_k5_formal_rerun2", [4, 16], True),
]


def latest_summary(label, c):
    files = sorted(SUMMARY_DIR.glob(f"{label}_c{c}_*.json"))
    return files[-1] if files else None


def read_json_summary(label, c):
    p = latest_summary(label, c)
    if p is None:
        return None
    x = json.loads(p.read_text())

    def ms(block, key):
        if not block:
            return None
        v = block.get(key)
        return v * 1000 if v is not None else None

    return {
        "summary_file": str(p),
        "successful_runs": x.get("successful_runs"),
        "failed_runs": x.get("failed_runs"),
        "ttft_p50_ms": ms(x.get("ttft_s"), "median_p50"),
        "ttft_p95_ms": ms(x.get("ttft_s"), "p95"),
        "e2e_p50_ms": ms(x.get("e2e_s"), "median_p50"),
        "e2e_p95_ms": ms(x.get("e2e_s"), "p95"),
        "tpot_p50_ms": ms(x.get("tpot_s"), "median_p50"),
        "tpot_p95_ms": ms(x.get("tpot_s"), "p95"),
        "rps": x.get("request_throughput_rps"),
        "tps": x.get("output_token_throughput_tps"),
    }


def read_prom(path):
    if not path.exists():
        return None
    vals = {}
    text = path.read_text()
    for key, metric in PROM_METRICS.items():
        value = None
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                value = float(line.rsplit(" ", 1)[1])
                break
        vals[key] = value
    return vals


def read_acceptance(label, c):
    before = read_prom(TELEMETRY_DIR / f"{label}_c{c}_before.prom")
    after = read_prom(TELEMETRY_DIR / f"{label}_c{c}_after.prom")

    empty = {
        "draft_rounds": None, "draft_tokens": None, "accepted_tokens": None,
        "K_observed": None, "acceptance_pct": None, "mean_acceptance_length": None,
    }
    if before is None or after is None:
        return empty

    delta = {}
    for k in PROM_METRICS:
        b, a = before.get(k), after.get(k)
        delta[k] = (a - b) if (a is not None and b is not None) else None

    drafts = delta.get("drafts")
    draft_tokens = delta.get("draft_tokens")
    accepted = delta.get("accepted_tokens")

    return {
        "draft_rounds": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "K_observed": (draft_tokens / drafts) if drafts else None,
        "acceptance_pct": (100.0 * accepted / draft_tokens) if draft_tokens else None,
        "mean_acceptance_length": (1.0 + accepted / drafts) if drafts else None,
    }


def q95(values):
    if not values:
        return None
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1))
    return xs[idx]


def read_telemetry(label, c):
    p = TELEMETRY_DIR / f"{label}_c{c}.csv"
    if not p.exists():
        return None

    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))

    def num(row, key):
        v = row.get(key)
        if v in (None, ""):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    active = []
    for r in rows:
        running = num(r, "vllm_running") or 0.0
        waiting = num(r, "vllm_waiting") or 0.0
        if running > 0 or waiting > 0:
            active.append(r)

    empty = {
        "gpu_avg": None, "gpu_p95": None,
        "waiting_avg": None, "waiting_max": None, "wait_active_pct": None,
        "waiting_capacity_avg": None, "waiting_deferred_avg": None,
        "kv_max_pct": None, "power_avg_w": None, "active_samples": 0,
    }
    if not active:
        return empty

    def vals(key):
        out = []
        for r in active:
            v = num(r, key)
            if v is not None:
                out.append(v)
        return out

    gpu = vals("gpu_util_pct")
    waiting = vals("vllm_waiting")
    waiting_capacity = vals("vllm_waiting_capacity")
    waiting_deferred = vals("vllm_waiting_deferred")
    kv = vals("vllm_kv_cache_usage")  # NOTE: correct column name (no "_perc")
    power = vals("power_w")

    kv_max = max(kv) if kv else None
    kv_max_pct = (kv_max * 100.0) if (kv_max is not None and kv_max <= 1.5) else kv_max

    return {
        "gpu_avg": statistics.mean(gpu) if gpu else None,
        "gpu_p95": q95(gpu),
        "waiting_avg": statistics.mean(waiting) if waiting else None,
        "waiting_max": max(waiting) if waiting else None,
        "wait_active_pct": (100.0 * sum(1 for x in waiting if x > 0) / len(waiting)) if waiting else None,
        "waiting_capacity_avg": statistics.mean(waiting_capacity) if waiting_capacity else None,
        "waiting_deferred_avg": statistics.mean(waiting_deferred) if waiting_deferred else None,
        "kv_max_pct": kv_max_pct,
        "power_avg_w": statistics.mean(power) if power else None,
        "active_samples": len(active),
    }


rows = []
for condition, label, concurrencies, has_specdec in RUNS:
    for c in concurrencies:
        js = read_json_summary(label, c)
        if js is None:
            print(f"[skip] missing benchmark summary: {label} c={c}")
            continue

        tel = read_telemetry(label, c)
        if tel is None:
            print(f"[warn] missing telemetry csv: {label} c={c}")
            tel = {}

        acc = read_acceptance(label, c) if has_specdec else {
            "draft_rounds": None, "draft_tokens": None, "accepted_tokens": None,
            "K_observed": None, "acceptance_pct": None, "mean_acceptance_length": None,
        }

        row = {
            "condition": condition,
            "label": label,
            "concurrency": c,
            **js,
            **tel,
            **acc,
        }
        rows.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
if rows:
    fieldnames = list(rows[0].keys())
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote: {OUT}\n")

print(
    f"{'C':>3} {'cond':<5} {'label':<32} {'TPS':>8} {'TTFT50':>8} "
    f"{'Acc%':>7} {'MeanLen':>8} {'GPU%':>6} {'Wait%':>7} {'WaitCap':>8} {'WaitDef':>8}"
)


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


for r in sorted(rows, key=lambda r: (r["concurrency"], r["condition"], r["label"])):
    print(
        f"{r['concurrency']:>3} {r['condition']:<5} {r['label']:<32} "
        f"{fmt(r.get('tps')):>8} {fmt(r.get('ttft_p50_ms')):>8} "
        f"{fmt(r.get('acceptance_pct')):>7} {fmt(r.get('mean_acceptance_length'), 2):>8} "
        f"{fmt(r.get('gpu_avg')):>6} {fmt(r.get('wait_active_pct')):>7} "
        f"{fmt(r.get('waiting_capacity_avg'), 2):>8} {fmt(r.get('waiting_deferred_avg'), 2):>8}"
    )
