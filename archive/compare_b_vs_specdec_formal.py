#!/usr/bin/env python3
import csv
import json
import math
import statistics
from pathlib import Path

B_LABEL = "matched_baseline_formal"
S_LABEL = "specdec_k5_formal"
CONCURRENCIES = [1, 2, 4, 8, 16, 32]

SUMMARY_DIR = Path("results/summary")
TELEMETRY_DIR = Path("results/telemetry")
OUT = Path("results/b_vs_specdec_k5_formal.csv")

PROM_METRICS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def latest_summary(label, c):
    files = sorted(SUMMARY_DIR.glob(f"{label}_c{c}_*.json"))
    if not files:
        raise FileNotFoundError(f"missing summary: {label} c={c}")
    return files[-1]


def read_json_summary(label, c):
    p = latest_summary(label, c)
    x = json.loads(p.read_text())
    return {
        "file": str(p),
        "success": x["successful_runs"],
        "fail": x["failed_runs"],
        "ttft50": x["ttft_s"]["median_p50"] * 1000,
        "ttft95": x["ttft_s"]["p95"] * 1000,
        "e2e50": x["e2e_s"]["median_p50"] * 1000,
        "e2e95": x["e2e_s"]["p95"] * 1000,
        "tpot50": x["tpot_s"]["median_p50"] * 1000,
        "tpot95": x["tpot_s"]["p95"] * 1000,
        "rps": x["request_throughput_rps"],
        "tps": x["output_token_throughput_tps"],
    }


def q95(values):
    if not values:
        return None
    xs = sorted(values)
    idx = math.ceil(0.95 * len(xs)) - 1
    idx = max(0, min(idx, len(xs)-1))
    return xs[idx]


def read_telemetry(label, c):
    p = TELEMETRY_DIR / f"{label}_c{c}.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing telemetry: {p}")

    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))

    def num(row, *keys):
        for key in keys:
            try:
                if key in row and row[key] not in ("", None):
                    return float(row[key])
            except ValueError:
                pass
        return None

    active = []
    for r in rows:
        running = num(r, "vllm_running") or 0.0
        waiting = num(r, "vllm_waiting") or 0.0
        if running > 0 or waiting > 0:
            active.append(r)

    def vals(*keys):
        out = []
        for r in active:
            v = num(r, *keys)
            if v is not None:
                out.append(v)
        return out

    gpu = vals("gpu_util_pct", "gpu_util")
    running = vals("vllm_running")
    waiting = vals("vllm_waiting")
    kv = vals("vllm_kv_cache_usage_perc")
    power = vals("power_w", "gpu_power_w")

    kvmax = max(kv) if kv else None
    if kvmax is not None and kvmax <= 1.5:
        kvmax *= 100.0

    return {
        "gpu_avg": statistics.mean(gpu) if gpu else None,
        "gpu_p95": q95(gpu),
        "run_avg": statistics.mean(running) if running else None,
        "run_max": max(running) if running else None,
        "wait_avg": statistics.mean(waiting) if waiting else None,
        "wait_max": max(waiting) if waiting else None,
        "wait_pct": (
            100.0 * sum(1 for x in waiting if x > 0) / len(waiting)
            if waiting else None
        ),
        "kv_max_pct": kvmax,
        "power_avg_w": statistics.mean(power) if power else None,
        "active_samples": len(active),
    }


def read_prom(path):
    vals = {}
    text = Path(path).read_text()
    for key, metric in PROM_METRICS.items():
        found = None
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                found = float(line.rsplit(" ", 1)[1])
                break
        if found is None:
            raise RuntimeError(f"{metric} missing from {path}")
        vals[key] = found
    return vals


def read_acceptance(c):
    before = read_prom(TELEMETRY_DIR / f"{S_LABEL}_c{c}_before.prom")
    after = read_prom(TELEMETRY_DIR / f"{S_LABEL}_c{c}_after.prom")
    d = {k: after[k] - before[k] for k in PROM_METRICS}
    return {
        "acc_pct": 100.0 * d["accepted_tokens"] / d["draft_tokens"] if d["draft_tokens"] else None,
        "mean_len": 1.0 + d["accepted_tokens"] / d["drafts"] if d["drafts"] else None,
        "k_obs": d["draft_tokens"] / d["drafts"] if d["drafts"] else None,
    }


def pct_change(s, b):
    return 100.0 * (s / b - 1.0)


rows = []
for c in CONCURRENCIES:
    b = read_json_summary(B_LABEL, c)
    s = read_json_summary(S_LABEL, c)
    bt = read_telemetry(B_LABEL, c)
    st = read_telemetry(S_LABEL, c)
    acc = read_acceptance(c)

    row = {
        "concurrency": c,
        "B_TPS": b["tps"],
        "S_TPS": s["tps"],
        "S_vs_B_TPS_pct": pct_change(s["tps"], b["tps"]),
        "B_TTFT50_ms": b["ttft50"],
        "S_TTFT50_ms": s["ttft50"],
        "S_vs_B_TTFT50_pct": pct_change(s["ttft50"], b["ttft50"]),
        "B_E2E50_ms": b["e2e50"],
        "S_E2E50_ms": s["e2e50"],
        "S_vs_B_E2E50_pct": pct_change(s["e2e50"], b["e2e50"]),
        "B_TPOT50_ms": b["tpot50"],
        "S_TPOT50_ms": s["tpot50"],
        "S_vs_B_TPOT50_pct": pct_change(s["tpot50"], b["tpot50"]),
        "B_GPU_avg": bt["gpu_avg"],
        "S_GPU_avg": st["gpu_avg"],
        "B_Wait_pct": bt["wait_pct"],
        "S_Wait_pct": st["wait_pct"],
        "B_Power_avg_W": bt["power_avg_w"],
        "S_Power_avg_W": st["power_avg_w"],
        "S_acceptance_pct": acc["acc_pct"],
        "S_mean_acceptance_length": acc["mean_len"],
        "S_K_observed": acc["k_obs"],
        "B_summary_file": b["file"],
        "S_summary_file": s["file"],
    }
    rows.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nWrote: {OUT}\n")
print(
    f"{'C':>3} "
    f"{'B TPS':>9} {'S TPS':>9} {'TPS Δ%':>8} "
    f"{'B TTFT':>9} {'S TTFT':>9} "
    f"{'B E2E':>9} {'S E2E':>9} "
    f"{'B TPOT':>9} {'S TPOT':>9} "
    f"{'B GPU':>7} {'S GPU':>7} "
    f"{'B Wait':>8} {'S Wait':>8}"
)
for r in rows:
    print(
        f"{r['concurrency']:>3} "
        f"{r['B_TPS']:>9.1f} {r['S_TPS']:>9.1f} {r['S_vs_B_TPS_pct']:>8.1f} "
        f"{r['B_TTFT50_ms']:>9.1f} {r['S_TTFT50_ms']:>9.1f} "
        f"{r['B_E2E50_ms']:>9.1f} {r['S_E2E50_ms']:>9.1f} "
        f"{r['B_TPOT50_ms']:>9.2f} {r['S_TPOT50_ms']:>9.2f} "
        f"{(r['B_GPU_avg'] if r['B_GPU_avg'] is not None else float('nan')):>7.1f} "
        f"{(r['S_GPU_avg'] if r['S_GPU_avg'] is not None else float('nan')):>7.1f} "
        f"{(r['B_Wait_pct'] if r['B_Wait_pct'] is not None else float('nan')):>8.1f} "
        f"{(r['S_Wait_pct'] if r['S_Wait_pct'] is not None else float('nan')):>8.1f}"
    )

print("\nInterpretation rule:")
print("  TPS Δ% > 0  => SpecDec throughput is better")
print("  TTFT/E2E/TPOT Δ% < 0 => SpecDec latency is better")
