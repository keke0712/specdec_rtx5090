import argparse
import csv
import os
import time
import urllib.request
from datetime import datetime

import pynvml


METRICS_URL = "http://127.0.0.1:8000/metrics"


def fetch_metrics():
    with urllib.request.urlopen(METRICS_URL, timeout=2) as r:
        return r.read().decode("utf-8")


def metric_value(text, metric, contains=None):
    for line in text.splitlines():
        if line.startswith("#"):
            continue

        if not line.startswith(metric):
            continue

        if contains is not None and contains not in line:
            continue

        try:
            return float(line.rsplit(" ", 1)[1])
        except Exception:
            return None

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    stem, _ = os.path.splitext(args.out)
    before_path = stem + "_before.prom"
    after_path = stem + "_after.prom"

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    # 保存实验开始前的完整 vLLM metrics
    before = fetch_metrics()
    with open(before_path, "w") as f:
        f.write(before)

    fields = [
        "timestamp_unix_s",
        "elapsed_s",
        "gpu_util_pct",
        "memory_used_mib",
        "power_w",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "temperature_c",
        "vllm_running",
        "vllm_waiting",
        "vllm_waiting_capacity",
        "vllm_waiting_deferred",
        "vllm_kv_cache_usage",
        "vllm_preemptions_total",
    ]

    start = time.perf_counter()

    print(f"Telemetry started -> {args.out}")
    print("Press Ctrl+C after benchmark finishes.")

    try:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            while True:
                loop_start = time.perf_counter()

                util = pynvml.nvmlDeviceGetUtilizationRates(gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(gpu)

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000.0
                except Exception:
                    power = None

                try:
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(
                        gpu, pynvml.NVML_CLOCK_SM
                    )
                except Exception:
                    sm_clock = None

                try:
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(
                        gpu, pynvml.NVML_CLOCK_MEM
                    )
                except Exception:
                    mem_clock = None

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        gpu, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = None

                metrics = fetch_metrics()

                row = {
                    "timestamp_unix_s": time.time(),
                    "elapsed_s": time.perf_counter() - start,
                    "gpu_util_pct": util.gpu,
                    "memory_used_mib": mem.used / 1024 / 1024,
                    "power_w": power,
                    "sm_clock_mhz": sm_clock,
                    "mem_clock_mhz": mem_clock,
                    "temperature_c": temp,
                    "vllm_running": metric_value(
                        metrics, "vllm:num_requests_running"
                    ),
                    "vllm_waiting": metric_value(
                        metrics, "vllm:num_requests_waiting"
                    ),
                    "vllm_waiting_capacity": metric_value(
                        metrics,
                        "vllm:num_requests_waiting_by_reason",
                        'reason="capacity"',
                    ),
                    "vllm_waiting_deferred": metric_value(
                        metrics,
                        "vllm:num_requests_waiting_by_reason",
                        'reason="deferred"',
                    ),
                    "vllm_kv_cache_usage": metric_value(
                        metrics, "vllm:kv_cache_usage_perc"
                    ),
                    "vllm_preemptions_total": metric_value(
                        metrics, "vllm:num_preemptions_total"
                    ),
                }

                writer.writerow(row)
                f.flush()

                spent = time.perf_counter() - loop_start
                time.sleep(max(0, args.interval - spent))

    except KeyboardInterrupt:
        pass

    finally:
        after = fetch_metrics()
        with open(after_path, "w") as f:
            f.write(after)

        pynvml.nvmlShutdown()

        print()
        print("Telemetry stopped.")
        print("CSV:   ", args.out)
        print("Before:", before_path)
        print("After: ", after_path)


if __name__ == "__main__":
    main()
