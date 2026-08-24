from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


# =========================
# 基本配置
# =========================

URL = os.getenv(
    "VLLM_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)

TIMEOUT = float(
    os.getenv("VLLM_TIMEOUT", "60")
)

MODEL = "Qwen/Qwen3-14B-AWQ"

PROMPT = "用三句话解释大语言模型推理中的投机解码。"

MAX_TOKENS = 128

RAW_DIR = Path("results/raw")
SUMMARY_DIR = Path("results/summary")

RAW_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 单次请求
# =========================

def run_once(run_type: str, run_index: int):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": True,

        # 让最后一个 streaming chunk 返回 token usage
        "stream_options": {
            "include_usage": True,
        },

        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        URL,
        data=body,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    wall_start = datetime.now().isoformat(timespec="milliseconds")
    start = time.perf_counter()

    first_token_time = None
    end_time = None

    chunk_count = 0
    completion_tokens = None
    prompt_tokens = None
    total_tokens = None

    generated_text = ""

    status = "success"
    error_message = None

    try:
        with urllib.request.urlopen(
            request,
            timeout=TIMEOUT,
        ) as response:

            for raw_line in response:

                line = raw_line.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                if not line:
                    continue

                if not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()

                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # -------------------------
                # usage
                # -------------------------

                usage = obj.get("usage")

                if usage:
                    prompt_tokens = usage.get(
                        "prompt_tokens"
                    )

                    completion_tokens = usage.get(
                        "completion_tokens"
                    )

                    total_tokens = usage.get(
                        "total_tokens"
                    )

                # -------------------------
                # generated content
                # -------------------------

                choices = obj.get(
                    "choices",
                    [],
                )

                if not choices:
                    continue

                delta = choices[0].get(
                    "delta",
                    {},
                )

                content = delta.get(
                    "content"
                )

                if not content:
                    continue

                now = time.perf_counter()

                if first_token_time is None:
                    first_token_time = now

                chunk_count += 1
                generated_text += content

        end_time = time.perf_counter()

    except urllib.error.HTTPError as e:
        end_time = time.perf_counter()
        status = "http_error"

        try:
            error_message = e.read().decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            error_message = str(e)

    except urllib.error.URLError as e:
        end_time = time.perf_counter()
        status = "connection_error"
        error_message = str(e.reason)

    except TimeoutError as e:
        end_time = time.perf_counter()
        status = "timeout"
        error_message = str(e)

    except Exception as e:
        end_time = time.perf_counter()
        status = "unknown_error"
        error_message = repr(e)


    # =========================
    # 计算指标
    # =========================

    e2e = (
        end_time - start
        if end_time is not None
        else None
    )

    ttft = (
        first_token_time - start
        if first_token_time is not None
        else None
    )

    tpot = None

    if (
        status == "success"
        and ttft is not None
        and e2e is not None
        and completion_tokens is not None
        and completion_tokens > 1
    ):
        tpot = (
            e2e - ttft
        ) / (
            completion_tokens - 1
        )


    result = {
        "run_type": run_type,
        "run_index": run_index,

        "status": status,
        "error": error_message,

        "wall_start": wall_start,

        "model": MODEL,
        "prompt": PROMPT,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,

        "ttft_s": ttft,
        "e2e_s": e2e,
        "tpot_s": tpot,

        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,

        "sse_chunk_count": chunk_count,

        "generated_text": generated_text,
    }

    return result


# =========================
# percentile
# =========================

def percentile(values, p):
    if not values:
        return None

    ordered = sorted(values)

    rank = math.ceil(
        (p / 100) * len(ordered)
    )

    index = max(
        0,
        rank - 1,
    )

    return ordered[index]


# =========================
# 一组指标的统计
# =========================

def summarize_metric(values):

    values = [
        x for x in values
        if x is not None
    ]

    if not values:
        return None

    return {
        "count": len(values),

        "mean": statistics.mean(
            values
        ),

        "median_p50": statistics.median(
            values
        ),

        "p95": percentile(
            values,
            95,
        ),

        "min": min(
            values
        ),

        "max": max(
            values
        ),

        "std": (
            statistics.stdev(values)
            if len(values) >= 2
            else 0.0
        ),
    }


# =========================
# benchmark
# =========================
def main():

    # =========================
    # 命令行参数
    # =========================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--warmup-waves",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--measure-waves",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-runs",
        type=int,
        default=64,
    )

    parser.add_argument(
    "--label",
    type=str,
    default="baseline",
    )

    args = parser.parse_args()


    # =========================
    # 根据 concurrency 自动算请求数
    # =========================

    warmup_runs = (
        args.warmup_waves
        * args.concurrency
    )

    min_runs_aligned = (
        math.ceil(
            args.min_runs
            / args.concurrency
        )
        * args.concurrency
    )

    measured_runs = max(
        args.measure_waves
        * args.concurrency,
        min_runs_aligned,
    )

    actual_measured_waves = (
        measured_runs
        / args.concurrency
    )


    # =========================
    # Experiment ID
    # =========================

    experiment_id = (
        f"{args.label}_c{args.concurrency}_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )


    print(
        f"\nExperiment: {experiment_id}"
    )

    print(
        f"Concurrency: {args.concurrency}"
    )

    print(
        f"Warm-up waves: {args.warmup_waves}"
    )

    print(
        f"Warm-up runs: {warmup_runs}"
    )

    print(
        f"Measured waves: {actual_measured_waves}"
    )

    print(
        f"Measured runs: {measured_runs}"
    )

    print(
        f"URL: {URL}\n"
    )


    # =========================
    # Warm-up
    # =========================

    print("=== Warm-up ===")

    warmup_results = []

    with ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:

        futures = {
            executor.submit(
                run_once,
                "warmup",
                i,
            ): i

            for i in range(
                1,
                warmup_runs + 1,
            )
        }

        for future in as_completed(futures):

            result = future.result()

            warmup_results.append(
                result
            )

            print(
                f"[warmup {result['run_index']:02d}] "
                f"status={result['status']} "
                f"TTFT={result['ttft_s']} "
                f"E2E={result['e2e_s']}"
            )

    warmup_results.sort(
        key=lambda r: r["run_index"]
    )


    # =========================
    # 正式测量
    # =========================

    print("\n=== Measured Runs ===")

    measured_results = []

    measured_start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:

        futures = {
            executor.submit(
                run_once,
                "measured",
                i,
            ): i

            for i in range(
                1,
                measured_runs + 1,
            )
        }

        for future in as_completed(futures):

            result = future.result()

            measured_results.append(
                result
            )

            print(
                f"[run {result['run_index']:02d}] "
                f"status={result['status']} "
                f"TTFT={result['ttft_s']} "
                f"E2E={result['e2e_s']} "
                f"tokens={result['completion_tokens']}"
            )

    measured_end = time.perf_counter()

    measured_results.sort(
        key=lambda r: r["run_index"]
    )

    measured_wall_time = (
        measured_end
        - measured_start
    )


    # =========================
    # 保存 raw
    # =========================

    raw_path = (
        RAW_DIR
        / f"{experiment_id}.json"
    )

    raw_data = {
        "experiment_id": experiment_id,

        "config": {
            "url": URL,
            "model": MODEL,
            "prompt": PROMPT,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,

            "concurrency":
                args.concurrency,

            "warmup_waves":
                args.warmup_waves,

            "warmup_runs":
                warmup_runs,

            "measure_waves_requested":
                args.measure_waves,

            "measured_waves_actual":
                actual_measured_waves,

            "measured_runs":
                measured_runs,

            "min_runs":
                args.min_runs,
        },

        "warmup_results":
            warmup_results,

        "measured_results":
            measured_results,
    }

    with raw_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            raw_data,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # =========================
    # 成功 / 失败请求
    # =========================

    successful = [
        r
        for r in measured_results
        if r["status"] == "success"
    ]

    failed = [
        r
        for r in measured_results
        if r["status"] != "success"
    ]


    # =========================
    # 提取 metrics
    # =========================

    ttft_values = [
        r["ttft_s"]
        for r in successful
    ]

    e2e_values = [
        r["e2e_s"]
        for r in successful
    ]

    tpot_values = [
        r["tpot_s"]
        for r in successful
        if r["tpot_s"] is not None
    ]


    total_output_tokens = sum(
        r["completion_tokens"] or 0
        for r in successful
    )


    # =========================
    # Throughput
    # =========================

    request_throughput = (
        len(successful)
        / measured_wall_time
        if measured_wall_time > 0
        else None
    )

    output_token_throughput = (
        total_output_tokens
        / measured_wall_time
        if measured_wall_time > 0
        else None
    )


    # =========================
    # Summary
    # =========================

    summary = {
        "experiment_id":
            experiment_id,

        "concurrency":
            args.concurrency,

        "requested_runs":
            measured_runs,

        "measured_waves":
            actual_measured_waves,

        "successful_runs":
            len(successful),

        "failed_runs":
            len(failed),

        "failure_rate": (
            len(failed)
            / measured_runs
            if measured_runs > 0
            else None
        ),

        "measured_wall_time_s":
            measured_wall_time,

        "ttft_s":
            summarize_metric(
                ttft_values
            ),

        "e2e_s":
            summarize_metric(
                e2e_values
            ),

        "tpot_s":
            summarize_metric(
                tpot_values
            ),

        "request_throughput_rps":
            request_throughput,

        "output_token_throughput_tps":
            output_token_throughput,

        "total_output_tokens":
            total_output_tokens,
    }


    # =========================
    # 保存 summary
    # =========================

    summary_path = (
        SUMMARY_DIR
        / f"{experiment_id}.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # =========================
    # 打印 summary
    # =========================

    print("\n=== Summary ===")

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\nRaw result: {raw_path}"
    )

    print(
        f"Summary:    {summary_path}"
    )

if __name__ == "__main__":
    main()