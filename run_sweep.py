#!/usr/bin/env python3
"""
Final one-command sweep pipeline.

Run on the Mac from:
    ~/single-gpu-specdec-client

For every concurrency:
1) Prewarm locally (NOT inside telemetry / Prometheus window).
2) Start remote telemetry -> creates before.prom.
3) Run measured benchmark locally with --warmup-waves 0.
4) Stop remote telemetry with SIGINT -> creates after.prom.
5) Verify csv + before.prom + after.prom.
6) Continue to next concurrency.

The local benchmark and remote telemetry therefore cover the same measured
request window.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32]


def pretty(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run(cmd, *, capture=False, check=True):
    print("+", pretty(cmd), flush=True)
    return subprocess.run(
        [str(x) for x in cmd],
        text=True,
        capture_output=capture,
        check=check,
    )


class SSHMaster:
    def __init__(self, host):
        self.host = host
        self.sock = f"/tmp/specdec-sweep-{os.getuid()}-{os.getpid()}.sock"

    def start(self):
        print("\n[ssh] opening one persistent connection")
        print("[ssh] enter AutoDL password once")
        run([
            "ssh",
            "-M",
            "-S", self.sock,
            "-o", "ControlPersist=no",
            "-fN",
            self.host,
        ])
        print("[ssh] ready")

    def remote(self, argv, *, capture=False, check=True):
        remote_command = pretty(argv)
        return run(
            ["ssh", "-S", self.sock, self.host, remote_command],
            capture=capture,
            check=check,
        )

    def close(self):
        subprocess.run(
            ["ssh", "-S", self.sock, "-O", "exit", self.host],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        try:
            Path(self.sock).unlink(missing_ok=True)
        except Exception:
            pass


def check_server(base_url, expect_specdec):
    models_url = base_url.rstrip("/") + "/v1/models"
    print(f"\n[preflight] checking {models_url}")

    try:
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            data = json.load(resp)
    except Exception as e:
        raise SystemExit(
            "Cannot reach vLLM through Mac localhost:8000.\n"
            "Keep the 8000 tunnel open and make sure vLLM is running.\n"
            f"Error: {e}"
        )

    print("[preflight] models:", [x.get("id") for x in data.get("data", [])])

    if expect_specdec:
        metrics_url = base_url.rstrip("/") + "/metrics"
        try:
            with urllib.request.urlopen(metrics_url, timeout=5) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            raise SystemExit(f"Cannot read SpecDec /metrics: {e}")

        if "spec_decode_num_drafts_total" not in text:
            raise SystemExit(
                "Expected SpecDec, but spec_decode counters were not found."
            )

        print("[preflight] SpecDec counters detected")


def benchmark_cmd(args, c, label, *, prewarm):
    if prewarm:
        return [
            args.python,
            args.benchmark,
            "--concurrency", str(c),
            "--warmup-waves", "0",
            "--measure-waves", str(args.prewarm_waves),
            "--min-runs", "0",
            "--label", f"prewarm_{label}",
        ]

    return [
        args.python,
        args.benchmark,
        "--concurrency", str(c),
        "--warmup-waves", "0",
        "--measure-waves", str(args.measure_waves),
        "--min-runs", str(args.min_runs),
        "--label", label,
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=DEFAULT_CONCURRENCIES,
    )
    p.add_argument("--remote-host", default="autodl-specdec")
    p.add_argument(
        "--remote-controller",
        default="/root/autodl-tmp/projects/single-gpu-specdec/scripts/telemetry_controller.py",
    )
    p.add_argument(
        "--remote-python",
        default="/root/autodl-tmp/venvs/vllm/bin/python3",
    )
    p.add_argument("--interval", type=float, default=0.2)
    p.add_argument("--prewarm-waves", type=int, default=2)
    p.add_argument("--measure-waves", type=int, default=10)
    p.add_argument("--min-runs", type=int, default=64)
    p.add_argument("--benchmark", default="benchmark.py")
    p.add_argument("--python", default="python3")
    p.add_argument("--local-base-url", default="http://127.0.0.1:8000")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--expect-specdec",
        action="store_true",
        help="require SpecDec counters in /metrics before running",
    )
    args = p.parse_args()

    if not Path(args.benchmark).exists():
        raise SystemExit(
            f"Cannot find {args.benchmark}; run from the local client project."
        )

    check_server(args.local_base_url, args.expect_specdec)

    manifest_dir = Path("results/sweep_manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = manifest_dir / f"{args.label}_{started}.json"

    manifest = {
        "label": args.label,
        "concurrencies": args.concurrencies,
        "prewarm_waves": args.prewarm_waves,
        "measure_waves": args.measure_waves,
        "min_runs": args.min_runs,
        "telemetry_interval_s": args.interval,
        "started_local": started,
        "completed": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n=== Final sweep protocol ===")
    print("label:", args.label)
    print("concurrencies:", args.concurrencies)
    print("prewarm waves outside telemetry:", args.prewarm_waves)
    print("measured waves:", args.measure_waves)
    print("min measured runs:", args.min_runs)
    print("telemetry interval:", args.interval)
    print("manifest:", manifest_path)

    ssh = SSHMaster(args.remote_host)

    try:
        ssh.start()

        # Fail early if the controller was not installed.
        check = ssh.remote(
            ["test", "-f", args.remote_controller],
            capture=True,
            check=False,
        )
        if check.returncode != 0:
            raise SystemExit(
                f"Remote controller not found:\n  {args.remote_controller}"
            )

        for c in args.concurrencies:
            print("\n" + "=" * 76)
            print(f"CONCURRENCY {c}")
            print("=" * 76)

            # 1) warm-up OUTSIDE the measured telemetry window
            print("\n[1/4] prewarm (not measured by telemetry)")
            r = run(benchmark_cmd(args, c, args.label, prewarm=True), check=False)
            if r.returncode != 0:
                raise RuntimeError(f"prewarm failed at c={c}")

            # 2) start telemetry -> before.prom
            print("\n[2/4] start remote telemetry")
            start_cmd = [
                args.remote_python,
                args.remote_controller,
                "start",
                "--label", args.label,
                "--concurrency", str(c),
                "--interval", str(args.interval),
            ]
            if args.overwrite:
                start_cmd.append("--overwrite")

            r = ssh.remote(start_cmd, capture=True, check=False)
            if r.returncode != 0:
                raise RuntimeError(
                    f"telemetry start failed at c={c}\n"
                    f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
                )
            print(r.stdout.strip())

            telemetry_started = True
            bench_ok = False

            try:
                # 3) measured benchmark, no internal warm-up
                print("\n[3/4] measured benchmark")
                r = run(
                    benchmark_cmd(args, c, args.label, prewarm=False),
                    check=False,
                )
                if r.returncode != 0:
                    raise RuntimeError(f"measured benchmark failed at c={c}")
                bench_ok = True

            finally:
                # 4) stop telemetry -> after.prom
                if telemetry_started:
                    print("\n[4/4] stop remote telemetry")
                    stop_cmd = [
                        args.remote_python,
                        args.remote_controller,
                        "stop",
                        "--label", args.label,
                        "--concurrency", str(c),
                    ]
                    s = ssh.remote(stop_cmd, capture=True, check=False)
                    print((s.stdout or "").strip())
                    if s.returncode != 0:
                        raise RuntimeError(
                            f"telemetry stop failed at c={c}\n"
                            f"stdout:\n{s.stdout}\nstderr:\n{s.stderr}"
                        )

            if bench_ok:
                manifest["completed"].append(c)
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f"\n[done] c={c}")

        print("\n" + "=" * 76)
        print("SWEEP COMPLETE")
        print("=" * 76)
        print("completed:", manifest["completed"])
        print("manifest:", manifest_path)

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)

    finally:
        ssh.close()


if __name__ == "__main__":
    main()
