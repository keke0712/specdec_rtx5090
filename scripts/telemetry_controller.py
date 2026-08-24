#!/usr/bin/env python3
"""
Remote controller for scripts/telemetry.py.

Place this file on AutoDL at:
    /root/autodl-tmp/projects/single-gpu-specdec/scripts/telemetry_controller.py

It starts telemetry.py as a detached process, stores its PID, and stops it with
SIGINT so telemetry.py can write *_after.prom cleanly.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT / "run"
TELEMETRY_DIR = PROJECT / "results" / "telemetry"
LOG_DIR = PROJECT / "logs"


def safe_label(label: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    out = "".join(ch if ch in allowed else "_" for ch in label)
    if not out:
        raise ValueError("label is empty after sanitization")
    return out


def paths(label: str, concurrency: int):
    label = safe_label(label)
    stem = f"{label}_c{concurrency}"
    return {
        "stem": stem,
        "csv": TELEMETRY_DIR / f"{stem}.csv",
        "before": TELEMETRY_DIR / f"{stem}_before.prom",
        "after": TELEMETRY_DIR / f"{stem}_after.prom",
        "pid": RUNTIME_DIR / f"telemetry_{stem}.pid",
        "log": LOG_DIR / f"telemetry_{stem}.log",
    }


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_pid(pidfile: Path):
    try:
        return int(pidfile.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def cmd_start(args):
    p = paths(args.label, args.concurrency)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    old_pid = read_pid(p["pid"])
    if old_pid and pid_alive(old_pid):
        raise SystemExit(
            f"telemetry already running for {p['stem']} (pid={old_pid})"
        )

    if p["csv"].exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing telemetry: {p['csv']}\n"
            "use --overwrite only for an intentional rerun"
        )

    if args.overwrite:
        for key in ("csv", "before", "after", "pid", "log"):
            try:
                p[key].unlink()
            except FileNotFoundError:
                pass

    log_f = open(p["log"], "wb", buffering=0)

    cmd = [
        sys.executable,
        str(PROJECT / "scripts" / "telemetry.py"),
        "--out",
        str(p["csv"].relative_to(PROJECT)),
        "--interval",
        str(args.interval),
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT),
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log_f.close()

    p["pid"].write_text(str(proc.pid))

    deadline = time.time() + args.start_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = p["log"].read_text(errors="replace")[-4000:]
            except Exception:
                pass
            raise SystemExit(
                f"telemetry exited during startup (code={proc.returncode})\n{tail}"
            )

        if p["before"].exists() and p["csv"].exists():
            print(json.dumps({
                "status": "started",
                "pid": proc.pid,
                "stem": p["stem"],
                "csv": str(p["csv"]),
                "before": str(p["before"]),
                "log": str(p["log"]),
            }, ensure_ascii=False))
            return

        time.sleep(0.2)

    raise SystemExit(
        f"telemetry pid={proc.pid} started but startup files were not ready "
        f"within {args.start_timeout}s; inspect {p['log']}"
    )


def cmd_stop(args):
    p = paths(args.label, args.concurrency)
    pid = read_pid(p["pid"])

    if not pid:
        raise SystemExit(f"no pid file for {p['stem']}")

    if pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    deadline = time.time() + args.stop_timeout
    while time.time() < deadline:
        if p["after"].exists() and not pid_alive(pid):
            try:
                p["pid"].unlink()
            except FileNotFoundError:
                pass

            print(json.dumps({
                "status": "stopped",
                "pid": pid,
                "stem": p["stem"],
                "csv": str(p["csv"]),
                "before": str(p["before"]),
                "after": str(p["after"]),
            }, ensure_ascii=False))
            return
        time.sleep(0.2)

    # The key correctness condition is that after.prom exists.
    if p["after"].exists():
        try:
            p["pid"].unlink()
        except FileNotFoundError:
            pass
        print(json.dumps({
            "status": "stopped_after_written",
            "pid": pid,
            "stem": p["stem"],
            "after": str(p["after"]),
        }, ensure_ascii=False))
        return

    raise SystemExit(
        f"telemetry did not stop cleanly or write after.prom within "
        f"{args.stop_timeout}s; pid={pid}, log={p['log']}"
    )


def cmd_status(args):
    p = paths(args.label, args.concurrency)
    pid = read_pid(p["pid"])
    print(json.dumps({
        "stem": p["stem"],
        "pid": pid,
        "alive": bool(pid and pid_alive(pid)),
        "csv": p["csv"].exists(),
        "before": p["before"].exists(),
        "after": p["after"].exists(),
        "log": str(p["log"]),
    }, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--label", required=True)
        sp.add_argument("--concurrency", type=int, required=True)

    start = sub.add_parser("start")
    common(start)
    start.add_argument("--interval", type=float, default=0.2)
    start.add_argument("--overwrite", action="store_true")
    start.add_argument("--start-timeout", type=float, default=10.0)
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop")
    common(stop)
    stop.add_argument("--stop-timeout", type=float, default=20.0)
    stop.set_defaults(func=cmd_stop)

    status = sub.add_parser("status")
    common(status)
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
