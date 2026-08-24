#!/usr/bin/env python3
"""
Generate the three report charts from results/crossover_summary.csv.

Chart 1 (chart1_tps_absolute.png): concurrency -> raw TPS, A/B/S. Shows
each condition's own absolute throughput and, critically, each
condition's own run-to-run spread (A's spread turned out to be the same
order of magnitude as B's -- see REPORT_PLAN.md Limitations #2). A only
has data at c=1/4/16, so its line is dashed (interpolated, not
measured) between those points.

Chart 2 (chart2_relative_to_A.png): concurrency -> % relative to A,
restricted to c=4/16 -- the only two concurrencies where A, B, and S
were ALL independently repeated three times, so this is the only
strictly "fair" cross-condition comparison. B and S are paired to A by
rerun tag (original/rerun1/rerun2), not by dividing every S value by a
single fixed A value (that was the bug behind the earlier, retracted
"c=4 ties" claim). Distance from 0 to B is the runtime tax; B to S is
SpecDec's own contribution; 0 to S is the net effect.

Chart 3 (chart3_speedup_pct.png): concurrency -> speedup %, two panels.
Left (S vs B) uses the full original concurrency sweep (c=1,2,4,8,16,32)
since B/S both ran it; c=8 only has a single measurement (no rerun) and
is marked accordingly. Right (S vs A) is restricted to c=4/16, the only
points where A has repeated measurements to pair against. Same-tag
points are connected with a line so the crossover reads as a
trajectory, not disconnected dots.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

IN = Path("results/crossover_summary.csv")
OUT_DIR = Path("results/charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_LABEL = {
    "A": "default_vllm_formal",
    "B": "matched_baseline_formal",
    "S": "specdec_k5_formal",
}
RERUN_TAGS = ["original", "rerun1", "rerun2"]
TAG_STYLE = {"original": ("o", 90), "rerun1": ("^", 90), "rerun2": ("s", 90)}
TAG_LINESTYLE = {"original": "-", "rerun1": "--", "rerun2": ":"}
COND_COLOR = {"A": "#444444", "B": "#1f77b4", "S": "#d62728"}
COND_LABEL = {"A": "A (default)", "B": "B (matched baseline)", "S": "S (SpecDec K=5)"}


def rerun_tag(condition, label):
    orig = ORIGINAL_LABEL[condition]
    if label == orig:
        return "original"
    if label == orig + "_rerun1":
        return "rerun1"
    if label == orig + "_rerun2":
        return "rerun2"
    return None


rows = []
with IN.open(newline="") as f:
    for r in csv.DictReader(f):
        tps = r.get("tps")
        if not tps:
            continue
        rows.append({
            "condition": r["condition"],
            "c": int(r["concurrency"]),
            "tps": float(tps),
            "tag": rerun_tag(r["condition"], r["label"]),
        })

by_key = {}  # (condition, c, tag) -> tps
for r in rows:
    by_key[(r["condition"], r["c"], r["tag"])] = r["tps"]


def pct_change(new, base):
    return 100.0 * (new / base - 1.0)


# ---------------------------------------------------------------
# Chart 1: concurrency -> raw TPS, A/B/S, showing each condition's own
# run-to-run spread
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

for cond in ["A", "B", "S"]:
    main = sorted(
        (r["c"], r["tps"]) for r in rows
        if r["condition"] == cond and r["tag"] == "original"
    )
    xs, ys = zip(*main)
    dense = len(xs) >= 5
    ax.plot(xs, ys, color=COND_COLOR[cond], linewidth=2,
             linestyle="-" if dense else "--",
             marker="o", markersize=6, label=COND_LABEL[cond])

for cond in ["A", "B", "S"]:
    reruns = [r for r in rows if r["condition"] == cond and r["tag"] in ("rerun1", "rerun2")]
    if reruns:
        xs = [r["c"] for r in reruns]
        ys = [r["tps"] for r in reruns]
        ax.scatter(xs, ys, facecolors="none", edgecolors=COND_COLOR[cond],
                   marker="o", s=40, linewidths=1.2, zorder=5)

ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.set_xticklabels([1, 2, 4, 8, 16, 32])
ax.set_xlabel("Concurrency (c)")
ax.set_ylabel("Throughput (tokens/sec)")
ax.set_title("Throughput vs. concurrency (A/B/S)", fontsize=13)
ax.text(0.5, 1.06,
        "dashed = A interpolated across an untested point (c=8); hollow = independent reruns",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#666666")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "chart1_tps_absolute.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Chart 2: concurrency -> % relative to A, restricted to c=4/16 only
# (the only points where A/B/S all have 3 independent reruns)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.5))

FAIR_CS = [4, 16]  # strict: A/B/S all repeated 3x here

for cond in ["B", "S"]:
    for tag in RERUN_TAGS:
        xs, ys = [], []
        for c in FAIR_CS:
            a = by_key.get(("A", c, tag))
            v = by_key.get((cond, c, tag))
            if a is not None and v is not None:
                xs.append(c)
                ys.append(pct_change(v, a))
        if xs:
            marker, size = TAG_STYLE[tag]
            ax.plot(xs, ys, color=COND_COLOR[cond], linestyle=TAG_LINESTYLE[tag],
                     linewidth=1.6, alpha=0.85, zorder=3)
            ax.scatter(xs, ys, color=COND_COLOR[cond], marker=marker, s=size, zorder=5)

ax.axhline(0, color=COND_COLOR["A"], linewidth=2)
ax.set_xscale("log", base=2)
ax.set_xticks(FAIR_CS)
ax.set_xticklabels(FAIR_CS)
ax.set_xlim(3, 20)
ax.set_xlabel("Concurrency (c)")
ax.set_ylabel("% relative to A (true default)")
ax.grid(True, alpha=0.3)

cond_handles = [Line2D([0], [0], color=COND_COLOR[c], marker="o", linestyle="-", label=COND_LABEL[c])
                for c in ["A", "B", "S"]]
tag_handles = [Line2D([0], [0], color="#666666", marker=TAG_STYLE[t][0], linestyle=TAG_LINESTYLE[t], label=t)
               for t in RERUN_TAGS]
ax.legend(handles=cond_handles + tag_handles, loc="lower left", fontsize=9)

fig.subplots_adjust(top=0.78)
fig.text(0.5, 0.97, "Runtime tax vs. SpecDec's own contribution (relative to A)",
        ha="center", fontsize=13)
fig.text(0.5, 0.905,
        "0 to B = runtime tax   |   B to S = SpecDec's own contribution   |   0 to S = net effect",
        ha="center", fontsize=8.5, color="#666666")
fig.text(0.5, 0.865,
        "restricted to c=4/16: the only points where A, B, S were each independently repeated 3x",
        ha="center", fontsize=8, color="#999999")
fig.savefig(OUT_DIR / "chart2_relative_to_A.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Chart 3: concurrency -> speedup %, two panels (S vs B, S vs A)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

# Panel 1: S vs B, full sweep. c=8 has no rerun -> marked distinctly.
ax = axes[0]
all_c = sorted(set(c for (cond, c, tag) in by_key if cond == "S"))
for tag in RERUN_TAGS:
    xs, ys = [], []
    for c in all_c:
        s = by_key.get(("S", c, tag))
        b = by_key.get(("B", c, tag))
        if s is not None and b is not None:
            xs.append(c)
            ys.append(pct_change(s, b))
    if xs:
        marker, size = TAG_STYLE[tag]
        ax.plot(xs, ys, linestyle=TAG_LINESTYLE[tag], color="#333333", linewidth=1.3, alpha=0.6, zorder=3)
        ax.scatter(xs, ys, marker=marker, s=size, color="#333333", label=tag, zorder=5)

# c=8: single measurement, no rerun -> distinct hollow marker + annotation
c8_s = by_key.get(("S", 8, "original"))
c8_b = by_key.get(("B", 8, "original"))
if c8_s is not None and c8_b is not None:
    y8 = pct_change(c8_s, c8_b)
    ax.scatter([8], [y8], marker="D", s=90, facecolors="none",
               edgecolors="#e07b00", linewidths=1.8, zorder=6)
    ax.annotate("c=8: single run,\nno rerun", xy=(8, y8), xytext=(8, y8 + 4),
                fontsize=7.5, color="#e07b00", ha="center")

ax.axhline(0, color="black", linewidth=1, alpha=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.set_xticklabels([1, 2, 4, 8, 16, 32])
ax.set_xlabel("Concurrency (c)")
ax.set_ylabel("Speedup (%)")
ax.set_title("S vs B (execution path held fixed)")
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 2: S vs A, restricted to c=4/16
ax = axes[1]
for tag in RERUN_TAGS:
    xs, ys = [], []
    for c in FAIR_CS:
        s = by_key.get(("S", c, tag))
        a = by_key.get(("A", c, tag))
        if s is not None and a is not None:
            xs.append(c)
            ys.append(pct_change(s, a))
    if xs:
        marker, size = TAG_STYLE[tag]
        ax.plot(xs, ys, linestyle=TAG_LINESTYLE[tag], color="#333333", linewidth=1.3, alpha=0.6, zorder=3)
        ax.scatter(xs, ys, marker=marker, s=size, color="#333333", label=tag, zorder=5)
ax.axhline(0, color="black", linewidth=1, alpha=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks(FAIR_CS)
ax.set_xticklabels(FAIR_CS)
ax.set_xlim(3, 20)
ax.set_xlabel("Concurrency (c)")
ax.set_title("S vs A (net effect vs. true default, c=4/16 only)")
ax.grid(True, alpha=0.3)
ax.legend()

fig.suptitle("Speculative decoding speedup: matched baseline vs. true default")
fig.tight_layout()
fig.savefig(OUT_DIR / "chart3_speedup_pct.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Wrote {OUT_DIR / 'chart1_tps_absolute.png'}")
print(f"Wrote {OUT_DIR / 'chart2_relative_to_A.png'}")
print(f"Wrote {OUT_DIR / 'chart3_speedup_pct.png'}")
