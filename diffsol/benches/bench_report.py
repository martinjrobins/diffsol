#!/usr/bin/env python3
"""Compare ode_solvers benchmarks between two criterion baselines.

Usage (after running benchmarks on both branches):

    # On branch
    cargo bench --bench ode_solvers --features diffsl-llvm21 --features diffsl-cranelift \\
        -- --save-baseline branch 

    # On main branch
    cargo bench --bench ode_solvers --features diffsl-llvm21 --features diffsl-cranelift \\
        -- --save-baseline main

    # Generate report
    python3 scripts/bench_report.py [--criterion-dir target/criterion]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def load_estimates(root, baseline_name):
    """Load criterion estimates for a given baseline."""
    estimates = {}
    # Each benchmark is a subdirectory under root
    pattern = os.path.join(root, "*", baseline_name, "estimates.json")
    for path in glob.glob(pattern):
        bench_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as f:
            data = json.load(f)
        estimates[bench_name] = {
            "mean": data["mean"]["point_estimate"],
            "std_dev": data["std_dev"]["point_estimate"],
        }
    return estimates


def format_ns(ns):
    if ns < 1000:
        return f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns / 1000:.1f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def main():
    parser = argparse.ArgumentParser(description="Compare ODE solver benchmarks")
    parser.add_argument(
        "--criterion-dir",
        default="target/criterion",
        help="Path to criterion output directory",
    )
    parser.add_argument(
        "--branch",
        default="branch",
        help="Name of the feature branch baseline",
    )
    parser.add_argument(
        "--base",
        default="main",
        help="Name of the base branch baseline",
    )
    args = parser.parse_args()

    branch_est = load_estimates(args.criterion_dir, args.branch)
    base_est = load_estimates(args.criterion_dir, args.base)

    results = []
    for name in sorted(set(base_est) & set(branch_est)):
        m = base_est[name]
        b = branch_est[name]
        ratio = b["mean"] / m["mean"]
        results.append(
            {
                "name": name,
                "base_ns": m["mean"],
                "branch_ns": b["mean"],
                "ratio": ratio,
            }
        )

    all_ratios = [r["ratio"] for r in results]
    gm_all = np.exp(np.mean(np.log(all_ratios)))
    num_reg = sum(1 for r in results if r["ratio"] > 1.05)
    num_imp = sum(1 for r in results if r["ratio"] < 0.95)
    num_same = len(results) - num_reg - num_imp

    lines = []
    lines.append("# ODE Solver Performance Report")
    lines.append("")
    lines.append(f"**Branch:** `{args.branch}`  |  **Baseline:** `{args.base}`")
    lines.append("")
    lines.append(
        f"**Geometric mean ratio:** {gm_all:.4f}  "
        f"({num_reg} regressions >5%, {num_imp} improvements >5% "
        f"out of {len(results)} benchmarks)"
    )
    lines.append("")

    # Describe the convention
    lines.append("| Label | Meaning |")
    lines.append("|-------|---------|")
    lines.append(f"| 🔴 Regression >5% | `{args.branch}` is slower than `{args.base}` |")
    lines.append(f"| 🟢 Improvement >5% | `{args.branch}` is faster than `{args.base}` |")
    lines.append(f"| ⚪ Within 5% | No significant change |")
    lines.append("")

    # Regressions
    regs = sorted(
        [r for r in results if r["ratio"] > 1.05],
        key=lambda r: r["ratio"],
        reverse=True,
    )
    if regs:
        lines.append("## Regressions (branch slower)")
        lines.append("")
        lines.append("| Benchmark | Base (ns) | Branch (ns) | Ratio |")
        lines.append("|-----------|-----------|-------------|-------|")
        for r in regs:
            lines.append(
                f"| {r['name']} | {format_ns(r['base_ns'])} "
                f"| {format_ns(r['branch_ns'])} | {r['ratio']:.4f} 🔴 |"
            )
        lines.append("")

    # Improvements
    imps = sorted(
        [r for r in results if r["ratio"] < 0.95],
        key=lambda r: r["ratio"],
    )
    if imps:
        lines.append("## Improvements (branch faster)")
        lines.append("")
        lines.append("| Benchmark | Base (ns) | Branch (ns) | Ratio |")
        lines.append("|-----------|-----------|-------------|-------|")
        for r in imps:
            lines.append(
                f"| {r['name']} | {format_ns(r['base_ns'])} "
                f"| {format_ns(r['branch_ns'])} | {r['ratio']:.4f} 🟢 |"
            )
        lines.append("")

    # Unchanged
    same = sorted(
        [r for r in results if 0.95 <= r["ratio"] <= 1.05],
        key=lambda r: abs(r["ratio"] - 1.0),
        reverse=True,
    )
    if same:
        lines.append("## Unchanged (within 5%)")
        lines.append("")
        lines.append("| Benchmark | Base (ns) | Branch (ns) | Ratio |")
        lines.append("|-----------|-----------|-------------|-------|")
        for r in same:
            lines.append(
                f"| {r['name']} | {format_ns(r['base_ns'])} "
                f"| {format_ns(r['branch_ns'])} | {r['ratio']:.4f} ⚪ |"
            )
        lines.append("")

    report = "\n".join(lines)
    print(report)

    # Also write to file
    out_path = "ode_solver_performance_report.md"
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(
        f"\nReport written to {out_path}  "
        f"Benchmarks: {len(results)}  "
        f"Geomean: {gm_all:.4f}  "
        f"Regressions: {num_reg}  "
        f"Improvements: {num_imp}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
