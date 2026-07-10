#!/usr/bin/env python3
"""Compare lin_alg_ops benchmarks between two criterion baselines.

Usage:
    # On branch
    cargo bench --bench lin_alg_ops --features diffsl-llvm21 --features diffsl-cranelift \\
        -- --save-baseline branch

    # On main branch
    cargo bench --bench lin_alg_ops --features diffsl-llvm21 --features diffsl-cranelift \\
        -- --save-baseline main

    # Generate report
    python3 diffsol/benches/compare_ops.py [--criterion-dir target/criterion]

Outputs: performance_la_report.md
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_estimates(root, baseline_name):
    """Load criterion estimates for a given baseline."""
    estimates = {}
    for bench_dir in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(bench_dir):
            continue
        # Try the baseline subdirectory (from --save-baseline)
        id_dir = os.path.join(bench_dir, baseline_name, "estimates.json")
        if os.path.exists(id_dir):
            bench_name = os.path.basename(bench_dir)
            with open(id_dir) as f:
                data = json.load(f)
            estimates[bench_name] = {
                "mean": data["mean"]["point_estimate"],
                "std_dev": data["std_dev"]["point_estimate"],
            }
            continue
        # Fallback: try the "new" subdirectory and per-size subdirs
        id_dir = os.path.join(bench_dir, "new", "estimates.json")
        if os.path.exists(id_dir):
            bench_name = os.path.basename(bench_dir)
            with open(id_dir) as f:
                data = json.load(f)
            estimates[bench_name] = {
                "mean": data["mean"]["point_estimate"],
                "std_dev": data["std_dev"]["point_estimate"],
            }
        else:
            for size_dir in glob.glob(os.path.join(bench_dir, "*")):
                if not os.path.isdir(size_dir):
                    continue
                estimates_file = os.path.join(size_dir, baseline_name, "estimates.json")
                if not os.path.exists(estimates_file):
                    estimates_file = os.path.join(size_dir, "new", "estimates.json")
                if not os.path.exists(estimates_file):
                    continue
                bench_name = os.path.basename(bench_dir)
                size_str = os.path.basename(size_dir)
                full_name = f"{bench_name}/{size_str}"
                with open(estimates_file) as f:
                    data = json.load(f)
                estimates[full_name] = {
                    "mean": data["mean"]["point_estimate"],
                    "std_dev": data["std_dev"]["point_estimate"],
                }
    return estimates


def parse_name(name):
    if "/" not in name:
        return None, None, None
    op_backend, size_str = name.split("/", 1)
    try:
        ns = int(size_str.strip())
    except ValueError:
        return None, None, None

    known_backends = ["faer_sparse", "nalgebra", "faer", "cuda"]
    for be in known_backends:
        if op_backend.endswith("_" + be):
            op = op_backend[: -len(be) - 1]
            return op, be, ns
    parts = op_backend.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1], ns
    return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Compare lin_alg_ops benchmarks between two criterion baselines"
    )
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

    main_est = load_estimates(args.criterion_dir, args.base)
    branch_est = load_estimates(args.criterion_dir, args.branch)

    results = []
    for name in sorted(set(main_est) & set(branch_est)):
        op, backend, ns = parse_name(name)
        if op is None:
            continue
        m = main_est[name]
        b = branch_est[name]
        ratio = b["mean"] / m["mean"]
        results.append(
            {
                "op": op,
                "backend": backend,
                "ns": ns,
                "main_ns": m["mean"],
                "branch_ns": b["mean"],
                "ratio": ratio,
            }
        )

    ops = sorted(set(r["op"] for r in results))
    backends = sorted(set(r["backend"] for r in results))

    grouped = {}
    for r in results:
        key = f"{r['op']}/{r['backend']}"
        grouped.setdefault(key, []).append(r["ratio"])

    lines = [
        "# Linear Algebra Performance Report",
        "",
        f"**Branch:** `{args.branch}`  |  **Baseline:** `{args.base}`",
        "",
        "Each benchmark measures a single linear algebra operation at nbatch=1.",
        f"**{len(results)}** benchmark points across **{len(ops)}** operations, "
        f"**{len(backends)}** backends.",
        "",
    ]
    lines.append("## Summary by operation x backend")
    lines.append("")
    lines.append(
        "| Operation | Backend | GeoMean Ratio | Min | Max |"
    )
    lines.append("|---|---|---|---|---|")
    for key in sorted(grouped.keys()):
        rs = grouped[key]
        gm = np.exp(np.mean(np.log(rs)))
        flag = "🔴" if gm > 1.05 else ("🟢" if gm < 0.95 else "⚪")
        lines.append(
            f"| {key} | | {gm:.4f} {flag} | {min(rs):.4f} | {max(rs):.4f} |"
        )
    lines.append("")

    lines.append("## Top 15 Regressions")
    lines.append("")
    lines.append(
        "| Op/Backend | nstates | Base (ns) | Branch (ns) | Ratio |"
    )
    lines.append("|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x["ratio"], reverse=True)[:15]:
        lines.append(
            f"| {r['op']}/{r['backend']} | {r['ns']} "
            f"| {r['main_ns']:.1f} | {r['branch_ns']:.1f} "
            f"| {r['ratio']:.4f} 🔴 |"
        )
    lines.append("")

    lines.append("## Top 15 Improvements")
    lines.append("")
    lines.append(
        "| Op/Backend | nstates | Base (ns) | Branch (ns) | Ratio |"
    )
    lines.append("|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x["ratio"])[:15]:
        lines.append(
            f"| {r['op']}/{r['backend']} | {r['ns']} "
            f"| {r['main_ns']:.1f} | {r['branch_ns']:.1f} "
            f"| {r['ratio']:.4f} 🟢 |"
        )
    lines.append("")

    lines.append("## Per-operation geometric mean")
    lines.append("")
    cols = "| Operation | " + " | ".join(backends) + " |"
    lines.append(cols)
    lines.append("|" + "---|" * (len(backends) + 1))
    for op in ops:
        vals = "| " + op + " |"
        for be in backends:
            rs = [
                r["ratio"]
                for r in results
                if r["op"] == op and r["backend"] == be
            ]
            vals += f" {np.exp(np.mean(np.log(rs))):.4f} |" if rs else " - |"
        lines.append(vals)
    lines.append("")

    all_ratios = [r["ratio"] for r in results]
    gm_all = np.exp(np.mean(np.log(all_ratios)))
    num_reg = sum(1 for r in results if r["ratio"] > 1.05)
    num_imp = sum(1 for r in results if r["ratio"] < 0.95)
    lines.append(
        f"**Geometric mean ratio: {gm_all:.4f}** "
        f"({num_reg} regressions >5%, {num_imp} improvements >5% "
        f"out of {len(results)} points)"
    )
    lines.append("")

    lines.append("## Absolute speed comparison (ns)")
    lines.append("")

    priority_order = {
        "axpy": 1, "copy_from": 2, "sub_assign": 3, "add_assign": 4,
        "squared_norm": 5, "axpy_v": 6, "copy_from_view": 7,
        "add_ref_ref": 8, "sub_ref_ref": 9, "gemv": 10,
        "matrix_column": 11, "matrix_columns": 12,
        "fill": 13, "scalar_mul_assign": 14, "scalar_mul": 15,
        "scalar_div": 16, "component_mul_assign": 17, "component_div_assign": 18,
        "norm_l2": 19, "set_column": 20, "scale_add_and_assign": 21,
        "matrix_copy_from": 22, "add_column_to_vector": 23,
        "from_diagonal": 24, "from_element": 25, "from_vec": 26,
        "zeros": 27, "clone": 28, "as_view": 29, "as_view_mut": 30,
        "len": 31, "clone_as_vec": 32, "as_slice": 33, "as_mut_slice": 34,
        "set_index": 35, "get_index": 36, "root_finding": 37,
    }

    for op in sorted(ops, key=lambda o: priority_order.get(o, 99)):
        prio = priority_order.get(op, 99)
        if prio <= 12:
            p = "🔴"
        elif prio <= 23:
            p = "🟡"
        else:
            p = "🟢"
        op_sizes = sorted(set(r["ns"] for r in results if r["op"] == op))
        if not op_sizes:
            continue
        lines.append(f"### {p} {op}")
        lines.append("")
        header = (
            "| nstates | "
            + " | ".join(f"{b} (base)" for b in backends)
            + " | "
            + " | ".join(f"{b} (branch)" for b in backends)
            + " |"
        )
        lines.append(header)
        sep = "|" + "---|" * (1 + 2 * len(backends))
        lines.append(sep)
        for ns in op_sizes:
            row = f"| {ns} |"
            for be in backends:
                m = [
                    r
                    for r in results
                    if r["op"] == op and r["backend"] == be and r["ns"] == ns
                ]
                if m:
                    v = m[0]["main_ns"]
                    row += (
                        f" {v / 1e3:.1f}µs |" if v >= 1000 else f" {v:.1f} |"
                    )
                else:
                    row += " - |"
            for be in backends:
                m = [
                    r
                    for r in results
                    if r["op"] == op and r["backend"] == be and r["ns"] == ns
                ]
                if m:
                    v = m[0]["branch_ns"]
                    ratio = m[0]["ratio"]
                    s = f" {v / 1e3:.1f}µs" if v >= 1000 else f" {v:.1f}"
                    if ratio > 1.05:
                        s += "🔴"
                    elif ratio < 0.95:
                        s += "🟢"
                    row += f" {s} |"
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

    OUTPUT_MD = "performance_la_report.md"
    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(
        f"Report: {OUTPUT_MD}  Points: {len(results)}  "
        f"Geomean: {gm_all:.4f}  "
        f"Regressions: {num_reg}  Improvements: {num_imp}"
    )


if __name__ == "__main__":
    main()
