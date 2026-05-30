#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1/3: Generate DiffSL model from Firedrake ==="
python3 python/export_diffsl_from_firedrake.py

echo ""
echo "=== Step 2/3: Solve with diffsol ==="
cargo run --release

echo ""
echo "=== Step 3/3: Plot solution ==="
python3 python/plot_diffsol_solution.py

echo ""
echo "=== Done ==="
