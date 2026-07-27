# Events Reset BDF Order

This example compares BDF and ESDIRK34 on a two-mode linear system:

```text
y1' = -y1
y2' = 999 y1 - 1000 y2
```

The stiffness ratio is 1000.

The problem starts at `(1, 1)`, the slow eigenvector. At `t = 0.05, 0.10,
..., 5`, the example adds `(1, 1)` to the state, so the solution remains on
that slow eigenspace and does not repeatedly excite a fast boundary layer.
Every jump restarts BDF's multistep history at first order. ESDIRK34 is a
one-step method, so it resumes directly at its fixed third order.

Run the comparison with:

```sh
cargo run -p performance-solver-events --release
```

The program warms each solver once, then reports the median of 11 timings,
accepted steps, rejected steps, RHS evaluations, BDF's mean maximum order per
interval, and maximum event-time error. The exact reference multiplies both
states by the slow exponential decay over an interval and then adds the next
unit dose.
