# Events Reset BDF Order

This example compares BDF and ESDIRK34 on two decoupled exponential-decay modes:

```text
y1' = -y1
y2' = -1000 y2
```

The stiffness ratio is 1000.

At `t = 0, 0.1, ..., 10`, the example adds one unit to each state. Every jump
restarts BDF's multistep history at first order. ESDIRK34 is a one-step method,
so it resumes directly at its fixed order.

The example permits 202 recoverable nonlinear-solver failures so BDF can
complete all restarts instead of stopping at the default limit of 50.

Run the comparison with:

```sh
cargo run -p performance-solver-events --release
```

The program warms each solver once, then reports the median of 11 timings,
accepted steps, rejected steps, RHS evaluations, and maximum event-time error.
The exact reference multiplies each state by its exponential decay over an
interval and then adds the next unit dose.
