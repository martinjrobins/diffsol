# Switched stiff heat-equation solver comparison

This example compares BDF, TR-BDF2, and ESDIRK34 on two decoupled heat-equation
modes with decay rates `pi^2` and `(40 pi)^2`. The stiffness ratio is 1600.

The forcing starts at `1` and alternates sign every `0.05` time units through
`t = 5`. The state remains continuous, but its derivative jumps at every
switch. Each solver stops exactly at those switches. The example then updates
the stored derivative and marks the state modified: BDF consequently restarts
its multistep history at first order, while the one-step solvers can resume
without a history rebuild.

The example permits 200 recoverable nonlinear-solver failures so BDF can
complete all 100 restarts instead of stopping at the default limit of 50.

Run the comparison with:

```sh
cargo run -p performance-solver-events --release
```

The program reports elapsed time, accepted steps, rejected steps, RHS
evaluations, and the maximum error at a switch time. The reference is the
closed-form scalar recurrence for each mode on every constant-forcing segment.
