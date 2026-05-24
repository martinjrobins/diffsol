# Taylor–Green Vortex

Solves the 2D time-dependent Stokes equations on a unit square with a Taylor–Green vortex initial condition using a fractional-step projection method.

## Equations

The incompressible Stokes equations:

```
∂u/∂t = ν∇²u − ∇p          momentum
∇·u = 0                     continuity
```

with no-slip BC (u = 0 on ∂Ω) and initial condition:

```
u₀(x, y) = [sin(2πx)cos(2πy), −cos(2πx)sin(2πy)]
p₀(x, y) = 0
```

FEM discretization (Taylor-Hood P2/P1) yields the index-2 DAE:

```
M du/dt = −νA u + Bᵀ p
0       = B u
```

where M is the velocity mass matrix, A is the viscous stiffness, and B is the divergence operator.

## Fractional-step projection method

Each time step is split into three decoupled solves:

### Step 1 — Viscous advance (ODE)

Advance velocity ignoring pressure:

```
M (u* − uⁿ)/Δt = −νA u*
```

Solved via **BDF** (implicit multistep) using `diffsol`.

### Step 2 — Pressure Poisson (linear solve)

Enforce incompressibility by solving for the pressure correction:

```
L φ = (1/Δt) B u*        where   L = B M_lumped⁻¹ Bᵀ
```

Solved via **sparse LU** (faer).

### Step 3 — Projection

Remove divergent component from velocity:

```
uⁿ⁺¹ = u* − Δt M_lumped⁻¹ Bᵀ φ
```

## Implementation

```
python/export_diffsl_from_firedrake.py   — FEM assembly + export
src/main.rs                              — time loop + projection
python/plot_diffsol_solution.py          — visualize
python/compare_analytical.py             — error analysis
```

**Generated matrices:** M, A, B → exported as DiffSL tensors (`.dsl`), G and L as NPY triplets for the projection step.

## Run

```bash
./run.sh
```

Requires: Firedrake (Python), diffsol + LLVM/Cranelift (Rust).

## References

- Taylor & Green (1937), "Mechanism of the production of small eddies from large ones"
- Chorin (1968), "Numerical solution of the Navier-Stokes equations" — projection method
