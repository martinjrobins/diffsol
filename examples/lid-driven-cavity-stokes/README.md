# Lid-Driven Cavity (Reduced Stokes)

Solves the 2D time-dependent Stokes equations for a lid-driven cavity
using a fractional-step projection method with **Dirichlet BC elimination**.

## Equations

The incompressible Stokes equations with FEM (Taylor-Hood P2/P1):

```
M du/dt = −νA u + Bᵀ p
0       = B u
```

Splitting velocity DOFs into free (u_f) and prescribed (u_b = g) yields the
**reduced index-1 DAE** on free DOFs only:

```
M_ff du_f/dt = −ν A_ff u_f − ν A_fb g + B_fᵀ p
0            = B_f u_f + B_b g
```

The BC forcing term `f_mom = −ν A_fb g` is constant (exported as FROSTT).
The divergence correction `f_div = B_b g` enters the projection step.

## Fractional-step projection method

### Step 1 — Viscous advance (ODE)

```
M_ff du_f/dt = −ν A_ff u_f + f_mom
```

Solved via TR-BDF2 (diffsol).

### Step 2 — Pressure Poisson

```
L φ = (1/Δt) (B_f u_f* + f_div)     where L = B_f M_lumped⁻¹ B_fᵀ
```

Solved via sparse LU (faer).

### Step 3 — Projection

```
u_fⁿ⁺¹ = u_f* − Δt M_lumped⁻¹ B_fᵀ φ
```

## Implementation

```
python/export_diffsl_from_firedrake.py  — FEM assembly, DOF splitting, export
src/main.rs                             — time loop + projection
python/plot_diffsol_solution.py         — visualization
python/compare_ghia.py                  — Ghia et al. (1982) benchmark
```

All DiffSL tensors are loaded from FROSTT `.tns` files via `read()`.
The DSL is embedded in Rust using `format!()`.

## Run

```bash
export LLVM_DIR=/usr/lib/llvm-21
export LLVM_SYS_211_PREFIX=/usr/lib/llvm-21
python3 python/export_diffsl_from_firedrake.py
cargo run --release --features diffsol-llvm21
python3 python/plot_diffsol_solution.py
```

## Solution (16×16, Re=400)

Stokes limit — no convection. At t=5 flow is approaching steady state:
max|u| = 1.0 (lid), min|u| ≈ −0.65 (recirculation).

## References

- Ghia, Ghia & Shin (1982) "High-Re solutions for incompressible flow..."
- Chorin (1968) "Numerical solution of the Navier-Stokes equations"
- Perot (1993) "An analysis of the fractional step method"
