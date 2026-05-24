#!/usr/bin/env python3
"""Export DiffSL model + projection matrices for the Taylor-Green vortex.

Generates:
  - taylor_green.dsl: heat-equation-only ODE (M * du/dt = -nu * A * u)
  - L.npz: pressure Poisson matrix L = B * diag(M)^-1 * B^T
  - G.npz: gradient operator  G = diag(M)^-1 * B^T
  - meta.json: mesh metadata
"""
import json
import numpy as np
from collections import defaultdict
from firedrake import *
from petsc4py import PETSc

def petsc_triplets(A):
    mat = A.M.handle
    rows, cols = mat.getSize()
    triplets = []
    for i in range(rows):
        cols_i, vals_i = mat.getRow(i)
        for j, v in zip(cols_i, vals_i):
            if abs(v) > 1.0e-14:
                triplets.append((i, j, float(v)))
    return triplets

def write_tensor(f, name, triplets):
    f.write(f"{name}_ij {{\n")
    for i, j, v in triplets:
        f.write(f"  ({i}, {j}): {v:.16e},\n")
    f.write("}\n\n")

def save_triplets_npz(path, triplets, nrows, ncols):
    i = np.array([t[0] for t in triplets], dtype=np.int64)
    j = np.array([t[1] for t in triplets], dtype=np.int64)
    v = np.array([t[2] for t in triplets], dtype=np.float64)
    np.savez(path, i=i, j=j, v=v, nrows=nrows, ncols=ncols)
    nnz = len(triplets)
    print(f"  Wrote {path} ({nrows}x{ncols}, {nnz} nnz)")

def save_triplets_npy(prefix, triplets, nrows, ncols):
    i = np.array([t[0] for t in triplets], dtype=np.int64)
    j = np.array([t[1] for t in triplets], dtype=np.int64)
    v = np.array([t[2] for t in triplets], dtype=np.float64)
    np.save(f"{prefix}_i.npy", i)
    np.save(f"{prefix}_j.npy", j)
    np.save(f"{prefix}_v.npy", v)
    # Also save dimensions
    np.save(f"{prefix}_dims.npy", np.array([nrows, ncols], dtype=np.int64))
    nnz = len(triplets)
    print(f"  Wrote {prefix}_*.npy ({nrows}x{ncols}, {nnz} nnz)")

# ---------------------------------------------------------------------------
nx = 16
ny = 16
nu = 1.0e-2

mesh = UnitSquareMesh(nx, ny)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

p = TrialFunction(Q)
q = TestFunction(Q)

bcu = DirichletBC(V, Constant((0.0, 0.0)), "on_boundary")

M = assemble(inner(u, v) * dx, bcs=bcu)
A = assemble(inner(grad(u), grad(v)) * dx, bcs=bcu)
B = assemble(div(u) * q * dx)

M_triplets = petsc_triplets(M)
A_triplets = petsc_triplets(A)
B_triplets = petsc_triplets(B)

n_u = V.dim()
n_p = Q.dim()
n_total = n_u + n_p

# Initial condition
x, y = SpatialCoordinate(mesh)
u_init_expr = as_vector([sin(2 * pi * x) * cos(2 * pi * y),
                         -cos(2 * pi * x) * sin(2 * pi * y)])
u_init_fun = Function(V, name="u_init")
u_init_fun.interpolate(u_init_expr)
bcu.apply(u_init_fun)

init_values = list(u_init_fun.dat.data_ro.ravel())

# ---------------------------------------------------------------------------
# Heat-equation-only DSL (velocity only, no pressure)
# ---------------------------------------------------------------------------
# H = -nu * A  for velocity DOFs
H_triplets = [(i, j, -nu * v) for i, j, v in A_triplets]

with open("taylor_green.dsl", "w") as f:
    f.write("in_i {\n")
    f.write(f"  nu = {nu:.16e},\n")
    f.write("}\n\n")

    write_tensor(f, "Mass", M_triplets)
    write_tensor(f, "H", H_triplets)

    f.write(f"init_i {{\n")
    for val in init_values:
        f.write(f"  {val:.16e},\n")
    f.write("}\n\n")

    f.write("u_i {\n")
    f.write("  y = init_i,\n")
    f.write("}\n\n")

    f.write("dudt_i {\n")
    f.write(f"  (0:{n_u}): dydt = 0,\n")
    f.write("}\n\n")

    f.write("M_i {\n")
    f.write("  Mass_ij * dydt_j,\n")
    f.write("}\n\n")

    f.write("F_i {\n")
    f.write("  H_ij * y_j,\n")
    f.write("}\n\n")

    f.write("out_i {\n")
    f.write("  u_i,\n")
    f.write("}\n")

print(f"Wrote taylor_green.dsl (heat eqn, n_u={n_u})")

# ---------------------------------------------------------------------------
# Projection matrices  (lumped mass approximation)
# ---------------------------------------------------------------------------
# Lumped mass:  m_i = sum_j |M(i,j)|   (row-sum, always positive for CG FEM)
M_mat = M.M.handle
m_lumped = np.zeros(n_u)
for i in range(n_u):
    _, vals = M_mat.getRow(i)
    m_lumped[i] = sum(abs(float(v)) for v in vals)

# G = diag(m)^-1 * B^T   (size n_u x n_p)
# GT = B * diag(m)^-1   (size n_p x n_u)  = transpose of G
G_triplets = []
GT_triplets = []
for i, j, v in B_triplets:           # B has (i,j,v) where i in [0,n_p), j in [0,n_u)
    G_triplets.append((j, i, v / m_lumped[j]))
    GT_triplets.append((i, j, v / m_lumped[j]))

# L = B * diag(m)^-1 * B^T   (size n_p x n_p)
# Assemble via shared velocity DOFs: for each velocity DOF j, all pressure
# rows (i1,i2) that connect to j contribute to L(i1,i2)
j_to_p_rows = defaultdict(list)
for i, j, v in B_triplets:
    j_to_p_rows[j].append((i, v))

L_rows = [defaultdict(float) for _ in range(n_p)]
for j, entries in j_to_p_rows.items():
    inv_m = 1.0 / m_lumped[j]
    for i1, v1 in entries:
        for i2, v2 in entries:
            L_rows[i1][i2] += v1 * v2 * inv_m

L_triplets = []
for i in range(n_p):
    for k, val in sorted(L_rows[i].items()):
        if abs(val) > 1e-14:
            L_triplets.append((i, k, val))

save_triplets_npy("L", L_triplets, n_p, n_p)
save_triplets_npy("G", G_triplets, n_u, n_p)
save_triplets_npy("GT", GT_triplets, n_p, n_u)
np.save("m_lumped.npy", m_lumped)
print(f"  Wrote m_lumped.npy ({len(m_lumped)} entries)")

# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------
meta = {"nx": nx, "ny": ny, "n_u": n_u, "n_p": n_p}
with open("meta.json", "w") as f:
    json.dump(meta, f)
print("Wrote meta.json")
