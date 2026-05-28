#!/usr/bin/env python3
"""Export heat-equation DSL and projection matrices for Taylor-Green vortex."""
import json
import numpy as np
from collections import defaultdict
from firedrake import *

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

def save_triplets_npy(prefix, triplets, nrows, ncols):
    i = np.array([t[0] for t in triplets], dtype=np.int64)
    j = np.array([t[1] for t in triplets], dtype=np.int64)
    v = np.array([t[2] for t in triplets], dtype=np.float64)
    np.save(f"{prefix}_i.npy", i)
    np.save(f"{prefix}_j.npy", j)
    np.save(f"{prefix}_v.npy", v)
    np.save(f"{prefix}_dims.npy", np.array([nrows, ncols], dtype=np.int64))
    print(f"  Wrote {prefix}_*.npy ({nrows}x{ncols}, {len(triplets)} nnz)")

# --- FEM setup ---
nx = 16; ny = 16; nu = 1.0e-2
mesh = UnitSquareMesh(nx, ny)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
bcu = DirichletBC(V, Constant((0.0, 0.0)), "on_boundary")

M = assemble(inner(TrialFunction(V), TestFunction(V)) * dx, bcs=bcu)
A = assemble(inner(grad(TrialFunction(V)), grad(TestFunction(V))) * dx, bcs=bcu)
B = assemble(div(TrialFunction(V)) * TestFunction(Q) * dx)

M_triplets = petsc_triplets(M)
A_triplets = petsc_triplets(A)
B_triplets = petsc_triplets(B)

n_u = V.dim()
n_p = Q.dim()

# --- Initial condition ---
x, y = SpatialCoordinate(mesh)
u_init_expr = as_vector([sin(2*pi*x)*cos(2*pi*y), -cos(2*pi*x)*sin(2*pi*y)])
u_init_fun = Function(V)
u_init_fun.interpolate(u_init_expr)
bcu.apply(u_init_fun)
init_values = list(u_init_fun.dat.data_ro.ravel())

# --- Heat-equation DiffSL model ---
H_triplets = [(i, j, -nu * v) for i, j, v in A_triplets]

with open("taylor_green.dsl", "w") as f:
    f.write(f"in_i {{ nu = {nu:.16e} }}\n\n")
    write_tensor(f, "Mass", M_triplets)
    write_tensor(f, "H", H_triplets)
    f.write("init_i {\n")
    for val in init_values:
        f.write(f"  {val:.16e},\n")
    f.write("}\n\n")
    f.write("u_i { y = init_i }\n\n")
    f.write(f"dudt_i {{ (0:{n_u}): dydt = 0 }}\n\n")
    f.write("M_i { Mass_ij * dydt_j }\n\n")
    f.write("F_i { H_ij * y_j }\n\n")
    f.write("out_i { u_i }\n")
print(f"Wrote taylor_green.dsl (heat eqn, n_u={n_u})")

# --- Lumped mass ---
M_mat = M.M.handle
m_lumped = np.zeros(n_u)
for i in range(n_u):
    _, vals = M_mat.getRow(i)
    m_lumped[i] = sum(abs(float(v)) for v in vals)

# --- Projection matrices ---
# G = M_lumped⁻¹ Bᵀ  (n_u × n_p),  GT = B M_lumped⁻¹  (n_p × n_u)
G_triplets, GT_triplets = [], []
for i, j, v in B_triplets:
    G_triplets.append((j, i, v / m_lumped[j]))
    GT_triplets.append((i, j, v / m_lumped[j]))

# L = B M_lumped⁻¹ Bᵀ  (n_p × n_p) — assembled via shared velocity DOFs
j_to_p = defaultdict(list)
for i, j, v in B_triplets:
    j_to_p[j].append((i, v))
L_rows = [defaultdict(float) for _ in range(n_p)]
for j, entries in j_to_p.items():
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

# --- Metadata ---
with open("meta.json", "w") as f:
    json.dump({"nx": nx, "ny": ny, "n_u": n_u, "n_p": n_p}, f)
print("Wrote meta.json")
