#!/usr/bin/env python3
"""Export DSL model and projection matrices for lid-driven cavity flow at Re=400."""
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

def write_tns(path, triplets, rank=2):
    """Write sparse tensor in FROSTT format (1-based coordinates)."""
    with open(path, "w") as f:
        for t in triplets:
            coords = " ".join(str(c + 1) for c in t[:-1])  # 1-based
            val = t[-1]
            f.write(f"{coords} {val:.16e}\n")
    print(f"  Wrote {path} ({len(triplets)} nnz)")

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
Re = 400.0
nu = 1.0 / Re
nx = 16; ny = 16

mesh = UnitSquareMesh(nx, ny)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# Lid-driven cavity BC: top wall (y=1) has u=(1,0), all others no-slip
bcu_walls = DirichletBC(V, Constant((0.0, 0.0)), (1, 2, 3))  # bottom, right, left
bcu_lid   = DirichletBC(V, Constant((1.0, 0.0)), 4)           # top

# M with BC (boundary rows → identity, enforces du/dt=0 via mass matrix)
# A without BC (raw stiffness, interior coupling is correct)
M = assemble(inner(TrialFunction(V), TestFunction(V)) * dx, bcs=[bcu_walls, bcu_lid])
A = assemble(inner(grad(TrialFunction(V)), grad(TestFunction(V))) * dx)
B = assemble(div(TrialFunction(V)) * TestFunction(Q) * dx)

# BC node indices and values (applied explicitly after each step)
u_bc = Function(V, name="u_bc")
bcu_lid.apply(u_bc)
bcu_walls.apply(u_bc)
bc_mask = np.abs(u_bc.dat.data_ro.ravel()) > 0.5  # which DOFs have BC applied
bc_vals = u_bc.dat.data_ro.ravel()[bc_mask]        # values at those DOFs

M_triplets = petsc_triplets(M)
A_triplets = petsc_triplets(A)
B_triplets = petsc_triplets(B)

n_u = V.dim()
n_p = Q.dim()

# --- Initial condition (quiescent, with BC applied) ---
u_init_fun = Function(V, name="u_init")
u_init_fun.assign(0.0)
bcu_lid.apply(u_init_fun)
bcu_walls.apply(u_init_fun)
init_values = list(u_init_fun.dat.data_ro.ravel())

# --- Convection trilinear tensor C_ijk = ∫ (ψ_k·∇)ψ_j · ψ_i dx ---
print("Assembling convection tensor C_ijk...")
u_adv = Function(V, name="u_adv")
F_conv = inner(dot(u_adv, grad(TrialFunction(V))), TestFunction(V)) * dx

C_triplets = []
for k in range(n_u):
    if k % 200 == 0:
        print(f"  C tensor: DOF {k}/{n_u}")
    with u_adv.dat.vec as v:
        v.zeroEntries()
        v.setValue(k, 1.0)
    C_k = assemble(F_conv)
    for i, j, v in petsc_triplets(C_k):
        C_triplets.append((i, j, k, v))
print(f"  C_ijk: {len(C_triplets)} nonzeros")

# --- Write FROSTT tensor files ---
H_triplets = [(i, j, -nu * v) for i, j, v in A_triplets]
write_tns("mass.tns", M_triplets, rank=2)
write_tns("H.tns", H_triplets, rank=2)
write_tns("C.tns", C_triplets, rank=3)

# Write init as dense FROSTT (all entries, even zeros, for dense layout)
with open("init.tns", "w") as f:
    for i, val in enumerate(init_values):
        f.write(f"{i+1} {val:.16e}\n")
np.save("init.npy", np.array(init_values))
print(f"  Wrote init.tns, init.npy ({len(init_values)} entries)")

# The DSL is so simple now we embed it directly in the Rust code via format!().
# Save the parameters Rust needs.
params = {"nx": nx, "ny": ny, "nu": nu, "n_u": n_u, "n_p": n_p, "Re": Re, "problem": "lid-driven cavity"}
with open("meta.json", "w") as f:
    json.dump(params, f)
print("Wrote meta.json")
print("DSL will be embedded in main.rs via format!()")

# --- Lumped mass ---
M_mat = M.M.handle
m_lumped = np.zeros(n_u)
for i in range(n_u):
    _, vals = M_mat.getRow(i)
    m_lumped[i] = sum(abs(float(v)) for v in vals)

# --- Projection matrices ---
G_triplets, GT_triplets = [], []
for i, j, v in B_triplets:
    G_triplets.append((j, i, v / m_lumped[j]))
    GT_triplets.append((i, j, v / m_lumped[j]))

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
np.save("bc_mask.npy", bc_mask)
np.save("bc_vals.npy", bc_vals)
print(f"  Wrote bc_mask.npy ({bc_mask.sum()} BC nodes)")
