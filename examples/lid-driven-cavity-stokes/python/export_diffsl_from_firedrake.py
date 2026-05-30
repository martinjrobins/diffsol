#!/usr/bin/env python3
"""Export reduced Stokes operators for lid-driven cavity (Re=400, 32×32).

Splits velocity DOFs: u_f (free) and u_b (prescribed).
Eliminates u_b → reduced ODE on free DOFs with BC forcing.
All DiffSL tensors exported as FROSTT .tns files.
"""
import json, numpy as np
from collections import defaultdict
from firedrake import *

def petsc_triplets(A):
    mat = A.M.handle; rows, cols = mat.getSize(); triplets = []
    for i in range(rows):
        cols_i, vals_i = mat.getRow(i)
        for j, v in zip(cols_i, vals_i):
            if abs(v) > 1e-14: triplets.append((i, j, float(v)))
    return triplets

def write_tns(path, triplets, rank=2):
    with open(path, "w") as f:
        for t in triplets:
            coords = " ".join(str(t[c] + 1) for c in range(rank))
            f.write(f"{coords} {t[-1]:.16e}\n")
    print(f"  {path} ({len(triplets)} nnz)")

def write_vec_tns(path, vec):
    with open(path, "w") as f:
        for i, v in enumerate(vec): f.write(f"{i+1} {v:.16e}\n")

def save_npy(prefix, triplets, nrows, ncols):
    i = np.array([t[0] for t in triplets], dtype=np.int64)
    j = np.array([t[1] for t in triplets], dtype=np.int64)
    v = np.array([t[2] for t in triplets], dtype=np.float64)
    for name, arr in [("i", i), ("j", j), ("v", v)]:
        np.save(f"{prefix}_{name}.npy", arr)
    np.save(f"{prefix}_dims.npy", np.array([nrows, ncols], dtype=np.int64))
    print(f"  {prefix}_*.npy ({nrows}×{ncols}, {len(triplets)} nnz)")

# ---------------------------------------------------------------------------
Re = 400.0; nu = 1.0 / Re; nx = 32; ny = 32
mesh = UnitSquareMesh(nx, ny)
V = VectorFunctionSpace(mesh, "CG", 2); Q = FunctionSpace(mesh, "CG", 1)
bcu_walls = DirichletBC(V, Constant((0.0, 0.0)), (1, 2, 3))
bcu_lid   = DirichletBC(V, Constant((1.0, 0.0)), 4)

M = assemble(inner(TrialFunction(V), TestFunction(V)) * dx, bcs=[bcu_walls, bcu_lid])
A = assemble(inner(grad(TrialFunction(V)), grad(TestFunction(V))) * dx)
B = assemble(div(TrialFunction(V)) * TestFunction(Q) * dx)
n_u = V.dim(); n_p = Q.dim()

# Tag prescribed DOFs
u_tag = Function(V)
for bc in [DirichletBC(V, Constant((1.0,1.0)), i) for i in (1,2,3)] + \
          [DirichletBC(V, Constant((2.0,2.0)), 4)]:
    bc.apply(u_tag)
tag = u_tag.dat.data_ro.ravel()
bc_mask = np.abs(tag) > 0.5
free_mask = ~bc_mask
free_idx = {}
for i in range(n_u):
    if not bc_mask[i]: free_idx[i] = len(free_idx)
n_free = len(free_idx)
print(f"DOFs: {n_u} total, {n_free} free, {n_u-n_free} prescribed")

# BC values
u_bc = Function(V); bcu_lid.apply(u_bc)
for i in (1,2,3): DirichletBC(V, Constant((0.0,0.0)), i).apply(u_bc)
bc_vals = {i: u_bc.dat.data_ro.ravel()[i] for i in range(n_u) if bc_mask[i]}

# Reduced operators
M_trip = [(free_idx[i], free_idx[j], v)
          for i,j,v in petsc_triplets(M) if free_mask[i] and free_mask[j]]
A_trip = [(free_idx[i], free_idx[j], v)
          for i,j,v in petsc_triplets(A) if free_mask[i] and free_mask[j]]
B_f = [(i, free_idx[j], v) for i,j,v in petsc_triplets(B) if free_mask[j]]

# BC forcing: f_mom = -ν A_fb * u_bc
f_mom = np.zeros(n_free)
for i, j, v in petsc_triplets(A):
    if free_mask[i] and bc_mask[j]:
        f_mom[free_idx[i]] -= nu * v * bc_vals[j]

# f_div = B_b * u_bc
f_div = np.zeros(n_p)
for i, j, v in petsc_triplets(B):
    if bc_mask[j]: f_div[i] += v * bc_vals[j]

# Init
u_init = Function(V); u_init.assign(0.0)
bcu_lid.apply(u_init)
for i in (1,2,3): DirichletBC(V, Constant((0.0,0.0)), i).apply(u_init)
init_free = [u_init.dat.data_ro.ravel()[i] for i in range(n_u) if free_mask[i]]

# Lumped mass from M_ff
m_lumped = np.zeros(n_free)
for i, j, v in M_trip: m_lumped[i] += abs(v)
inv_mass = 1.0 / m_lumped

# Write DiffSL tensors (FROSTT)
H_trip = [(i, j, -nu*v) for i,j,v in A_trip]
write_tns("mass.tns", M_trip, rank=2)
write_tns("H.tns", H_trip, rank=2)
write_vec_tns("f_mom.tns", f_mom)
write_vec_tns("init.tns", init_free)
write_vec_tns("inv_mass.tns", inv_mass)

# Metadata
with open("meta.json", "w") as f:
    json.dump({"nx": nx, "ny": ny, "nu": nu, "n_u": n_u, "n_free": n_free,
               "n_p": n_p, "Re": Re, "problem": "lid-driven cavity"}, f)

# Lumped mass
np.save("m_lumped.npy", m_lumped)

# Projection matrices
G, GT = [], []
for i, j, v in B_f:
    G.append((j, i, v / m_lumped[j]))
    GT.append((i, j, v / m_lumped[j]))
j2p = defaultdict(list)
for i, j, v in B_f:
    j2p[j].append((i, v))
L_rows = [defaultdict(float) for _ in range(n_p)]
for j, entries in j2p.items():
    inv_m = 1.0 / m_lumped[j]
    for i1, v1 in entries:
        for i2, v2 in entries: L_rows[i1][i2] += v1 * v2 * inv_m
L_trip = [(i, k, val) for i in range(n_p)
          for k, val in sorted(L_rows[i].items()) if abs(val) > 1e-14]
save_npy("L", L_trip, n_p, n_p)
save_npy("G", G, n_free, n_p)
save_npy("GT", GT, n_p, n_free)

np.save("f_div.npy", f_div)
np.save("free_mask.npy", free_mask)
np.save("bc_vals_full.npy", np.array([bc_vals.get(i,0.0) for i in range(n_u)]))
print(f"  m_lumped, f_div, free_mask, bc_vals_full → .npy")
