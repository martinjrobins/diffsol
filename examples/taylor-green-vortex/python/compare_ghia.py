"""Compare against Ghia et al. (1982) benchmarks at Re=400."""
import json
import numpy as np
from firedrake import *

# Ghia et al. Table I: u-velocity along vertical centerline (x=0.5)
ghia_u = np.array([
    [1.0000,  1.00000],
    [0.9766,  0.75837],
    [0.9688,  0.68439],
    [0.9609,  0.61756],
    [0.9531,  0.55892],
    [0.8516,  0.29093],
    [0.7344,  0.16256],
    [0.6172,  0.02135],
    [0.5000, -0.11477],
    [0.4531, -0.17119],
    [0.2813, -0.32726],
    [0.1719, -0.24299],
    [0.1016, -0.14612],
    [0.0703, -0.10338],
    [0.0625, -0.09266],
    [0.0547, -0.08186],
    [0.0000,  0.00000],
])

# Ghia et al. Table II: v-velocity along horizontal centerline (y=0.5)
ghia_v = np.array([
    [1.00000,  0.00000],
    [0.9688,  -0.12146],
    [0.9609,  -0.15663],
    [0.9531,  -0.19254],
    [0.9453,  -0.22847],
    [0.9063,  -0.23827],
    [0.8594,  -0.44993],
    [0.8047,  -0.38598],
    [0.5000,   0.05186],
    [0.2344,   0.30174],
    [0.2266,   0.30203],
    [0.1563,   0.28124],
    [0.0938,   0.22965],
    [0.0781,   0.20920],
    [0.0703,   0.19713],
    [0.0625,   0.18360],
    [0.0000,   0.00000],
])

with open("meta.json") as f:
    meta = json.load(f)

Y = np.load("solution.npy")
n_u = meta["n_u"]
u_final = Y[:n_u, -1].reshape(-1, 2)

mesh = UnitSquareMesh(meta["nx"], meta["ny"])
V = VectorFunctionSpace(mesh, "CG", 2)
u_fun = Function(V)
u_fun.dat.data[:] = u_final

print("u-velocity along x=0.5 vertical centerline")
print(f"{'y':>8s}  {'num u':>10s}  {'Ghia u':>10s}  {'error':>10s}")
for y_ref, u_ref in ghia_u:
    pt = np.array([[0.5, y_ref]])
    u_num = np.array(u_fun.at(pt))[0, 0]
    err = u_num - u_ref
    print(f"{y_ref:8.4f}  {u_num:10.5f}  {u_ref:10.5f}  {err:10.5f}")

print()
print("v-velocity along y=0.5 horizontal centerline")
print(f"{'x':>8s}  {'num v':>10s}  {'Ghia v':>10s}  {'error':>10s}")
for x_ref, v_ref in ghia_v:
    pt = np.array([[x_ref, 0.5]])
    u_num = np.array(u_fun.at(pt))[0, 1]
    err = u_num - v_ref
    print(f"{x_ref:8.4f}  {u_num:10.5f}  {v_ref:10.5f}  {err:10.5f}")
