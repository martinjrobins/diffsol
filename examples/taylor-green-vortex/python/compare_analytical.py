import json
import numpy as np
from firedrake import *

with open("meta.json") as f:
    meta = json.load(f)
nx = int(meta["nx"])
ny = int(meta["ny"])
n_u = int(meta["n_u"])
n_p = int(meta["n_p"])

Y = np.load("solution.npy")
ts = np.load("time.npy")
nt = Y.shape[1]

mesh = UnitSquareMesh(nx, ny)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
u_fun = Function(V, name="velocity")

# Evaluate at interior points (away from boundaries)
xi = np.linspace(0.05, 0.95, 40)
yi = np.linspace(0.05, 0.95, 40)
X, Ygrid = np.meshgrid(xi, yi)
pts = np.column_stack([X.ravel(), Ygrid.ravel()])

nu = 0.01
k2 = 2.0 * (2.0 * np.pi) ** 2

print(f"{'t':>8s}  {'L2(u_err)':>12s}  {'L∞(u_err)':>12s}  {'rel L2':>12s}  {'max|p|':>12s}")
print("-" * 70)

stride = max(1, nt // 40)

for k in range(0, nt, stride):
    t = ts[k]
    d = np.exp(-nu * k2 * t)

    u_fun.dat.data[:] = Y[:n_u, k].reshape(-1, 2)
    u_num = np.array(u_fun.at(pts))

    u_ana_x = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Ygrid) * d
    u_ana_y = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Ygrid) * d

    err = u_num - np.column_stack([u_ana_x.ravel(), u_ana_y.ravel()])
    l2 = np.sqrt(np.mean(err[:, 0]**2 + err[:, 1]**2))
    linfo = np.max(np.sqrt(err[:, 0]**2 + err[:, 1]**2))

    u_ana_norm = np.sqrt(np.mean(u_ana_x**2 + u_ana_y**2))
    rel_l2 = l2 / max(u_ana_norm, 1e-15)

    max_p = np.max(np.abs(Y[n_u:, k] - np.mean(Y[n_u:, k])))

    print(f"{t:8.4f}  {l2:12.6e}  {linfo:12.6e}  {rel_l2:12.6e}  {max_p:12.6e}")
