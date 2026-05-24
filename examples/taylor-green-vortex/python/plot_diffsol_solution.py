import json
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio

from firedrake import *
from firedrake.pyplot import tricontourf


def main():
    with open("meta.json", "r") as f:
        meta = json.load(f)

    nx = int(meta["nx"])
    ny = int(meta["ny"])
    n_u = int(meta["n_u"])
    n_p = int(meta["n_p"])

    Y = np.load("solution.npy")  # shape: (n_u + n_p, nt)
    ts = np.load("time.npy")     # shape: (nt,)
    nt = Y.shape[1]

    mesh = UnitSquareMesh(nx, ny)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)

    u_fun = Function(V, name="velocity")
    p_fun = Function(Q, name="pressure")

    # Extract mesh vertex coordinates for quiver
    vx = mesh.coordinates.dat.data_ro[:, 0]
    vy = mesh.coordinates.dat.data_ro[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    frames = []
    stride = max(1, nt // 200)

    for k in range(0, nt, stride):
        u_fun.dat.data[:] = Y[:n_u, k].reshape(-1, 2)
        pvals = Y[n_u:, k].copy()
        pvals -= pvals.mean()
        p_fun.dat.data[:] = pvals

        # Evaluate velocity at mesh vertices
        u_at_verts = np.array(u_fun.at(mesh.coordinates.dat.data_ro))
        speed = np.sqrt(u_at_verts[:, 0]**2 + u_at_verts[:, 1]**2)

        for ax in axes:
            ax.clear()

        # Quiver colored by speed magnitude
        q = axes[0].quiver(
            vx, vy,
            u_at_verts[:, 0], u_at_verts[:, 1],
            speed, cmap="viridis", scale=8, width=0.005,
        )
        axes[0].set_title(f"u, t={ts[k]:.3f}")
        axes[0].set_aspect("equal")

        tricontourf(p_fun, axes=axes[1], levels=30)
        axes[1].set_title(f"p, t={ts[k]:.3f}")
        axes[1].set_aspect("equal")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(frame[:, :, :3].copy())

    imageio.mimsave("solution.gif", frames, duration=0.08)
    print("Wrote solution.gif")


if __name__ == "__main__":
    main()
