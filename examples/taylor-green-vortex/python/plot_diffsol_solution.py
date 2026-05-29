import json
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio

from firedrake import *
from firedrake.pyplot import tricontourf, tripcolor


def main():
    print("Loading meta.json...", flush=True)
    with open("meta.json", "r") as f:
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
    p_fun = Function(Q, name="pressure")
    speed_fun = Function(Q, name="speed")

    coords = mesh.coordinates.dat.data_ro

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    frames = []
    stride = max(1, nt // 200)

    print(f"Plotting {nt} time steps, stride {stride} -> {1 + nt // stride} frames", flush=True)

    for k in range(0, nt, stride):
        u_fun.dat.data[:] = Y[:n_u, k].reshape(-1, 2)
        pvals = Y[n_u:, k].copy()
        pvals -= pvals.mean()
        p_fun.dat.data[:] = pvals

        speed_fun.interpolate(sqrt(dot(u_fun, u_fun)))
        uv = np.array(u_fun.at(coords))

        for ax in axes:
            ax.clear()

        # Velocity: magnitude as background + arrow overlay
        tripcolor(speed_fun, axes=axes[0], shading="gouraud", cmap="inferno")
        axes[0].quiver(coords[:, 0], coords[:, 1],
                       uv[:, 0], uv[:, 1], color="white", scale=15, width=0.004)
        axes[0].set_title(f"|u| + arrows, t={ts[k]:.3f}")
        axes[0].set_aspect("equal")

        tricontourf(p_fun, axes=axes[1], levels=30)
        axes[1].set_title(f"p, t={ts[k]:.3f}")
        axes[1].set_aspect("equal")

        print(f"  frame {len(frames)+1}/{1+nt//stride}", flush=True)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(frame[:, :, :3].copy())

    imageio.mimsave("solution.gif", frames, duration=0.08)
    print("Wrote solution.gif")


if __name__ == "__main__":
    main()
