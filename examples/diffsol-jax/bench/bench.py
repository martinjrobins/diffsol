"""Benchmark of diffsol-jax against native C callbacks, diffsl and diffrax.

Models (`--models`, comma separated):
  robertson       stiff, 3 states, 3 params:  diffsol BDF       vs diffrax Kvaerno5
  lotka_volterra  not stiff, 2 states, 4 params: diffsol tsit45 vs diffrax Tsit5 (same tableau)
  oscillators     not stiff, N coupled nonlinear oscillators (2N states, 3 params), N = 4, 32,
                  256: diffsol tsit45 vs diffrax Tsit5. A chain with fixed ends:
                  dq_i = v_i, dv_i = f(q_{i+1} - q_i) - f(q_i - q_{i-1}) - c v_i,
                  f(D) = k D + beta D^3, q_i(0) = exp(-(i - (N+1)/2)^2), v_i(0) = 0

Methods:
  diffsol-jax  JAX callbacks compiled with PJRT, the solve as an XLA FFI custom call;
  native       hand-written C callbacks (bench/models_native.c) through the same diffsol-c API;
  diffsl       the diffsl LLVM JIT through the diffsol-c C API (one lane per solve, so B
               solves). Build diffsol-c with `--features diffsl-llvm17`;
  diffrax      pure JAX; `jax.vmap` for many lanes; gradient with its default
               RecursiveCheckpointAdjoint.

The gradient is of `sum(ys[..., 0])` (the first state at all the output times and lanes).

usage: python bench.py <libdiffsol_c> <libmodels_native> [--models ...] [--lanes 1,64,1024]
                       [--max-tape-gib 2] [--out build/bench_results.jsonl]

The gradients are skipped when the estimated VJP tape is larger than --max-tape-gib (for
example oscillators with N = 256 and B = 1024): a full run on a 16 GiB machine crashed.
"""

import argparse
import ctypes as c
import json
import sys
import time
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from diffsol_jax import Model  # noqa: E402

RTOL, ATOL = 1e-6, 1e-8
SOLVER_IDS = {"bdf": 0, "esdirk34": 1, "tr_bdf2": 2, "tsit45": 3}

# ---- models -----------------------------------------------------------------------------------


def robertson_rhs(x, p, t):
    k1, k2, k3 = p
    return jnp.array(
        [
            -k1 * x[0] + k3 * x[1] * x[2],
            k1 * x[0] - k2 * x[1] ** 2 - k3 * x[1] * x[2],
            k2 * x[1] ** 2,
        ]
    )


def lv_rhs(x, p, t):
    a, b, c_, d = p
    return jnp.array([a * x[0] - b * x[0] * x[1], c_ * x[0] * x[1] - d * x[1]])


def osc_rhs(x, p, t):
    k, beta, damp = p
    n = x.shape[0] // 2
    q, v = x[:n], x[n:]
    qe = jnp.concatenate([jnp.zeros(1), q, jnp.zeros(1)])  # fixed ends
    d = qe[1:] - qe[:-1]  # d[i] = q_{i+1} - q_i
    f = k * d + beta * d**3
    return jnp.concatenate([v, f[1:] - f[:-1] - damp * v])


def osc_init(n):
    """A narrow bump in the middle: it excites the fast modes too, so the solve has several
    periods for any N (the lowest mode alone is very slow for large N)."""
    i = np.arange(1, n + 1)
    q0 = np.exp(-((i - (n + 1) / 2) ** 2))
    return np.concatenate([q0, np.zeros(n)])


def osc_diffsl(n):
    """diffsl code with N-vectors. Only the sizes (integer literals in the ranges) and the
    initial values depend on N.

    D (N+1 x N) gives the differences d_i = q_{i+1} - q_i with the fixed ends, and G (N x N+1)
    gives g_i = f_{i+1} - f_i. Notes on diffsl: `a..b` excludes `b`, and two `..` ranges make a
    diagonal (two `:` ranges make a dense block); an index that is not in the output is summed
    over the whole expression, so each matrix-vector product is in its own tensor (in
    `G_ij * f_j - damp * v_i`, the term `damp * v_i` would be summed N+1 times)."""
    q0 = ", ".join(repr(float(x)) for x in osc_init(n)[:n])
    return f"""
in_i {{ ks = 1.0, beta = 0.1, damp = 0.05 }}
D_ij {{ (0..{n}, 0..{n}): 1.0, (1..{n + 1}, 0..{n}): -1.0 }}
G_ij {{ (0..{n}, 1..{n + 1}): 1.0, (0..{n}, 0..{n}): -1.0 }}
q0_i {{ {q0} }}
u_i {{ (0:{n}): q = q0_i, ({n}:{2 * n}): v = 0.0 }}
d_i {{ D_ij * q_j }}
f_i {{ ks * d_i + beta * pow(d_i, 3) }}
g_i {{ G_ij * f_j }}
a_i {{ g_i - damp * v_i }}
F_i {{ v_i, a_i }}
out_i {{ q_i, v_i }}
""".encode()


def lane_scale(B):
    return 1.0 + 0.1 * np.arange(B) / max(B, 1)


def models(oscillator_sizes):
    out = {
        "robertson": dict(
            nstates=3,
            nparams=3,
            rhs=robertson_rhs,
            y0=np.array([1.0, 0.0, 0.0]),
            params=lambda B: np.stack(
                [0.04 * lane_scale(B), 3e7 * np.ones(B), 1e4 * lane_scale(B)], axis=-1
            ),
            t_eval=np.concatenate([[0.0], np.logspace(-4, 4, 40)]),
            solver="bdf",
            diffrax_solver=diffrax.Kvaerno5(),
            dt0=1e-6,
            diffsl=b"""
in_i { k1 = 0.04, k2 = 30000000.0, k3 = 10000.0 }
u_i { x = 1.0, y = 0.0, z = 0.0 }
F_i { -k1 * x + k3 * y * z, k1 * x - k2 * y * y - k3 * y * z, k2 * y * y }
out_i { x, y, z }
""",
            native=("robertson_desc", None),
        ),
        "lotka_volterra": dict(
            nstates=2,
            nparams=4,
            rhs=lv_rhs,
            y0=np.array([10.0, 5.0]),
            params=lambda B: np.stack(
                [1.5 * lane_scale(B), np.ones(B), 0.1 * np.ones(B), 3.0 * lane_scale(B)], axis=-1
            ),
            t_eval=np.linspace(0.0, 20.0, 41),
            solver="tsit45",
            diffrax_solver=diffrax.Tsit5(),
            dt0=None,
            diffsl=b"""
in_i { a = 1.5, b = 1.0, c = 0.1, d = 3.0 }
u_i { x = 10.0, y = 5.0 }
F_i { a * x - b * x * y, c * x * y - d * y }
out_i { x, y }
""",
            native=("lotka_volterra_desc", None),
        ),
    }
    for n in oscillator_sizes:
        out[f"oscillators_{n}"] = dict(
            nstates=2 * n,
            nparams=3,
            rhs=osc_rhs,
            y0=osc_init(n),
            params=lambda B: np.stack(
                [lane_scale(B), 0.1 * np.ones(B), 0.05 * np.ones(B)], axis=-1
            ),
            t_eval=np.linspace(0.0, 20.0, 41),
            solver="tsit45",
            diffrax_solver=diffrax.Tsit5(),
            dt0=None,
            diffsl=osc_diffsl(n),
            native=("oscillators_desc", n),
        )
    return out


# ---- timing -----------------------------------------------------------------------------------


def bench(fn, budget=1.0, min_runs=3, max_runs=20):
    """The median time of `fn`, after one warm-up run (which also compiles)."""
    fn()
    times = []
    start = time.perf_counter()
    while len(times) < max_runs and (
        len(times) < min_runs or time.perf_counter() - start < budget
    ):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---- diffsol-c through ctypes -----------------------------------------------------------------


class DiffsolC:
    def __init__(self, path, native_path):
        lib = self.lib = c.CDLL(path)
        self.native = c.CDLL(native_path)
        vp, sz, dp = c.c_void_p, c.c_size_t, c.POINTER(c.c_double)
        lib.diffsol_ode_new_jit.argtypes = [c.c_char_p] + [c.c_int32] * 4
        lib.diffsol_ode_new_jit.restype = vp
        lib.diffsol_jit_backend_type_count.restype = sz
        lib.diffsol_jit_backend_type_name.argtypes = [c.c_int32]
        lib.diffsol_jit_backend_type_name.restype = c.c_char_p
        lib.diffsol_ode_new_callbacks_f64.argtypes = [vp, c.c_int32, c.c_int32, c.c_int32, sz]
        lib.diffsol_ode_new_callbacks_f64.restype = vp
        lib.diffsol_ode_set_rtol.argtypes = lib.diffsol_ode_set_atol.argtypes = [vp, c.c_double]
        lib.diffsol_ode_solve_dense.argtypes = [vp, dp, sz, sz, dp, sz, c.POINTER(vp)]
        lib.diffsol_ode_solve_vjp_fwd.argtypes = [vp, dp, sz, sz, dp, sz] + [c.POINTER(vp)] * 2
        lib.diffsol_ode_solve_vjp_bkwd.argtypes = [vp, vp, dp, sz, sz, sz, c.POINTER(vp)]
        lib.diffsol_solution_wrapper_get_ys.argtypes = [vp, c.POINTER(vp)]
        lib.diffsol_tensor_ptr.argtypes = [vp]
        lib.diffsol_tensor_ptr.restype = c.POINTER(c.c_double)
        for f in (
            "diffsol_solution_wrapper_free",
            "diffsol_vjp_checkpoint_free",
            "diffsol_tensor_free",
            "diffsol_ode_free",
        ):
            getattr(lib, f).argtypes = [vp]
        lib.diffsol_error.restype = c.c_char_p
        self.native.desc_size.restype = sz
        self.native.robertson_desc.argtypes = [vp]
        self.native.lotka_volterra_desc.argtypes = [vp]
        self.native.oscillators_desc.argtypes = [vp, c.POINTER(sz)]
        self._keep = []

    def check(self, rc):
        if rc != 0:
            raise RuntimeError(self.lib.diffsol_error().decode())

    def _ode(self, ode):
        if not ode:
            raise RuntimeError(self.lib.diffsol_error().decode())
        self.lib.diffsol_ode_set_rtol(ode, RTOL)
        self.lib.diffsol_ode_set_atol(ode, ATOL)
        return ode

    def jit_backend(self, name):
        """The ids are fixed; a backend that is not in the build has no name (NULL)."""
        for i in range(self.lib.diffsol_jit_backend_type_count()):
            n = self.lib.diffsol_jit_backend_type_name(i)
            if n is not None and n.decode() == name:
                return i
        raise RuntimeError(f"diffsol-c has no '{name}' JIT backend")

    def diffsl_ode(self, code, solver):
        backend = self.jit_backend("llvm")
        return self._ode(self.lib.diffsol_ode_new_jit(code, backend, 0, 0, SOLVER_IDS[solver]))

    def native_ode(self, native, solver, B):
        fn, n = native
        desc = c.create_string_buffer(self.native.desc_size())
        if n is None:
            getattr(self.native, fn)(desc)
        else:
            n_ref = c.c_size_t(n)
            self._keep.append(n_ref)  # user_data: must outlive the ODE
            getattr(self.native, fn)(desc, c.byref(n_ref))
        return self._ode(self.lib.diffsol_ode_new_callbacks_f64(desc, 0, 0, SOLVER_IDS[solver], B))

    @staticmethod
    def ptr(a):
        return a.ctypes.data_as(c.POINTER(c.c_double))

    def solve(self, ode, p, t_eval, lanes, nout):
        """ys [lanes, nt, nout]."""
        lib, sol, ys = self.lib, c.c_void_p(), c.c_void_p()
        nt = len(t_eval)
        self.check(
            lib.diffsol_ode_solve_dense(
                ode, self.ptr(p), p.shape[-1], lanes, self.ptr(t_eval), nt, c.byref(sol)
            )
        )
        self.check(lib.diffsol_solution_wrapper_get_ys(sol, c.byref(ys)))
        out = np.ctypeslib.as_array(lib.diffsol_tensor_ptr(ys), (lanes * nt * nout,)).copy()
        lib.diffsol_tensor_free(ys)
        lib.diffsol_solution_wrapper_free(sol)
        return out.reshape(lanes, nt, nout)

    def grad(self, ode, p, t_eval, lanes, nout):
        """The gradient of sum(ys[..., 0]): [lanes, nparams]."""
        lib = self.lib
        sol, ckpt, grad = c.c_void_p(), c.c_void_p(), c.c_void_p()
        nt, npar = len(t_eval), p.shape[-1]
        self.check(
            lib.diffsol_ode_solve_vjp_fwd(
                ode, self.ptr(p), npar, lanes, self.ptr(t_eval), nt, c.byref(sol), c.byref(ckpt)
            )
        )
        dgdu = np.zeros((lanes, nt, nout))
        dgdu[..., 0] = 1.0
        self.check(
            lib.diffsol_ode_solve_vjp_bkwd(
                ode, ckpt, self.ptr(dgdu), nout, nt, lanes, c.byref(grad)
            )
        )
        out = np.ctypeslib.as_array(lib.diffsol_tensor_ptr(grad), (lanes * npar,)).copy()
        lib.diffsol_tensor_free(grad)
        lib.diffsol_vjp_checkpoint_free(ckpt)
        lib.diffsol_solution_wrapper_free(sol)
        return out.reshape(lanes, npar)


# ---- one model --------------------------------------------------------------------------------


# callback calls per step, to estimate the number of steps from the measured callback count
CALLS_PER_STEP = {"bdf": 1, "esdirk34": 3, "tr_bdf2": 2, "tsit45": 6}
# state vectors per step in a VJP tape (an over-estimate: the state, its derivative and the
# stages), for the memory estimate
VECTORS_PER_STEP = 10


def tape_gib(ncalls, solver, nstates, B):
    steps = ncalls / CALLS_PER_STEP[solver]
    return steps * VECTORS_PER_STEP * nstates * B * 8 / 2**30


def run_model(name, m, lanes, dc, max_tape_gib, out):
    nst, npar, t_eval = m["nstates"], m["nparams"], m["t_eval"]
    t_jax = jnp.asarray(t_eval)
    y0 = jnp.asarray(m["y0"])
    model = Model(
        rhs=m["rhs"],
        init=lambda p, t: y0,
        nstates=nst,
        nparams=npar,
        ode_solver=m["solver"],
        rtol=RTOL,
        atol=ATOL,
        # one checkpoint: the registry is cleared after each gradient timing anyway
        max_checkpoints=1,
    )
    jax_solve = jax.jit(model.solve)
    jax_grad = jax.jit(jax.grad(lambda p: model.solve(p, t_jax)[..., 0].sum()))

    def dfx_one(p):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: m["rhs"](y, args, t)),
            m["diffrax_solver"],
            t0=float(t_eval[0]),
            t1=float(t_eval[-1]),
            dt0=m["dt0"],
            y0=y0,
            args=p,
            saveat=diffrax.SaveAt(ts=t_jax),
            stepsize_controller=diffrax.PIDController(rtol=RTOL, atol=ATOL),
            max_steps=200_000,
        )
        return sol.ys, sol.stats["num_steps"]

    dfx_solve = jax.jit(lambda P: jax.vmap(dfx_one)(P)[0])
    dfx_steps = jax.jit(lambda P: jax.vmap(dfx_one)(P)[1])
    dfx_grad = jax.jit(jax.grad(lambda P: jax.vmap(dfx_one)(P)[0][..., 0].sum()))

    diffsl = dc.diffsl_ode(m["diffsl"], m["solver"])
    rows = []
    for B in lanes:
        p = np.ascontiguousarray(m["params"](B))  # [B, npar]
        pj = jnp.asarray(p)

        ys = np.asarray(jax_solve(pj, t_jax))  # [B, nt, nst]
        t_js = bench(lambda: jax_solve(pj, t_jax).block_until_ready())
        model.callback_stats(B, reset=True)
        jax_solve(pj, t_jax).block_until_ready()
        ncalls, _, _ = model.callback_stats(B, reset=True)
        # a VJP tape (and the diffrax checkpoints) can be several GiB for many lanes and states:
        # skip the gradients above the limit (a run with a 16 GiB machine crashed)
        est = tape_gib(ncalls, m["solver"], nst, B)
        do_grad = est <= max_tape_gib
        if not do_grad:
            print(
                f"{name} B={B}: gradients skipped, estimated tape {est:.1f} GiB > "
                f"--max-tape-gib {max_tape_gib}",
                flush=True,
            )
        nan = float("nan")
        g_jax, t_jg = None, nan
        if do_grad:
            g_jax = np.asarray(jax_grad(pj))
            t_jg = bench(lambda: jax_grad(pj).block_until_ready())
            model.clear_checkpoints()

        nat = dc.native_ode(m["native"], m["solver"], B)
        ys_nat = dc.solve(nat, p, t_eval, B, nst)
        t_ns = bench(lambda: dc.solve(nat, p, t_eval, B, nst))
        g_nat, t_ng = None, nan
        if do_grad:
            g_nat = dc.grad(nat, p, t_eval, B, nst)
            t_ng = bench(lambda: dc.grad(nat, p, t_eval, B, nst))
        dc.lib.diffsol_ode_free(nat)

        ys_dsl = np.stack([dc.solve(diffsl, p[b].copy(), t_eval, 1, nst)[0] for b in range(B)])
        t_ds = bench(lambda: [dc.solve(diffsl, p[b].copy(), t_eval, 1, nst) for b in range(B)])
        t_dg = nan
        if do_grad:
            t_dg = bench(
                lambda: [dc.grad(diffsl, p[b].copy(), t_eval, 1, nst) for b in range(B)]
            )

        ys_dfx = np.asarray(dfx_solve(pj))
        nsteps = int(np.max(dfx_steps(pj)))
        t_xs = bench(lambda: dfx_solve(pj).block_until_ready())
        t_xg = bench(lambda: dfx_grad(pj).block_until_ready()) if do_grad else nan

        scale = float(np.max(np.abs(ys)))
        gscale = float(np.max(np.abs(g_jax))) if do_grad else nan
        row = dict(
            model=name,
            B=B,
            solve=dict(jax=t_js, native=t_ns, diffsl=t_ds, diffrax=t_xs),
            grad=dict(jax=t_jg, native=t_ng, diffsl=t_dg, diffrax=t_xg),
            callbacks_per_solve=ncalls,
            diffrax_steps=nsteps,
            # agreement with diffsol-jax, relative to max |y| and max |grad|
            ys_diff=dict(
                native=float(np.max(np.abs(ys_nat - ys))) / scale,
                diffsl=float(np.max(np.abs(ys_dsl - ys))) / scale,
                diffrax=float(np.max(np.abs(ys_dfx - ys))) / scale,
            ),
            grad_diff_native=(
                float(np.max(np.abs(g_nat - g_jax.reshape(B, npar)))) / gscale if do_grad else nan
            ),
            tape_gib_estimate=est,
        )
        rows.append(row)
        # one line per row at once, so a crash does not lose the finished rows
        out.write(json.dumps(row) + "\n")
        out.flush()
        s, g, d = row["solve"], row["grad"], row["ys_diff"]
        print(
            f"{name:16s} B={B:5d} | solve ms: jax {s['jax']*1e3:8.2f} native "
            f"{s['native']*1e3:8.2f} diffsl {s['diffsl']*1e3:8.2f} diffrax {s['diffrax']*1e3:8.2f}"
            f" | grad ms: jax {g['jax']*1e3:8.2f} native {g['native']*1e3:8.2f} diffsl "
            f"{g['diffsl']*1e3:8.2f} diffrax {g['diffrax']*1e3:8.2f} | callbacks {ncalls}, "
            f"diffrax steps {nsteps} | ys diff: native {d['native']:.1e} diffsl "
            f"{d['diffsl']:.1e} diffrax {d['diffrax']:.1e}; grad diff native "
            f"{row['grad_diff_native']:.1e}",
            flush=True,
        )
    dc.lib.diffsol_ode_free(diffsl)
    return rows


def markdown(rows):
    out = [
        "| model | B | diffsol-jax | native C | diffsl | diffrax | diffsol-jax grad "
        "| native C grad | diffsl grad | diffrax grad |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def ms(x):
        return "skipped" if x != x else f"{x*1e3:.3g}"

    for r in rows:
        s, g = r["solve"], r["grad"]
        out.append(
            f"| {r['model']} | {r['B']} | {ms(s['jax'])} | {ms(s['native'])} "
            f"| {ms(s['diffsl'])} | {ms(s['diffrax'])} | {ms(g['jax'])} | {ms(g['native'])} "
            f"| {ms(g['diffsl'])} | {ms(g['diffrax'])} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("libdiffsol_c")
    ap.add_argument("libmodels_native")
    ap.add_argument("--models", default="robertson,lotka_volterra,oscillators")
    ap.add_argument("--lanes", default="1,64,1024")
    ap.add_argument("--oscillators", default="4,32,256", help="the values of N")
    ap.add_argument(
        "--max-tape-gib",
        type=float,
        default=2.0,
        help="skip the gradients if the estimated VJP tape is larger (GiB)",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "build" / "bench_results.jsonl"),
        help="append one JSON line per result row",
    )
    args = ap.parse_args()
    lanes = [int(x) for x in args.lanes.split(",")]
    sizes = [int(x) for x in args.oscillators.split(",")]
    all_models = models(sizes)
    names = []
    for name in args.models.split(","):
        names += [f"oscillators_{n}" for n in sizes] if name == "oscillators" else [name]
    dc = DiffsolC(args.libdiffsol_c, args.libmodels_native)
    rows = []
    with open(args.out, "a") as out:
        for name in names:
            rows += run_model(name, all_models[name], lanes, dc, args.max_tape_gib, out)
    print()
    print(markdown(rows))


if __name__ == "__main__":
    main()
