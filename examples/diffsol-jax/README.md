# diffsol-jax (prototype)

Use diffsol solves as JAX operations. Write the ODE equations as JAX functions. Use the solves
in `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd` and `jax.jacrev`.

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from diffsol_jax import Model

model = Model(
    rhs=lambda x, p, t: p[0] * x * (1.0 - x),   # one lane: x [nstates], p [nparams]
    init=lambda p, t: jnp.array([0.1]),
    nstates=1, nparams=1,
)
t = jnp.linspace(0.0, 5.0, 11)
ys = model.solve(jnp.array([1.0]), t)                     # [nt, nstates]
ys = jax.vmap(model.solve, (0, None))(jnp.ones((64, 1)), t)  # one solve with 64 lanes
g = jax.grad(lambda p: model.solve(p, t).sum())(jnp.array([1.0]))
```

## How it works

1. Python makes each diffsol callback (`rhs`, `rhs_jac`, `rhs_adjoint`, ...) from the user
   functions with JAX AD (`jvp` and `vjp`), and lowers it to StableHLO for all the lanes.
2. `libdiffsol_jax` (C++) compiles the StableHLO with the PJRT C API and puts the executables
   behind the function pointers of `diffsol_ode_new_callbacks_f64`.
3. `libdiffsol_jax` also gives the diffsol-c solves as XLA FFI custom calls:
   `diffsol_solve_dense`, `diffsol_solve_fwd_sens`, `diffsol_solve_vjp_fwd` and
   `diffsol_solve_vjp_bkwd`.

The callback inputs go to PJRT with no copy: PJRT gets views of the diffsol memory
(`PJRT_Client_CreateViewOfDeviceBuffer`). The XLA CPU client accepts a view only of 64-byte
aligned memory. An input that is not aligned is copied into a persistent aligned staging buffer
(no allocation after the first call). The output is copied into the diffsol output vector
with `memcpy` from the PJRT output buffer (CPU: host memory). `PJRT_Buffer_ToHostBuffer` is the
fallback; it costs about 9 µs more per call. One copy of the output, with no host allocation.
There is no allocation of data memory in a callback. The PJRT C API makes a small output
`PJRT_Buffer` and `PJRT_Event` objects in each call (and a view when the view cache misses);
these are the only allocations.

Measured cost (Apple M-series, CPU plugin): about 2.5 µs for each callback call, for a small
model (execute and wait about 2.0 µs, copy out 0.1 µs; with the view cache, almost no view is
made). Output aliasing with a donated view of `y` does not work with the CPU client.

Benchmark (`bench/bench.py`, Apple M-series, 16 GiB): rtol 1e-6, atol 1e-8, 41 output times.
Times in ms (median). The gradient is of `sum(ys[..., 0])`.

- native C: the same diffsol-c callback API with hand-written C callbacks
  (`bench/models_native.c`);
- diffsl: the LLVM 17 backend, one lane per solve (so B solves);
- diffrax 0.7.2: the same tolerances, `jax.vmap` for B > 1, gradient with its default
  `RecursiveCheckpointAdjoint`. For Robertson, diffrax has no BDF, so it uses `Kvaerno5`. For
  the other models, diffsol `tsit45` and diffrax `Tsit5` have the same tableau;
- oscillators: a chain of N nonlinear oscillators (2N states, 3 parameters), see the bench
  docstring. The gradients for N = 256, B = 1024 are skipped: the estimated VJP tape (5.4 GiB)
  is above `--max-tape-gib` (a run that kept several of these tapes crashed the machine).

The script checks all results against diffsol-jax: native C agrees to 1e-11 or better
(solutions and gradients), diffsl and diffrax to the solver tolerances.

| model (diffsol vs diffrax method) | B | diffsol-jax | native C | diffsl | diffrax | diffsol-jax grad | native C grad | diffsl grad | diffrax grad |
|---|---|---|---|---|---|---|---|---|---|
| Robertson (BDF vs Kvaerno5) | 1 | 1.38 | 0.0842 | 0.0837 | 4.04 | 6.49 | 0.346 | 0.346 | 22 |
| Robertson (BDF vs Kvaerno5) | 64 | 2.94 | 1.62 | 5.78 | 40.2 | 20.7 | 6.77 | 24.7 | 179 |
| Robertson (BDF vs Kvaerno5) | 1024 | 34 | 25.8 | 90.9 | 556 | 148 | 111 | 419 | 2.37e+03 |
| Lotka–Volterra (tsit45 vs Tsit5) | 1 | 4.24 | 0.0726 | 0.0617 | 0.0814 | 11.7 | 0.265 | 0.272 | 0.221 |
| Lotka–Volterra (tsit45 vs Tsit5) | 64 | 6.74 | 1.36 | 4.07 | 1.25 | 28.5 | 4.89 | 18.8 | 8.42 |
| Lotka–Volterra (tsit45 vs Tsit5) | 1024 | 49.8 | 21.8 | 66.2 | 24.1 | 166 | 82.5 | 306 | 138 |
| oscillators N=4 (tsit45 vs Tsit5) | 1 | 2.93 | 0.0702 | 0.105 | 0.071 | 9.24 | 0.252 | 0.366 | 0.194 |
| oscillators N=4 (tsit45 vs Tsit5) | 64 | 11.4 | 2.25 | 6.87 | 2.33 | 33.9 | 6.61 | 23.8 | 11.9 |
| oscillators N=4 (tsit45 vs Tsit5) | 1024 | 69.3 | 37.3 | 113 | 45.7 | 238 | 116 | 378 | 221 |
| oscillators N=32 (tsit45 vs Tsit5) | 1 | 2.69 | 0.128 | 0.452 | 0.166 | 9.71 | 0.472 | 1.55 | 0.66 |
| oscillators N=32 (tsit45 vs Tsit5) | 64 | 27.5 | 8.72 | 29.6 | 16.1 | 91 | 30.3 | 101 | 85.5 |
| oscillators N=32 (tsit45 vs Tsit5) | 1024 | 391 | 147 | 474 | 342 | 1.01e+03 | 517 | 1.84e+03 | 1.3e+03 |
| oscillators N=256 (tsit45 vs Tsit5) | 1 | 6.46 | 0.641 | 2.44 | 0.601 | 24.6 | 2.19 | 8.94 | 4.36 |
| oscillators N=256 (tsit45 vs Tsit5) | 64 | 266 | 55.8 | 179 | 126 | 554 | 193 | 613 | 518 |
| oscillators N=256 (tsit45 vs Tsit5) | 1024 | 2.27e+03 | 1.36e+03 | 2.73e+03 | 3.21e+03 | skipped | skipped | skipped | skipped |

Notes:
- Stiff (Robertson): diffsol-jax is 1.3 times slower than native C at B = 1024, and 14–16
  times faster than diffrax for B ≥ 64 (solve and gradient).
- Not stiff, small models (Lotka–Volterra, N = 4): diffrax is faster than diffsol-jax at all B
  (for example 0.08 against 4.1 ms for one Lotka–Volterra lane). An explicit method makes about
  6 callback calls per step, so the fixed cost of PJRT (about 2.5 µs per call) dominates, and
  diffrax compiles the whole loop with XLA.
- Not stiff, larger models (N = 32, 256): diffsol-jax and diffrax are similar; diffsol-jax is
  faster for gradients at B = 1024 (N = 32: about 1000 against 1300 ms) and for the largest
  solve (N = 256, B = 1024: about 2300 against 3400 ms).
- The large diffsol-jax runs vary between full benchmark runs (for example N = 256, B = 64:
  172 ms and 266 ms), because the other work in the same process (smaller models, diffsl,
  diffrax with JAX's own XLA thread pool) slows them. Run alone (`build/profile/jax_var.py`, 3
  processes, median of 20 solves), they are stable and faster: N = 256, B = 64: 111 ms (so
  faster than diffrax, about 120–135 ms); N = 32, B = 1024: 288 ms. The other columns vary much
  less. A more robust benchmark would run each method in its own process.
- The native C times for one lane include a fix in diffsol-c (`RawLayout::from_arrays` in
  `crates/diffsol-c/src/tensor.rs`): the callback path made and checked about 6 layouts per call,
  with small `memcpy` calls, which was 67 % of a Lotka–Volterra solve. Native C for one
  Lotka–Volterra lane went from 187 to 64 µs (diffsl: 54 µs), measured in C.
- diffsol-jax is about 1.5–2 times slower than native C for the large oscillator runs, where
  the per-call cost of PJRT is small (for N = 256, B = 1024: about 840 calls, 2367 against
  1343 ms). So the difference is in the callbacks themselves (the XLA code, and one copy of the
  output per call); this is not profiled yet.
- The diffsl vector form is slower than native C for N = 256 (2.56 against 0.71 ms at B = 1).

### Profile of the large-model gap

Oscillators, N = 256, B = 1024, tsit45 solve: diffsol-jax about 2.2–2.4 s, native C 1.34 s, about
930 `rhs` calls per solve. Tools: macOS `sample` on a loop of solves, and standalone timings
(scripts in `build/profile/`, not in git). The gap of about 1.05 s per solve:

| part | diffsol-jax | native C | difference |
|---|---|---|---|
| `rhs`: XLA kernel (the solve thread waits on `PJRT_Event_Await`) | about 730 ms | about 230 ms | +500 ms |
| `rhs`: execute, views, events | about 130 ms | – | +130 ms |
| `rhs`: copy of the output into `y` | about 80 ms | – | +80 ms |
| diffsol solver (same steps) | about 1400 ms | about 1110 ms | +290 ms |
| copy of the solution in the FFI handler | about 30 ms | – | +30 ms |

- The XLA kernel is 2.6–3 times slower than the C function (0.78–0.85 ms against 0.29 ms per
  call, measured alone). XLA:CPU makes 6 kernels, each with a full pass over a 2–4 MiB array: it
  does not fuse `pad` and `concatenate` with the other operations, and one kernel is only a copy
  to the `[B, 1, n]` output shape that `_lower.py` makes for `rhs`, `init` and `out`.
- Fix (measured alone): return `[B, n]` for these callbacks (the same bytes). This makes the
  kernel 0.59 ms instead of 0.78–0.85 ms, so about 200 ms less per solve.
- The diffsol solver itself is slower when the callbacks run in XLA. Cache pollution from the
  XLA temporaries explains about 70 ms (native C with 16 MiB of writes per call: 1110 to 1185
  ms). The rest (about 220 ms) is not explained. A possible cause, not tested: the solve thread
  blocks and wakes about 930 times per solve while XLA's worker threads run the kernel, and the
  scheduler can then run it on an efficiency core or at a lower clock.
  `--xla_cpu_multi_thread_eigen=false` has no effect.


The executables depend on the number of lanes `B`. `B` is known only after `jax.vmap`, so the
lowering rule of each solve compiles the callbacks for its `B`.

Reverse mode (`ad="adjoint"`, the default) uses the diffsol VJP. The forward solve keeps the
checkpoint in C++ and gives an int64 handle to the backward solve. The model keeps at most
`max_checkpoints` checkpoints. If the checkpoint of a backward solve was freed (for example in
`jax.grad` of a `lax.scan` with many solves), the backward solve runs the forward solve again
from `params` and `t_eval`. The gradient is the same; `max_checkpoints` controls only memory.
Forward mode (`ad="forward"`) uses diffsol forward sensitivities.

Limits of the prototype: CPU only, `f64` only, no `root`/`reset`, no derivative with respect to
`t_eval`, no `vmap` over `t_eval`.

## Build

1. The CPU PJRT plugin. jaxlib does not include one. Build it from openxla/xla:

   ```sh
   git clone --depth 1 https://github.com/openxla/xla.git && cd xla
   python3 configure.py --backend=CPU --host_compiler=clang
   bazelisk build -c opt //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
   ```

   `third_party/xla/pjrt/c/pjrt_c_api.h` comes from the same commit (see `third_party/README`).

2. diffsol-c: `cargo build -p diffsol-c --release` (from the repository root).

3. libdiffsol_jax (needs `jax` in the active Python, for the XLA FFI header):

   ```sh
   cmake -S examples/diffsol-jax -B examples/diffsol-jax/build -DCMAKE_BUILD_TYPE=Release
   cmake --build examples/diffsol-jax/build
   ```

4. Run the tests:

   ```sh
   export DIFFSOL_JAX_PJRT_PLUGIN=/path/to/xla/bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so
   python -m pytest examples/diffsol-jax/tests
   ```

5. Run the benchmark (needs `pip install diffrax`, and diffsol-c built with the LLVM diffsl
   backend: `LLVM_SYS_170_PREFIX=/opt/homebrew/opt/llvm@17 cargo build -p diffsol-c --release
   --features diffsl-llvm17`):

   ```sh
   clang -O3 -shared -fPIC -I target/release/include examples/diffsol-jax/bench/models_native.c \
     -o examples/diffsol-jax/build/libmodels_native.dylib
   python examples/diffsol-jax/bench/bench.py target/release/deps/libdiffsol_c.dylib \
     examples/diffsol-jax/build/libmodels_native.dylib [--models robertson,lotka_volterra,oscillators]
   ```

   Load `target/release/deps/libdiffsol_c.dylib`: it is the path that `libdiffsol_jax` links
   to, so only one copy of diffsol-c is in the process.

## GPU (not done)

The same C++ code can use the CUDA PJRT plugin (`jax-cuda12-pjrt`) when diffsol-c has a GPU
backend. The views then hold device pointers. The FFI handlers must sync with the XLA stream
(`ffi::PlatformStream<cudaStream_t>`). JAX and this library then have two GPU clients in one
process: set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
