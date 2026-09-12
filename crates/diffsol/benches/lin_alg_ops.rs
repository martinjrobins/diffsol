use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use diffsol::{
    Context, DenseMatrix, FaerLU, FaerMat, FaerSparseMat, FaerVec, IndexType, Matrix, MatrixCommon,
    NalgebraLU, NalgebraMat, NalgebraVec, Scale, Vector,
};
#[cfg(feature = "cuda")]
use diffsol::{CudaLU, CudaMat, CudaVec};
#[cfg(feature = "cuda-oxide")]
use diffsol::{OxideLU, OxideMat, OxideVec};
use diffsol_la::{LinearOp as LaLinearOp, LinearSolver as LaLinearSolver};
use diffsol_nl::{
    Convergence, NewtonNonlinearSolver, NoLineSearch, NonLinearOp, NonLinearOpJacobian,
    NonLinearSolver,
};
use std::hint::black_box;

const VSIZES: &[usize] = &[2, 10, 100, 500];
const MSIZES: &[usize] = &[10, 100, 500];
const ONE_SIZE: &[usize] = &[50];

// Note: Every `b.iter` body ends with `ctx.synchronize()` so device backends do the work.
// This means the device backends have a launch-plus-sync timing floor (~8.5us on an A40).

/// Sizes the `for_each_*` benches run: a one-thread-per-lane launch only shows its cost at a
/// lane long enough to be worth splitting.
const LANE_SIZES: &[usize] = &[50, 10_000];

// ─────────────────────────────────────────────────────────
// Helper: binary mutating op on two owned vectors
// ─────────────────────────────────────────────────────────
fn bench_vector_op<V>(c: &mut Criterion, label: &str, sizes: &[usize], op: fn(&mut V, &V))
where
    V: Vector<T = f64> + 'static,
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let mut y = V::from_element(ns, 1.0, ctx.clone());
            let x = V::from_element(ns, 2.0, ctx.clone());
            b.iter(|| {
                op(&mut y, &x);
                black_box(&y);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

fn bench_vector_ro<V, R>(c: &mut Criterion, label: &str, sizes: &[usize], op: fn(&V, &V, &V) -> R)
where
    V: Vector<T = f64> + 'static,
    V::C: Default + Clone,
    R: 'static,
{
    let mut group = c.benchmark_group(label);
    for &ns in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let v = V::from_element(ns, 1.0, ctx.clone());
            let y = V::from_element(ns, 1.0, ctx.clone());
            let a = V::from_element(ns, 0.1, ctx.clone());
            b.iter(|| op(&v, &y, &a));
        });
    }
    group.finish();
}

fn bench_vector_unary<V>(c: &mut Criterion, label: &str, sizes: &[usize], op: fn(&mut V))
where
    V: Vector<T = f64> + 'static,
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let mut v = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                op(&mut v);
                black_box(&v);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

fn bench_vector_prop<V>(c: &mut Criterion, label: &str, sizes: &[usize], op: fn(&V) -> f64)
where
    V: Vector<T = f64> + 'static,
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let v = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| op(&v));
        });
    }
    group.finish();
}

fn bench_vector_construct<V>(
    c: &mut Criterion,
    label: &str,
    sizes: &[usize],
    op: fn(usize, V::C) -> V,
) where
    V: Vector<T = f64> + 'static,
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            b.iter(|| {
                let v = op(ns, ctx.clone());
                ctx.synchronize();
                v
            });
        });
    }
    group.finish();
}

fn bench_vector_rb<V>(
    c: &mut Criterion,
    label: &str,
    sizes: &[usize],
    op: fn(&V, &V) -> (bool, f64, i32),
) where
    V: Vector<T = f64> + 'static,
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let v = V::from_element(ns, 1.0, ctx.clone());
            let g = V::from_element(ns, -1.0, ctx.clone());
            b.iter(|| op(&v, &g));
        });
    }
    group.finish();
}

fn bench_matrix_op<M>(
    c: &mut Criterion,
    label: &str,
    sizes: &[usize],
    setup_mat: fn(&mut M, usize),
    op: fn(&M, &M::V, &mut M::V),
) where
    M: Matrix<T = f64> + 'static,
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = M::C::default();
            let mut mat = M::zeros(ns, ns, ctx.clone());
            setup_mat(&mut mat, ns);
            let x = M::V::zeros(ns, ctx.clone());
            let mut y = M::V::zeros(ns, ctx.clone());
            b.iter(|| {
                op(&mat, &x, &mut y);
                black_box(&y);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

fn fill_dense<M: Matrix<T = f64>>(mat: &mut M, _ns: usize)
where
    M::V: Vector<T = f64, C = M::C>,
    M::C: Clone,
{
    let ctx = mat.context().clone();
    let v = M::V::from_element(mat.nrows(), 2.0, ctx);
    mat.set_column(0, &v);
}

// ═════════════════════════════════════════════════════════
// 🔴 HIGH — inner solver loop, every Newton iteration / stage
// ═════════════════════════════════════════════════════════

/// 🔴 axpy — Predictor correction, Newton updates, RK stage accumulation
fn bench_axpy<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_op::<V>(c, label, VSIZES, |y, x| y.axpy(1.0, x, 0.5));
}

/// 🔴 copy_from — State copies before every Newton iteration
fn bench_copy_from<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_op::<V>(c, label, VSIZES, |y, x| y.copy_from(x));
}

/// 🔴 sub_assign — Newton step x -= delta
fn bench_sub_assign<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_op::<V>(c, label, VSIZES, |y, x| *y -= x);
}

/// 🔴 add_assign — RHS assembly tmp += psi
fn bench_add_assign<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_op::<V>(c, label, VSIZES, |y, x| *y += x);
}

/// 🔴 squared_norm — Newton convergence check, error estimation
fn bench_squared_norm<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_ro::<V, f64>(c, label, VSIZES, |v, y, a| v.squared_norm(y, a, 0.1));
}

/// 🔴 axpy_v — Nordsieck psi setup, SDIRK stage prediction
fn bench_axpy_v<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in VSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let mut y = V::from_element(ns, 1.0, ctx.clone());
            let x = V::from_element(ns, 2.0, ctx.clone());
            let x_view = x.as_view();
            b.iter(|| {
                y.axpy_v(1.0, &x_view, 0.5);
                black_box(&y);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🔴 copy_from_view — SDIRK stage prediction, BDF diff column copy
fn bench_copy_from_view<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in VSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let mut y = V::from_element(ns, 1.0, ctx.clone());
            let x = V::from_element(ns, 2.0, ctx.clone());
            let x_view = x.as_view();
            b.iter(|| {
                y.copy_from_view(&x_view);
                black_box(&y);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🔴 add (ref+ref) — `&V + &V`, used to build temporary vectors from references
fn bench_add_ref_ref<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
    for<'b> &'b V: std::ops::Add<&'b V, Output = V>,
{
    let mut group = c.benchmark_group(label);
    for &ns in VSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let y = V::from_element(ns, 1.0, ctx.clone());
            let x = V::from_element(ns, 2.0, ctx.clone());
            b.iter(|| {
                black_box(&y + &x);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🔴 sub (ref+ref) — `&V - &V`, used to build temporary vectors from references
fn bench_sub_ref_ref<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
    for<'b> &'b V: std::ops::Sub<&'b V, Output = V>,
{
    let mut group = c.benchmark_group(label);
    for &ns in VSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let y = V::from_element(ns, 1.0, ctx.clone());
            let x = V::from_element(ns, 2.0, ctx.clone());
            b.iter(|| {
                black_box(&y - &x);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🔴 add (owned lhs) — `V + &V`, which writes into the left operand's allocation.  The
/// left operand is rebuilt outside the timed section, so only the addition is measured.
fn bench_add_owned_lhs<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in VSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let y = V::from_element(ns, 1.0, ctx.clone());
            let x = V::from_element(ns, 2.0, ctx.clone());
            b.iter_batched(
                || {
                    let y = y.clone();
                    // the clone enqueues too, so let it land before the timed body starts
                    ctx.synchronize();
                    y
                },
                |y| {
                    black_box(y + &x);
                    ctx.synchronize();
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// 🔴 sub (owned lhs) — `V - &V`, the Newton-residual shape: the left operand is consumed
/// and reused for the result.
fn bench_sub_owned_lhs<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in VSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let y = V::from_element(ns, 1.0, ctx.clone());
            let x = V::from_element(ns, 2.0, ctx.clone());
            b.iter_batched(
                || {
                    let y = y.clone();
                    // the clone enqueues too, so let it land before the timed body starts
                    ctx.synchronize();
                    y
                },
                |y| {
                    black_box(y - &x);
                    ctx.synchronize();
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// 🔴 gemv — Matrix-vector multiply in RHS, error estimation, interpolation
fn bench_gemv<M: Matrix<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    bench_matrix_op::<M>(c, label, MSIZES, fill_dense, |mat, x, y| {
        mat.gemv(1.0, x, 1.0, y)
    });
}

/// `(states, lanes)` cells the wide-sensitivity gemv bench sweeps.
const GEMV_WIDE_CELLS: &[(usize, usize)] = &[(3, 10), (32, 10), (100, 10), (256, 10)];

/// 🔴 gemv (batched) — the same multiply over many lanes. `nparams > 1` is the grouped broadcast
/// the sensitivity equations run: `nbatch * nparams` right-hand-side lanes over `nbatch` matrices.
/// `grouped_x` flips which operand is wide -- `nbatch * nparams` matrix lanes against `nbatch`
/// lanes of `x`, the shape a user's `jac_mul_inplace` makes when it multiplies a matrix it built
/// at the augmented width by a state-width vector.
fn bench_gemv_batched<M: Matrix<T = f64> + 'static>(
    c: &mut Criterion,
    label: &str,
    cells: &[(usize, usize)],
    nparams: usize,
    grouped_x: bool,
) where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    group.sample_size(20);
    for &(ns, nb) in cells {
        let id = if nparams == 1 {
            format!("n{ns}_nbatch{nb}")
        } else {
            format!("n{ns}_nbatch{nb}x{nparams}")
        };
        // the setup is inside the closure so that a filtered-out cell costs nothing
        group.bench_function(id, |b| {
            let ctx = M::C::default()
                .clone_with_nbatch(nb)
                .expect("backend declined nbatch");
            let wide = ctx
                .clone_with_nbatch(nb * nparams)
                .expect("backend declined nbatch");
            // `y` always carries every lane; the other wide operand is the matrix or `x`
            let (mat_ctx, x_ctx) = if grouped_x {
                (wide.clone(), ctx.clone())
            } else {
                (ctx.clone(), wide.clone())
            };
            let mut mat = M::zeros(ns, ns, mat_ctx);
            fill_dense(&mut mat, ns);
            let x = M::V::from_element(ns, 1.0, x_ctx);
            let mut y = M::V::zeros(ns, wide);
            b.iter(|| {
                // beta = 0, so repeated iterations neither drift nor overflow
                mat.gemv(1.0, &x, 0.0, &mut y);
                black_box(&y);
                ctx.synchronize();
            })
        });
    }
    group.finish();
}

/// 🔴 matrix_column — Extract a vector view of one matrix column per batch.
/// Called every RK stage (diff.column(i)) and every BDF Nordsieck/diff update.
fn bench_matrix_column<M: Matrix<T = f64> + DenseMatrix + 'static>(c: &mut Criterion, label: &str)
where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in MSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = M::C::default();
            let mat = M::zeros(ns, ns + 1, ctx.clone());
            b.iter(|| {
                black_box(mat.column(0));
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🔴 gemv_cols — y = alpha * diff[:, start..end] * x + beta * y, a BDF order + 1 column sum
/// and every RK stage combination. Called twice every BDF step (predictor and psi), once per
/// RK stage, and once per interpolation.
fn bench_gemv_cols<M: Matrix<T = f64> + DenseMatrix + 'static>(c: &mut Criterion, label: &str)
where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    // a fifth-order BDF difference table: sum columns 0..=5
    let weights = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];
    for &ns in MSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = M::C::default();
            let mut mat = M::zeros(ns, weights.len() + 3, ctx.clone());
            fill_dense(&mut mat, ns);
            let mut v = M::V::zeros(ns, ctx.clone());
            b.iter(|| {
                mat.gemv_cols(0, weights.len(), 1.0, &weights, 0.0, &mut v);
                black_box(&v);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// Sizes seen by the RK stage accumulation: the hot benchmark problems are tiny
/// (`exponential_decay` is 2 states, `robertson` 3), which `MSIZES` does not cover.
const RK_SIZES: &[usize] = &[2, 3, 10, 100, 500];

/// 🔴 stage_accumulate — `y = y0 + sum_{j<k} w_j * diff[:, j]`, the RK stage combination
/// (`Rk::do_stage`, `SdirkCallable::set_phi`) and the BDF predictor/psi update, on the
/// tall-thin `ns x k` shape the solvers actually use.
///
/// `gemv_cols` is the whole operation; `full_gemv` multiplies the whole matrix instead, as a
/// reference point for how much of the cost is the multiply itself rather than the column
/// range. Keep an eye on the small `ns` rows: the ODE benchmarks are 2-3 states, and that is
/// where backend dispatch overhead dominates the arithmetic.
fn bench_stage_accumulate<M: Matrix<T = f64> + DenseMatrix + 'static>(
    c: &mut Criterion,
    label: &str,
) where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    // a mid-tableau RK stage: 6 preceding stages to combine
    let weights = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];
    let k = weights.len();
    for &ns in RK_SIZES {
        let ctx = M::C::default();
        let mut mat = M::zeros(ns, k + 1, ctx.clone());
        fill_dense(&mut mat, ns);
        let y0 = M::V::from_element(ns, 1.0, ctx.clone());
        let mut y = M::V::zeros(ns, ctx.clone());

        group.bench_with_input(BenchmarkId::new("full_gemv", ns), &ns, |b, _| {
            let w_full = M::V::from_vec(vec![1.0; k + 1], ctx.clone());
            b.iter(|| {
                y.copy_from(&y0);
                mat.gemv(1.0, &w_full, 1.0, &mut y);
                black_box(&y);
                ctx.synchronize();
            });
        });
        group.bench_with_input(BenchmarkId::new("gemv_cols", ns), &ns, |b, _| {
            b.iter(|| {
                y.copy_from(&y0);
                mat.gemv_cols(0, k, 1.0, &weights, 1.0, &mut y);
                black_box(&y);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// Row counts `mul_cols_by` sees: the ODE benchmark problems are 2-3 states, the sparse ones
/// run into the thousands, and the staging buffer is a fixed size at both ends.
const MUL_COLS_SIZES: &[usize] = &[2, 3, 10, 100, 1000, 10000];

/// 🟡 mul_cols_by — `D[:, 0..k] = D[:, 0..k] * (R * U)`, the BDF difference-table rescale on a
/// step-size change (`Bdf::_update_diff_for_step_size`). `k` is `order + 1`, so 1..=6.
///
/// The serial backends stage each row tile in a fixed `MUL_COLS_TILE` buffer sized so the tile
/// stays in L1; the buffer is the same size whatever `ns` is, so watch both the 2-3 row rows
/// (where the staging setup is the whole cost) and the tall ones (where the tiling pays off).
fn bench_mul_cols_by<M: Matrix<T = f64> + DenseMatrix + 'static>(c: &mut Criterion, label: &str)
where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    for &k in &[2usize, 6] {
        // the identity leaves the values unchanged across iterations -- same arithmetic, but
        // nothing drifts or denormalises over a long measurement
        let mut rhs = vec![0.0; k * k];
        for j in 0..k {
            rhs[j * k + j] = 1.0;
        }
        for &ns in MUL_COLS_SIZES {
            group.bench_with_input(BenchmarkId::new(format!("k{k}"), ns), &ns, |b, &ns| {
                let ctx = M::C::default();
                let mut mat = M::zeros(ns, k + 3, ctx.clone());
                fill_dense(&mut mat, ns);
                b.iter(|| {
                    mat.mul_cols_by(k, &rhs);
                    black_box(&mat);
                    ctx.synchronize();
                });
            });
        }
    }
    group.finish();
}

// ═════════════════════════════════════════════════════════
// 🟡 MEDIUM — once or few times per step
// ═════════════════════════════════════════════════════════

/// 🟡 fill — Predictor zeroing, pre-accumulation
fn bench_fill<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_unary::<V>(c, label, VSIZES, |y| y.fill(7.0));
}

/// 🟡 scalar_mul_assign — Step-size rescaling dy *= 1/h
fn bench_scalar_mul_assign<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_unary::<V>(c, label, VSIZES, |y| *y *= Scale(2.0));
}

/// 🟡 scalar_mul — Scaled vector copies (ref * scale)
fn bench_scalar_mul<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
    for<'b> &'b V: std::ops::Mul<Scale<f64>, Output = V>,
{
    let mut group = c.benchmark_group(label);
    for &ns in VSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let y = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                black_box(&y * Scale(2.0));
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🟡 scalar_div — Inverse scaling (owned only; &V / Scale not implemented)
fn bench_scalar_div<V: Vector<T = f64> + Clone + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in VSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let y = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                black_box(y.clone() / Scale(2.0));
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🟡 component_mul_assign — Jacobian estimation squared
fn bench_component_mul_assign<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_op::<V>(c, label, VSIZES, |y, x| y.component_mul_assign(x));
}

/// 🟡 component_div_assign — Error norm denominator
fn bench_component_div_assign<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_op::<V>(c, label, VSIZES, |y, x| y.component_div_assign(x));
}

/// 🟡 norm — Jacobian estimation, convergence checks
fn bench_norm<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_prop::<V>(c, label, VSIZES, |v| v.norm(2));
}

/// 🟡 set_column — Matrix column population
fn bench_set_column<M: Matrix<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in MSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = M::C::default();
            let mut mat = M::zeros(ns, ns, ctx.clone());
            let v = M::V::from_element(ns, 2.0, ctx.clone());
            b.iter(|| {
                mat.set_column(0, &v);
                black_box(&mat);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🟡 scale_add_and_assign — Matrix blending
fn bench_scale_add_and_assign<M: Matrix<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in MSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = M::C::default();
            let mut mat = M::zeros(ns, ns, ctx.clone());
            let x = M::zeros(ns, ns, ctx.clone());
            let y = M::zeros(ns, ns, ctx.clone());
            b.iter(|| {
                mat.scale_add_and_assign(&x, 2.0, &y);
                black_box(&mat);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🟡 copy_from (matrix) — Matrix duplication
fn bench_matrix_copy_from<M: Matrix<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in MSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = M::C::default();
            let mut mat = M::zeros(ns, ns, ctx.clone());
            let other = M::zeros(ns, ns, ctx.clone());
            b.iter(|| {
                mat.copy_from(&other);
                black_box(&mat);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

// ═════════════════════════════════════════════════════════
// 🟢 LOW — rare: construction, indexing, batch ops
// ═════════════════════════════════════════════════════════

/// 🟢 set_index — Algebraic constraint application
fn bench_set_index<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_unary::<V>(c, label, ONE_SIZE, |y| y.set_index(0, 5.0));
}

/// 🟢 get_index — Finite difference Jacobian
fn bench_get_index<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in ONE_SIZE {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let v = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                black_box(v.get_index(0));
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🟢 root_finding — Event detection between steps
fn bench_root_finding<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_rb::<V>(c, label, VSIZES, |v, g| v.root_finding(g));
}

/// 🟢 from_element — Vector construction
fn bench_from_element<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_construct::<V>(c, label, ONE_SIZE, |ns, ctx| {
        V::from_element(ns, 1.0, ctx.clone())
    });
}

/// 🟢 from_vec — Vector construction from Vec
fn bench_from_vec<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in ONE_SIZE {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let vec = vec![1.0_f64; ns];
            b.iter(|| V::from_vec(vec.clone(), ctx.clone()));
        });
    }
    group.finish();
}

/// 🟢 zeros — Vector allocation
fn bench_zeros<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_construct::<V>(c, label, ONE_SIZE, |ns, ctx| V::zeros(ns, ctx.clone()));
}

/// 🟢 clone — Vector duplication
fn bench_clone<V: Vector<T = f64> + Clone + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in ONE_SIZE {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let v = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                black_box(v.clone());
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🟢 as_view — Immutable view creation (trivial)
fn bench_as_view<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in ONE_SIZE {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let v = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                black_box(v.as_view());
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🟢 as_view_mut — Mutable view creation (trivial)
fn bench_as_view_mut<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in ONE_SIZE {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let mut v = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                black_box(v.as_view_mut());
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

/// 🟢 len — Vector length (trivial)
fn bench_len<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    bench_vector_prop::<V>(c, label, ONE_SIZE, |v| v.len() as f64);
}

/// 🟢 clone_as_vec — Export to Vec
fn bench_clone_as_vec<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in ONE_SIZE {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let v = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| v.clone_as_vec());
        });
    }
    group.finish();
}

/// 🟢 for_each_batch — Per-batch slice access, on the device where the backend has one
fn bench_for_each_batch<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in LANE_SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let x = V::from_element(ns, 1.0, ctx.clone());
            let mut y = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                y.for_each_batch([&x], copy_lane);
                ctx.synchronize();
                black_box(&y);
            });
        });
    }
    group.finish();
}

/// 🟢 for_each_batch_host — the same lane loop, always staged through host memory
fn bench_for_each_batch_host<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in LANE_SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let x = V::from_element(ns, 1.0, ctx.clone());
            let mut y = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                y.for_each_batch_host([&x], copy_lane);
                ctx.synchronize();
                black_box(&y);
            });
        });
    }
    group.finish();
}

/// 🟢 for_each_elem — the same copy, a thread per element on the device where the backend has one
fn bench_for_each_elem<V: Vector<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    V::C: Default + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in LANE_SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = V::C::default();
            let x = V::from_element(ns, 1.0, ctx.clone());
            let mut y = V::from_element(ns, 1.0, ctx.clone());
            b.iter(|| {
                y.for_each_elem([&x], copy_elem);
                ctx.synchronize();
                black_box(&y);
            });
        });
    }
    group.finish();
}

/// The lane body both `for_each_batch` benches run, so the two price the same work.
fn copy_lane(y: &mut [f64], [x]: [&[f64]; 1], _lane: usize) {
    for (y, x) in y.iter_mut().zip(x.iter()) {
        *y = *x;
    }
}

/// The element body of the same copy, so `for_each_elem` prices the work `copy_lane` does.
fn copy_elem(y: &mut f64, [x]: [&[f64]; 1], _lane: usize, i: usize) {
    *y = x[i];
}

/// 🟢 from_diagonal — Diagonal matrix creation
fn bench_from_diagonal<M: Matrix<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in ONE_SIZE {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = M::C::default();
            let v = M::V::from_element(ns, 2.0, ctx.clone());
            b.iter(|| {
                let m = M::from_diagonal(&v);
                ctx.synchronize();
                m
            });
        });
    }
    group.finish();
}

/// 🟢 add_column_to_vector — v += mat[:, j]
fn bench_add_column_to_vector<M: Matrix<T = f64> + 'static>(c: &mut Criterion, label: &str)
where
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C> + Clone,
{
    let mut group = c.benchmark_group(label);
    for &ns in MSIZES {
        group.bench_with_input(BenchmarkId::from_parameter(ns), &ns, |b, &ns| {
            let ctx = M::C::default();
            let mut mat = M::zeros(ns, ns, ctx.clone());
            fill_dense(&mut mat, ns);
            let mut v = M::V::zeros(ns, ctx.clone());
            b.iter(|| {
                mat.add_column_to_vector(0, &mut v);
                black_box(&v);
                ctx.synchronize();
            });
        });
    }
    group.finish();
}

// ═════════════════════════════════════════════════════════
// 🔴 LU — one factorisation per Jacobian update, one solve per Newton iteration
// ═════════════════════════════════════════════════════════

/// `(states, lanes)` cells the LU benches sweep. `robertson_ode` is 3 states per lane,
/// `heat2d` and `foodweb` 25 to 1800, so the cells bracket the point where per-lane work
/// starts to dominate launch overhead -- and no cell costs more than about a GFLOP, so the
/// CPU backends stay affordable.
const LU_CELLS: &[(usize, usize)] = &[
    (3, 1),
    (8, 1),
    (16, 1),
    (64, 1),
    (256, 1),
    (500, 1),
    (3, 100),
    (3, 1000),
    (32, 1),
    (32, 100),
    (32, 1000),
    (100, 1),
    (100, 100),
    (256, 100),
    (500, 10),
];
/// Parameters the grouped-broadcast solve mimics: the sensitivity equations run on
/// `nbatch * nparams` lanes against `nbatch` factorisations.
const LU_NPARAMS: usize = 4;

/// `A_b = I`, so a solve leaves the right-hand side unchanged and repeated `b.iter` solves
/// neither drift nor underflow. Factorisation cost does not depend on the values.
struct IdentityOp<M: Matrix> {
    matrix: M,
}

impl<M: Matrix> LaLinearOp for IdentityOp<M> {
    type T = M::T;
    type V = M::V;
    type M = M;
    type C = M::C;

    fn nrows(&self) -> IndexType {
        self.matrix.nrows()
    }
    fn ncols(&self) -> IndexType {
        self.matrix.ncols()
    }
    fn context(&self) -> &Self::C {
        self.matrix.context()
    }
    fn matrix_inplace(&self, y: &mut Self::M) {
        y.copy_from(&self.matrix);
    }
}

fn identity_op<M>(ns: usize, ctx: M::C) -> IdentityOp<M>
where
    M: Matrix<T = f64>,
    M::V: Vector<T = f64, C = M::C>,
{
    let diag = M::V::from_element(ns, 1.0, ctx);
    IdentityOp {
        matrix: M::from_diagonal(&diag),
    }
}

/// 🔴 lu_factor — refactorise on every Jacobian update
fn bench_lu_factor<M, LS>(c: &mut Criterion, label: &str)
where
    M: Matrix<T = f64> + 'static,
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C>,
    LS: LaLinearSolver<M>,
{
    let mut group = c.benchmark_group(label);
    group.sample_size(20);
    for &(ns, nb) in LU_CELLS {
        // the setup is inside the closure so that a filtered-out cell costs nothing
        group.bench_function(format!("n{ns}_nbatch{nb}"), |b| {
            let ctx = M::C::default()
                .clone_with_nbatch(nb)
                .expect("backend declined nbatch");
            let op = identity_op::<M>(ns, ctx.clone());
            let mut s = LS::default();
            s.set_sparsity(&op);
            b.iter(|| {
                s.set_linearisation(&op);
                ctx.synchronize();
            })
        });
    }
    group.finish();
}

/// 🔴 lu_solve — one solve per Newton iteration. `nparams > 1` is the grouped broadcast the
/// sensitivity equations run: `nbatch * nparams` right-hand side lanes over `nbatch`
/// factorisations.
fn bench_lu_solve<M, LS>(c: &mut Criterion, label: &str, nparams: usize)
where
    M: Matrix<T = f64> + 'static,
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C>,
    LS: LaLinearSolver<M>,
{
    let mut group = c.benchmark_group(label);
    group.sample_size(20);
    for &(ns, nb) in LU_CELLS {
        let id = if nparams == 1 {
            format!("n{ns}_nbatch{nb}")
        } else {
            format!("n{ns}_nbatch{nb}x{nparams}")
        };
        // the setup is inside the closure so that a filtered-out cell costs nothing
        group.bench_function(id, |b| {
            let ctx = M::C::default()
                .clone_with_nbatch(nb)
                .expect("backend declined nbatch");
            let rhs_ctx = ctx
                .clone_with_nbatch(nb * nparams)
                .expect("backend declined nbatch");
            let op = identity_op::<M>(ns, ctx.clone());
            let mut s = LS::default();
            s.set_sparsity(&op);
            s.set_linearisation(&op);
            let mut x = M::V::from_element(ns, 1.0, rhs_ctx);
            b.iter(|| {
                s.solve_in_place(&mut x).unwrap();
                black_box(&x);
                ctx.synchronize();
            })
        });
    }
    group.finish();
}

/// 🔴 lu_solve_sens — the solve pattern of a sensitivity step: the state's lanes and the
/// augmented `nbatch * nparams` lanes alternate against one factorisation, each with its own
/// vector, so a solver caching anything per right-hand side has to survive the switch
fn bench_lu_solve_sens<M, LS>(c: &mut Criterion, label: &str)
where
    M: Matrix<T = f64> + 'static,
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C>,
    LS: LaLinearSolver<M>,
{
    let mut group = c.benchmark_group(label);
    group.sample_size(20);
    for &(ns, nb) in LU_CELLS {
        group.bench_function(format!("n{ns}_nbatch{nb}x{LU_NPARAMS}"), |b| {
            let ctx = M::C::default()
                .clone_with_nbatch(nb)
                .expect("backend declined nbatch");
            let aug_ctx = ctx
                .clone_with_nbatch(nb * LU_NPARAMS)
                .expect("backend declined nbatch");
            let op = identity_op::<M>(ns, ctx.clone());
            let mut s = LS::default();
            s.set_sparsity(&op);
            s.set_linearisation(&op);
            let mut x = M::V::from_element(ns, 1.0, ctx.clone());
            let mut x_aug = M::V::from_element(ns, 1.0, aug_ctx);
            b.iter(|| {
                s.solve_in_place(&mut x).unwrap();
                s.solve_in_place(&mut x_aug).unwrap();
                black_box((&x, &x_aug));
                ctx.synchronize();
            })
        });
    }
    group.finish();
}

/// `(states, lanes)` cells the Newton bench sweeps. The switch cost it prices is a fixed
/// per-solve overhead, so wide matrices would only bury it.
const NEWTON_CELLS: &[(usize, usize)] = &[(3, 100), (3, 1000), (32, 100)];

/// `F(x) = 2 x .* x - 8`, elementwise so that every operation is one launch whatever the lane
/// count.
///
/// Elementwise on purpose: the sensitivity test models are host-slice closures that stage
/// through host memory on a device backend, and a `gemv`-based operator would price
/// `OxideMat::gemv`'s per-lane cuBLAS loop instead -- either would bury what this bench is
/// for, which is the per-solve overhead of switching lane counts.
struct SquareOp<M: DenseMatrix> {
    nstates: IndexType,
    eights: M::V,
    ctx: M::C,
}

impl<M: DenseMatrix<T = f64>> SquareOp<M>
where
    M::V: Vector<T = f64, C = M::C>,
    M::C: Clone,
{
    fn new(nstates: usize, ctx: M::C) -> Self {
        let eights = M::V::from_element(nstates, 8.0, ctx.clone());
        Self {
            nstates,
            eights,
            ctx,
        }
    }
}

impl<M: DenseMatrix<T = f64>> NonLinearOp for SquareOp<M>
where
    M::V: Vector<T = f64, C = M::C>,
{
    type T = f64;
    type V = M::V;
    type M = M;
    type C = M::C;

    fn nstates(&self) -> IndexType {
        self.nstates
    }
    fn nout(&self) -> IndexType {
        self.nstates
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }
    fn call_inplace(&self, x: &Self::V, y: &mut Self::V) {
        // y = 2 x .* x - 8
        y.copy_from(x);
        y.component_mul_assign(x);
        *y *= Scale(2.0);
        y.axpy(-1.0, &self.eights, 1.0);
    }
}

impl<M: DenseMatrix<T = f64>> NonLinearOpJacobian for SquareOp<M>
where
    M::V: Vector<T = f64, C = M::C>,
{
    fn jac_mul_inplace(&self, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        // J v = 4 x .* v
        y.copy_from(x);
        y.component_mul_assign(v);
        *y *= Scale(4.0);
    }
}

/// 🔴 newton_sens — a sensitivity step's two Newton solves: the state's lanes then the
/// augmented `nbatch * nparams`, against one Jacobian, sharing one solver
fn bench_newton_sens<M, LS>(c: &mut Criterion, label: &str)
where
    M: DenseMatrix<T = f64> + 'static,
    M::C: Default + Clone,
    M::V: Vector<T = f64, C = M::C>,
    LS: LaLinearSolver<M>,
{
    let mut group = c.benchmark_group(label);
    group.sample_size(20);
    for &(ns, nb) in NEWTON_CELLS {
        group.bench_function(format!("n{ns}_nbatch{nb}x{LU_NPARAMS}"), |b| {
            let ctx = M::C::default()
                .clone_with_nbatch(nb)
                .expect("backend declined nbatch");
            let aug_ctx = ctx
                .clone_with_nbatch(nb * LU_NPARAMS)
                .expect("backend declined nbatch");
            let op = SquareOp::<M>::new(ns, ctx.clone());
            let atol = M::V::from_element(ns, 1e-6, ctx.clone());
            let x0 = M::V::from_element(ns, 2.1, ctx.clone());
            let x0_aug = M::V::from_element(ns, 2.1, aug_ctx.clone());
            let mut s =
                NewtonNonlinearSolver::<M, LS, NoLineSearch>::new(LS::default(), NoLineSearch);
            s.set_problem(&op);
            s.reset_jacobian(&op, &x0);
            let mut x = x0.clone();
            let mut x_aug = x0_aug.clone();
            let mut convergence = Convergence::new(1e-6, &atol);
            b.iter(|| {
                x.copy_from(&x0);
                x_aug.copy_from(&x0_aug);
                s.solve_in_place(&op, &mut x, &x0, &mut convergence)
                    .unwrap();
                s.solve_in_place(&op, &mut x_aug, &x0_aug, &mut convergence)
                    .unwrap();
                black_box((&x, &x_aug));
                ctx.synchronize();
            })
        });
    }
    group.finish();
}

// ═════════════════════════════════════════════════════════
// Backend macros — add new backends with one call each
// ═════════════════════════════════════════════════════════

macro_rules! bench_vector_backend {
    ($c:expr, $label:expr, $V:ty) => {
        bench_axpy::<$V>($c, concat!("axpy/", $label));
        bench_copy_from::<$V>($c, concat!("copy_from/", $label));
        bench_sub_assign::<$V>($c, concat!("sub_assign/", $label));
        bench_add_assign::<$V>($c, concat!("add_assign/", $label));
        bench_add_ref_ref::<$V>($c, concat!("add_ref_ref/", $label));
        bench_sub_ref_ref::<$V>($c, concat!("sub_ref_ref/", $label));
        bench_add_owned_lhs::<$V>($c, concat!("add_owned_lhs/", $label));
        bench_sub_owned_lhs::<$V>($c, concat!("sub_owned_lhs/", $label));
        bench_squared_norm::<$V>($c, concat!("squared_norm/", $label));
        bench_axpy_v::<$V>($c, concat!("axpy_v/", $label));
        bench_copy_from_view::<$V>($c, concat!("copy_from_view/", $label));
        bench_fill::<$V>($c, concat!("fill/", $label));
        bench_scalar_mul::<$V>($c, concat!("scalar_mul/", $label));
        bench_scalar_mul_assign::<$V>($c, concat!("scalar_mul_assign/", $label));
        bench_scalar_div::<$V>($c, concat!("scalar_div/", $label));
        bench_component_mul_assign::<$V>($c, concat!("component_mul_assign/", $label));
        bench_component_div_assign::<$V>($c, concat!("component_div_assign/", $label));
        bench_norm::<$V>($c, concat!("norm_l2/", $label));
        bench_set_index::<$V>($c, concat!("set_index/", $label));
        bench_get_index::<$V>($c, concat!("get_index/", $label));
        bench_root_finding::<$V>($c, concat!("root_finding/", $label));
        bench_from_element::<$V>($c, concat!("from_element/", $label));
        bench_from_vec::<$V>($c, concat!("from_vec/", $label));
        bench_zeros::<$V>($c, concat!("zeros/", $label));
        bench_clone::<$V>($c, concat!("clone/", $label));
        bench_as_view::<$V>($c, concat!("as_view/", $label));
        bench_as_view_mut::<$V>($c, concat!("as_view_mut/", $label));
        bench_len::<$V>($c, concat!("len/", $label));
        bench_clone_as_vec::<$V>($c, concat!("clone_as_vec/", $label));
        bench_for_each_batch::<$V>($c, concat!("for_each_batch/", $label));
        bench_for_each_batch_host::<$V>($c, concat!("for_each_batch_host/", $label));
        bench_for_each_elem::<$V>($c, concat!("for_each_elem/", $label));
    };
}

macro_rules! bench_matrix_backend {
    ($c:expr, $label:expr, $M:ty) => {
        bench_gemv::<$M>($c, concat!("gemv/", $label));
        bench_set_column::<$M>($c, concat!("set_column/", $label));
        bench_scale_add_and_assign::<$M>($c, concat!("scale_add_and_assign/", $label));
        bench_matrix_copy_from::<$M>($c, concat!("matrix_copy_from/", $label));
        bench_from_diagonal::<$M>($c, concat!("from_diagonal/", $label));
        bench_add_column_to_vector::<$M>($c, concat!("add_column_to_vector/", $label));
    };
}

/// Backends whose context takes `nbatch > 1`; the sparse backend is not one of them.
macro_rules! bench_batched_matrix_backend {
    ($c:expr, $label:expr, $M:ty) => {
        bench_gemv_batched::<$M>($c, concat!("gemv_batched/", $label), LU_CELLS, 1, false);
        bench_gemv_batched::<$M>(
            $c,
            concat!("gemv_batched_grouped/", $label),
            LU_CELLS,
            LU_NPARAMS,
            false,
        );
        bench_gemv_batched::<$M>(
            $c,
            concat!("gemv_batched_grouped_x/", $label),
            LU_CELLS,
            LU_NPARAMS,
            true,
        );
        bench_gemv_batched::<$M>(
            $c,
            concat!("gemv_batched_grouped100/", $label),
            GEMV_WIDE_CELLS,
            100,
            false,
        );
    };
}

macro_rules! bench_dense_matrix_backend {
    ($c:expr, $label:expr, $M:ty) => {
        bench_matrix_column::<$M>($c, concat!("matrix_column/", $label));
        bench_gemv_cols::<$M>($c, concat!("gemv_cols/", $label));
        bench_stage_accumulate::<$M>($c, concat!("stage_accumulate/", $label));
        bench_mul_cols_by::<$M>($c, concat!("mul_cols_by/", $label));
    };
}

macro_rules! bench_lu_backend {
    ($c:expr, $label:expr, $M:ty, $LS:ty) => {
        bench_lu_factor::<$M, $LS>($c, concat!("lu_factor/", $label));
        bench_lu_solve::<$M, $LS>($c, concat!("lu_solve/", $label), 1);
        bench_lu_solve::<$M, $LS>($c, concat!("lu_solve_grouped/", $label), LU_NPARAMS);
        bench_lu_solve_sens::<$M, $LS>($c, concat!("lu_solve_sens/", $label));
        bench_newton_sens::<$M, $LS>($c, concat!("newton_sens/", $label));
    };
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_vector_backend!(c, "nalgebra", NalgebraVec<f64>);
    bench_matrix_backend!(c, "nalgebra", NalgebraMat<f64>);
    bench_dense_matrix_backend!(c, "nalgebra", NalgebraMat<f64>);
    bench_batched_matrix_backend!(c, "nalgebra", NalgebraMat<f64>);
    bench_lu_backend!(c, "nalgebra", NalgebraMat<f64>, NalgebraLU<f64>);

    bench_vector_backend!(c, "faer", FaerVec<f64>);
    bench_matrix_backend!(c, "faer", FaerMat<f64>);
    bench_dense_matrix_backend!(c, "faer", FaerMat<f64>);
    bench_batched_matrix_backend!(c, "faer", FaerMat<f64>);
    bench_lu_backend!(c, "faer", FaerMat<f64>, FaerLU<f64>);

    bench_matrix_backend!(c, "faer_sparse", FaerSparseMat<f64>);

    #[cfg(feature = "cuda")]
    {
        bench_vector_backend!(c, "cuda", CudaVec<f64>);
        bench_matrix_backend!(c, "cuda", CudaMat<f64>);
        bench_dense_matrix_backend!(c, "cuda", CudaMat<f64>);
        bench_batched_matrix_backend!(c, "cuda", CudaMat<f64>);
        bench_lu_backend!(c, "cuda", CudaMat<f64>, CudaLU<f64>);
    }

    #[cfg(feature = "cuda-oxide")]
    {
        bench_vector_backend!(c, "cuda_oxide", OxideVec);
        bench_matrix_backend!(c, "cuda_oxide", OxideMat);
        bench_dense_matrix_backend!(c, "cuda_oxide", OxideMat);
        bench_batched_matrix_backend!(c, "cuda_oxide", OxideMat);
        bench_lu_backend!(c, "cuda_oxide", OxideMat, OxideLU);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
