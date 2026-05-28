use diffsol::{
    CraneliftJitModule, FaerContext, FaerSparseLU, FaerSparseMat, FaerVec,
    LinearSolver, Matrix, NonLinearOp, NonLinearOpJacobian, OdeBuilder,
    OdeEquations, OdeSolverMethod, OdeSolverStopReason, Op, Vector, VectorHost,
};
use diffsol::ConstantOp as _;
use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use std::cell::RefCell;
use std::fs;

type M = FaerSparseMat<f64>;
type V = FaerVec<f64>;
type LS = FaerSparseLU<f64>;
type CG = CraneliftJitModule;

/// Wrapper for a constant matrix usable with LinearSolver
struct ConstMatOp {
    mat: RefCell<M>,
    ctx: FaerContext,
    nstates_val: usize,
    nout_val: usize,
    sparsity_val: Option<faer::sparse::SymbolicSparseColMat<usize>>,
}

impl Op for ConstMatOp {
    type T = f64;
    type V = V;
    type M = M;
    type C = FaerContext;
    fn nstates(&self) -> usize { self.nstates_val }
    fn nout(&self) -> usize { self.nout_val }
    fn nparams(&self) -> usize { 0 }
    fn context(&self) -> &Self::C { &self.ctx }
}

impl NonLinearOp for ConstMatOp {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, _y: &mut Self::V) {
        panic!("call_inplace not needed for linear solve");
    }
}

impl NonLinearOpJacobian for ConstMatOp {
    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, _y: &mut Self::V) {
        panic!("jac_mul_inplace not needed");
    }
    fn jacobian_inplace(&self, _x: &Self::V, _t: Self::T, y: &mut Self::M) {
        y.copy_from(&self.mat.borrow());
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.sparsity_val.clone()
    }
}

fn load_mat(prefix: &str, ctx: &FaerContext) -> M {
    let dims: Array1<i64> = read_npy::<_, Array1<i64>>(format!("{prefix}_dims.npy")).unwrap();
    let nrows = dims[0] as usize;
    let ncols = dims[1] as usize;
    let i: Array1<i64> = read_npy::<_, Array1<i64>>(format!("{prefix}_i.npy")).unwrap();
    let j: Array1<i64> = read_npy::<_, Array1<i64>>(format!("{prefix}_j.npy")).unwrap();
    let v: Array1<f64> = read_npy::<_, Array1<f64>>(format!("{prefix}_v.npy")).unwrap();

    let triplets: Vec<(usize, usize, f64)> = (0..i.len())
        .map(|k| (i[k] as usize, j[k] as usize, v[k]))
        .collect();

    M::try_from_triplets(nrows, ncols, triplets, ctx.clone()).expect("Failed matrix")
}

fn factor_square(prefix: &str, ctx: &FaerContext) -> LS {
    let mat = load_mat(prefix, ctx);
    let nrows = mat.sparsity().map(|s| s.nrows()).unwrap_or(0);
    let ncols = mat.sparsity().map(|s| s.ncols()).unwrap_or(0);

    let sparsity = mat.sparsity().map(|s| s.to_owned()).transpose().ok().flatten();
    let op = ConstMatOp {
        nstates_val: ncols,
        nout_val: nrows,
        sparsity_val: sparsity,
        ctx: ctx.clone(),
        mat: RefCell::new(mat),
    };
    let mut solver = LS::default();
    solver.set_problem(&op);
    solver.set_linearisation(&op, &V::zeros(ncols, ctx.clone()), 0.0);
    solver
}

fn main() {
    let solver_name = std::env::args().nth(1).unwrap_or_else(|| "tr_bdf2".to_string());

    let code = fs::read_to_string("taylor_green.dsl").expect("failed to read DSL");
    let m_lumped: Array1<f64> = read_npy::<_, Array1<f64>>("m_lumped.npy").expect("failed to read m_lumped");

    let mut problem = OdeBuilder::<M>::new()
        .rtol(1e-6)
        .atol([1e-8])
        .build_from_diffsl::<CG>(&code)
        .unwrap();

    let ctx = problem.context().clone();
    let n_u = problem.eqn().rhs().nstates();
    let n_p: usize = read_npy::<_, Array1<i64>>("L_dims.npy").unwrap()[0] as usize;

    let g = load_mat("G", &ctx);
    let gt = load_mat("GT", &ctx);
    let mut l_solver = factor_square("L", &ctx);

    let mut div = V::zeros(n_p, ctx.clone());
    let mut g_phi = V::zeros(n_u, ctx.clone());
    let mut tmp = V::zeros(n_u, ctx.clone());
    let mut u_tmp = V::zeros(n_u, ctx.clone());

    let t_final = 5.0;
    let n_save = 201;
    let mut sol = ndarray::Array2::<f64>::zeros((n_u + n_p, n_save));
    let mut ts_save = vec![0.0f64; n_save];

    let init_y = problem.eqn().init().call(0.0);
    let init_dy = problem.eqn().rhs().call(&init_y, 0.0);

    let mut step_count = 0usize;
    let dt_save = t_final / (n_save - 1) as f64;
    let start = std::time::Instant::now();

    // Build solver using a match on name
    macro_rules! run_solver {
        ($solver:expr) => {{
            let mut solver = $solver;
            {
                let state = solver.state_mut();
                state.y.copy_from(&init_y);
                state.dy.copy_from(&init_dy);
                *state.t = 0.0;
                *state.h = dt_save;
            }

            for (dest, src) in sol.column_mut(0).slice_mut(ndarray::s![0..n_u]).iter_mut().zip(init_y.as_slice()) {
                *dest = *src;
            }
            ts_save[0] = 0.0;
            let mut save_idx = 1;
            let mut t_save_next = dt_save;

            loop {
                let reason = solver.step().expect("step");
                step_count += 1;
                let t = solver.state().t;
                let u: Vec<f64> = solver.state().y.as_slice().to_vec();
                let h = solver.state().h;

                // Compute divergence: div = GT * (m_lumped ⊙ u)
                for j in 0..n_u {
                    tmp.as_mut_slice()[j] = m_lumped[j] * u[j];
                }
                gt.gemv(1.0, &tmp, 0.0, &mut div);

                // Pressure Poisson: L * phi = div / h
                let mut rhs = V::zeros(n_p, ctx.clone());
                for j in 0..n_p {
                    rhs.as_mut_slice()[j] = div.as_slice()[j] / h;
                }
                let phi = l_solver.solve(&rhs).expect("L solve");

                // Project: u_next = u - h * G * phi
                g.gemv(1.0, &phi, 0.0, &mut g_phi);
                for j in 0..n_u {
                    u_tmp.as_mut_slice()[j] = u[j] - h * g_phi.as_slice()[j];
                }

                {
                    let state = solver.state_mut();
                    state.y.copy_from(&u_tmp);
                }

                while t >= t_save_next - 1e-12 && save_idx < n_save {
                    for (dest, src) in sol.column_mut(save_idx).slice_mut(ndarray::s![0..n_u]).iter_mut().zip(u_tmp.as_slice()) {
                        *dest = *src;
                    }
                    for (dest, src) in sol.column_mut(save_idx).slice_mut(ndarray::s![n_u..n_u + n_p]).iter_mut().zip(phi.as_slice()) {
                        *dest = *src;
                    }
                    ts_save[save_idx] = t;
                    save_idx += 1;
                    t_save_next = save_idx as f64 * dt_save;
                }

                if matches!(reason, OdeSolverStopReason::TstopReached) && t >= t_final - 1e-12 {
                    break;
                }
                if t >= t_final - 1e-12 {
                    break;
                }
                solver.set_stop_time(t_final).expect("set_stop_time");
            }

            let elapsed = start.elapsed();
            println!("[{solver_name}] {step_count} steps, t = {:.6}, wall: {:.2}s", solver.state().t, elapsed.as_secs_f64());
            save_idx
        }};
    }

    let save_idx = match solver_name.as_str() {
        "esdirk34" => run_solver!(problem.esdirk34::<LS>().unwrap()),
        "tr_bdf2" => run_solver!(problem.tr_bdf2::<LS>().unwrap()),
        "tsit45" => run_solver!(problem.tsit45().unwrap()),
        other => panic!("Unknown solver: {other}. Use esdirk34, tr_bdf2, or tsit45"),
    };

    let sol_slice = sol.slice(ndarray::s![.., ..save_idx]);
    let sol_trimmed = sol_slice.to_owned();
    let ts_trimmed = Array1::from_vec(ts_save[..save_idx].to_vec());
    write_npy("solution.npy", &sol_trimmed).expect("write solution");
    write_npy("time.npy", &ts_trimmed).expect("write time");
}
