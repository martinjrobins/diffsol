use diffsol::{
    CraneliftJitModule, FaerContext, FaerSparseLU, FaerSparseMat, FaerVec,
    LinearSolver, Matrix, NonLinearOp, NonLinearOpJacobian, OdeBuilder,
    OdeEquations, OdeSolverMethod, Op, Vector, VectorHost,
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

    let G = load_mat("G", &ctx);
    let GT = load_mat("GT", &ctx);
    let mut l_solver = factor_square("L", &ctx);

    let mut div = V::zeros(n_p, ctx.clone());
    let mut g_phi = V::zeros(n_u, ctx.clone());
    let mut tmp = V::zeros(n_u, ctx.clone());
    let mut u_tmp = V::zeros(n_u, ctx.clone());

    let dt = 0.1;
    let t_final = 5.0;
    let n_steps = (t_final / dt) as usize;
    let save_every = (n_steps / 200).max(1);
    let n_save = n_steps / save_every + 1;
    let mut sol = ndarray::Array2::<f64>::zeros((n_u + n_p, n_save));
    let mut ts_save = vec![0.0f64; n_save];

    let init_y = problem.eqn().init().call(0.0);
    let init_dy = problem.eqn().rhs().call(&init_y, 0.0);

    let mut solver = problem.bdf::<LS>().unwrap();
    {
        let state = solver.state_mut();
        state.y.copy_from(&init_y);
        state.dy.copy_from(&init_dy);
        *state.t = 0.0;
        *state.h = dt;
    }

    for (dest, src) in sol.column_mut(0).slice_mut(ndarray::s![0..n_u]).iter_mut().zip(init_y.as_slice()) {
        *dest = *src;
    }
    ts_save[0] = 0.0;
    let mut save_idx = 1;

    for step in 0..n_steps {
        let t_next = (step + 1) as f64 * dt;

        solver.set_stop_time(t_next).expect("set_stop_time");
        let (_ys, _times, _reason) = solver.solve(t_next).expect("solve");

        let u_star: Vec<f64> = solver.state().y.as_slice().to_vec();

        for j in 0..n_u {
            tmp.as_mut_slice()[j] = m_lumped[j] * u_star[j];
        }
        GT.gemv(1.0, &tmp, 0.0, &mut div);

        let mut rhs = V::zeros(n_p, ctx.clone());
        for j in 0..n_p {
            rhs.as_mut_slice()[j] = div.as_slice()[j] / dt;
        }
        let phi = l_solver.solve(&rhs).expect("L solve");

        G.gemv(1.0, &phi, 0.0, &mut g_phi);
        for j in 0..n_u {
            u_tmp.as_mut_slice()[j] = u_star[j] - dt * g_phi.as_slice()[j];
        }

        {
            let state = solver.state_mut();
            state.y.copy_from(&u_tmp);
            *state.t = t_next;
            *state.h = dt;
        }

        if step % save_every == 0 || step == n_steps - 1 {
            for (dest, src) in sol.column_mut(save_idx).slice_mut(ndarray::s![0..n_u]).iter_mut().zip(u_tmp.as_slice()) {
                *dest = *src;
            }
            for (dest, src) in sol.column_mut(save_idx).slice_mut(ndarray::s![n_u..n_u + n_p]).iter_mut().zip(phi.as_slice()) {
                *dest = *src;
            }
            ts_save[save_idx] = t_next;
            save_idx += 1;
        }
    }

    let sol_slice = sol.slice(ndarray::s![.., ..save_idx]);
    let sol_trimmed = sol_slice.to_owned();
    let ts_trimmed = Array1::from_vec(ts_save[..save_idx].to_vec());
    write_npy("solution.npy", &sol_trimmed).expect("write solution");
    write_npy("time.npy", &ts_trimmed).expect("write time");
    println!(
        "Wrote solution.npy ({} x {}) and time.npy ({})",
        sol_trimmed.nrows(),
        sol_trimmed.ncols(),
        ts_trimmed.len()
    );
}
