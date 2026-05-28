use diffsol::{
    ConstantOp as _, CraneliftJitModule, FaerContext, FaerSparseLU, FaerSparseMat, FaerVec,
    LinearSolver, Matrix, NonLinearOp, NonLinearOpJacobian,
    OdeBuilder, OdeEquations, OdeSolverMethod, Op, Vector, VectorHost,
};
use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use std::cell::RefCell;
use std::fs;

type M = FaerSparseMat<f64>;
type V = FaerVec<f64>;
type LS = FaerSparseLU<f64>;
type CG = CraneliftJitModule;

// --- Load / factor helpers ---

fn load(prefix: &str, ctx: &FaerContext) -> M {
    let dims: Array1<i64> = read_npy::<_, Array1<i64>>(format!("{prefix}_dims.npy")).unwrap();
    let i: Array1<i64> = read_npy::<_, Array1<i64>>(format!("{prefix}_i.npy")).unwrap();
    let j: Array1<i64> = read_npy::<_, Array1<i64>>(format!("{prefix}_j.npy")).unwrap();
    let v: Array1<f64> = read_npy::<_, Array1<f64>>(format!("{prefix}_v.npy")).unwrap();
    let triplets: Vec<(usize, usize, f64)> = (0..i.len())
        .map(|k| (i[k] as usize, j[k] as usize, v[k]))
        .collect();
    M::try_from_triplets(dims[0] as usize, dims[1] as usize, triplets, ctx.clone()).unwrap()
}

struct ConstOp {
    mat: RefCell<M>,
    ctx: FaerContext,
    n: usize,
    sp: Option<<M as Matrix>::Sparsity>,
}
impl Op for ConstOp {
    type T = f64; type V = V; type M = M; type C = FaerContext;
    fn nstates(&self) -> usize { self.n }
    fn nout(&self) -> usize { self.n }
    fn nparams(&self) -> usize { 0 }
    fn context(&self) -> &Self::C { &self.ctx }
}
impl NonLinearOp for ConstOp {
    fn call_inplace(&self, _x: &V, _t: f64, _y: &mut V) {}
}
impl NonLinearOpJacobian for ConstOp {
    fn jac_mul_inplace(&self, _x: &V, _t: f64, _v: &V, _y: &mut V) {}
    fn jacobian_inplace(&self, _x: &V, _t: f64, y: &mut M) { y.copy_from(&self.mat.borrow()); }
    fn jacobian_sparsity(&self) -> Option<<M as Matrix>::Sparsity> { self.sp.clone() }
}

fn factor(mat: M) -> LS {
    let n = mat.sparsity().map(|s| s.nrows()).unwrap_or(0);
    let sp = mat.sparsity().map(|s| s.to_owned()).transpose().ok().flatten();
    let op = ConstOp { n, sp, ctx: mat.context().clone(), mat: RefCell::new(mat) };
    let mut solver = LS::default();
    solver.set_problem(&op);
    solver.set_linearisation(&op, &V::zeros(n, op.ctx.clone()), 0.0);
    solver
}

// --- Main ---

fn main() {
    let code = fs::read_to_string("taylor_green.dsl").unwrap();
    let m_lumped: Array1<f64> = read_npy::<_, Array1<f64>>("m_lumped.npy").unwrap();

    let problem = OdeBuilder::<M>::new().rtol(1e-6).atol([1e-8])
        .build_from_diffsl::<CG>(&code).unwrap();

    let ctx = problem.context().clone();
    let n_u = problem.eqn().rhs().nstates();
    let n_p = read_npy::<_, Array1<i64>>("L_dims.npy").unwrap()[0] as usize;

    let g  = load("G",  &ctx);    // G  = M⁻¹ Bᵀ  (n_u × n_p)
    let gt = load("GT", &ctx);    // GT = B M⁻¹   (n_p × n_u)
    let l_solver = factor(load("L", &ctx));  // L = B M⁻¹ Bᵀ  (factored)

    let mut div   = V::zeros(n_p, ctx.clone());
    let mut g_phi = V::zeros(n_u, ctx.clone());
    let mut tmp   = V::zeros(n_u, ctx.clone());

    let t_final = 5.0;
    let n_save = 201;
    let dt_save = t_final / (n_save - 1) as f64;
    let mut sol = ndarray::Array2::<f64>::zeros((n_u + n_p, n_save));
    let mut ts_save = vec![0.0; n_save];

    let init_y = problem.eqn().init().call(0.0);
    let init_dy = problem.eqn().rhs().call(&init_y, 0.0);
    let mut solver = problem.tr_bdf2::<LS>().unwrap();
    {
        let s = solver.state_mut();
        s.y.copy_from(&init_y); s.dy.copy_from(&init_dy);
        *s.t = 0.0; *s.h = dt_save;
    }
    sol.column_mut(0).slice_mut(ndarray::s![0..n_u]).iter_mut()
        .zip(init_y.as_slice()).for_each(|(d, s)| *d = *s);
    ts_save[0] = 0.0;
    let mut save_idx = 1;
    let mut t_save_next = dt_save;

    let mut u = vec![0.0; n_u];
    let start = std::time::Instant::now();
    let mut step_count = 0;
    let mut t;

    loop {
        solver.step().unwrap();
        step_count += 1;
        t = solver.state().t;
        let h = solver.state().h;

        let u_s = solver.state().y.as_slice();
        u.copy_from_slice(u_s);

        for j in 0..n_u { tmp.as_mut_slice()[j] = m_lumped[j] * u[j]; }
        gt.gemv(1.0, &tmp, 0.0, &mut div);

        let mut rhs = V::zeros(n_p, ctx.clone());
        for j in 0..n_p { rhs.as_mut_slice()[j] = div.as_slice()[j] / h; }
        let phi = l_solver.solve(&rhs).unwrap();

        g.gemv(1.0, &phi, 0.0, &mut g_phi);
        for j in 0..n_u { u[j] -= h * g_phi.as_slice()[j]; }
        solver.state_mut().y.as_mut_slice().copy_from_slice(&u);

        while t >= t_save_next - 1e-12 && save_idx < n_save {
            let mut col = sol.column_mut(save_idx);
            col.slice_mut(ndarray::s![0..n_u]).iter_mut().zip(u.iter())
                .for_each(|(d, s)| *d = *s);
            col.slice_mut(ndarray::s![n_u..n_u + n_p]).iter_mut().zip(phi.as_slice().iter())
                .for_each(|(d, s)| *d = *s);
            ts_save[save_idx] = t;
            save_idx += 1;
            t_save_next = save_idx as f64 * dt_save;
        }

        if t >= t_final - 1e-12 { break; }
        solver.set_stop_time(t_final).unwrap();
    }

    println!("tr_bdf2: {step_count} steps, t = {:.6}, wall = {:.2}s",
             t, start.elapsed().as_secs_f64());

    let trimmed = sol.slice(ndarray::s![.., ..save_idx]).to_owned();
    write_npy("solution.npy", &trimmed).unwrap();
    write_npy("time.npy", &Array1::from_vec(ts_save[..save_idx].to_vec())).unwrap();
}
