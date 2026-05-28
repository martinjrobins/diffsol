use diffsol::{
    ConstantOp as _, CraneliftJitModule, FaerSparseLU, FaerSparseMat,
    NonLinearOp, OdeBuilder, OdeEquations, OdeSolverMethod, Op, Vector, VectorHost,
};
use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use std::fs;

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};

type M = FaerSparseMat<f64>;
type LS = FaerSparseLU<f64>;
type CG = CraneliftJitModule;

// --- Sparse matrix helpers using raw faer ---

/// Load sparse matrix from NPY triplet files
fn load(prefix: &str) -> SparseColMat<usize, f64> {
    let dims: Array1<i64> = read_npy::<_, Array1<i64>>(format!("{prefix}_dims.npy")).unwrap();
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let i: Array1<i64> = read_npy::<_, Array1<i64>>(format!("{prefix}_i.npy")).unwrap();
    let j: Array1<i64> = read_npy::<_, Array1<i64>>(format!("{prefix}_j.npy")).unwrap();
    let v: Array1<f64> = read_npy::<_, Array1<f64>>(format!("{prefix}_v.npy")).unwrap();
    let triplets: Vec<Triplet<usize, usize, f64>> = (0..i.len())
        .map(|k| Triplet::new(i[k] as usize, j[k] as usize, v[k]))
        .collect();
    SparseColMat::try_new_from_triplets(rows, cols, &triplets).unwrap()
}

/// y = alpha * A * x (CSC format)
fn gemv(alpha: f64, a: &SparseColMat<usize, f64>, x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    for j in 0..a.ncols() {
        let xj = alpha * x[j];
        if xj == 0.0 { continue; }
        let col = a.col_range(j);
        if col.is_empty() { continue; }
        let rows = &a.row_idx()[col.start..col.end];
        let vals = &a.val()[col.start..col.end];
        for (&row, &val) in rows.iter().zip(vals.iter()) {
            y[row] += val * xj;
        }
    }
}

/// Solve L * x = b (factored sparse matrix)
fn lu_solve(lu: &Lu<usize, f64>, b: &[f64]) -> Vec<f64> {
    let mut x = b.to_vec();
    let n = x.len();
    unsafe {
        lu.solve_in_place(
            faer::MatMut::from_raw_parts_mut(x.as_mut_ptr(), n, 1, 1, n as isize)
        );
    }
    x
}

// --- Main ---

fn main() {
    let code = fs::read_to_string("taylor_green.dsl").unwrap();
    let m_lumped: Array1<f64> = read_npy::<_, Array1<f64>>("m_lumped.npy").unwrap();

    let problem = OdeBuilder::<M>::new().rtol(1e-6).atol([1e-8])
        .build_from_diffsl::<CG>(&code).unwrap();

    let n_u = problem.eqn().rhs().nstates();
    let n_p = read_npy::<_, Array1<i64>>("L_dims.npy").unwrap()[0] as usize;

    // Load and factor projection operators (raw faer)
    let g  = load("G");          // G  = M⁻¹ Bᵀ  (n_u × n_p)
    let gt = load("GT");         // GT = B M⁻¹   (n_p × n_u)
    let l_mat = load("L");       // L  = B M⁻¹ Bᵀ (n_p × n_p)
    let l_lu = {
        let sym = SymbolicLu::try_new(l_mat.symbolic()).unwrap();
        Lu::try_new_with_symbolic(sym, l_mat.as_ref()).unwrap()
    };

    // Work buffers
    let mut div = vec![0.0; n_p];
    let mut g_phi = vec![0.0; n_u];
    let mut tmp = vec![0.0; n_u];
    let mut u = vec![0.0; n_u];

    // Output storage
    let t_final = 5.0;
    let n_save = 201;
    let dt_save = t_final / (n_save - 1) as f64;
    let mut sol = ndarray::Array2::<f64>::zeros((n_u + n_p, n_save));
    let mut ts_save = vec![0.0; n_save];

    // Initialise TR-BDF2 solver
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

    // Time-stepping loop
    let mut save_idx = 1;
    let mut t_save_next = dt_save;
    let start = std::time::Instant::now();
    let mut step_count = 0;
    let mut t;

    loop {
        solver.step().unwrap();
        step_count += 1;
        t = solver.state().t;
        let h = solver.state().h;

        u.copy_from_slice(solver.state().y.as_slice());

        // div = GT * (m_lumped ⊙ u)
        for j in 0..n_u { tmp[j] = m_lumped[j] * u[j]; }
        gemv(1.0, &gt, &tmp, &mut div);

        // Project: u = u - h * G * phi  where  L * phi = div / h
        let phi = lu_solve(&l_lu, &div.iter().map(|d| d / h).collect::<Vec<_>>());
        gemv(1.0, &g, &phi, &mut g_phi);
        for j in 0..n_u { u[j] -= h * g_phi[j]; }
        solver.state_mut().y.as_mut_slice().copy_from_slice(&u);

        // Save output
        while t >= t_save_next - 1e-12 && save_idx < n_save {
            let mut col = sol.column_mut(save_idx);
            col.slice_mut(ndarray::s![0..n_u]).iter_mut().zip(u.iter())
                .for_each(|(d, s)| *d = *s);
            col.slice_mut(ndarray::s![n_u..n_u + n_p]).iter_mut().zip(phi.iter())
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
