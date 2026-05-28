use diffsol::{
    CraneliftJitModule, FaerSparseLU, FaerSparseMat, OdeBuilder,
    OdeEquations, OdeSolverMethod, Op, Vector, VectorCommon,
};
use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use std::fs;

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Col, unzip, zip};

type M = FaerSparseMat<f64>;
type LS = FaerSparseLU<f64>;
type CG = CraneliftJitModule;

// --- Sparse matrix helpers ---

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

/// Load vector from NPY file
fn load_vec(name: &str) -> Col<f64> {
    let arr: Array1<f64> = read_npy::<_, Array1<f64>>(format!("{name}.npy")).unwrap();
    Col::from_iter(arr)
}

/// Save 1 or 2 faer Cols to column i in the Array2, if i > len then double the size of the Array2 and save to column i
fn save_vec(arr: &mut ndarray::Array2<f64>, i: usize, vec: &Col<f64>, vec2: Option<&Col<f64>>) {
    if i >= arr.ncols() {
        let new_ncols = arr.ncols() * 2;
        let mut new_arr = ndarray::Array2::<f64>::zeros((arr.nrows(), new_ncols));
        new_arr.slice_mut(ndarray::s![.., ..arr.ncols()]).assign(arr);
        *arr = new_arr;
    }
    let mut col = arr.column_mut(i);
    let n1 = vec.nrows();
    col.slice_mut(ndarray::s![..n1]).iter_mut().zip(vec.iter()).for_each(|(d, s)| *d = *s);
    if let Some(vec2) = vec2 {
        let n2 = vec2.nrows();
        col.slice_mut(ndarray::s![n1..(n1 + n2)])
            .iter_mut()
            .zip(vec2.iter())
            .for_each(|(d, s)| *d = *s);
    }
}   

// --- Main ---

fn main() {
    let code = fs::read_to_string("taylor_green.dsl").unwrap();
    let m_lumped = load_vec("m_lumped");

    let problem = OdeBuilder::<M>::new()
        .rtol(1e-6)
        .atol([1e-8])
        .build_from_diffsl::<CG>(&code)
        .unwrap();


    // Load and factor projection operators (raw faer)
    let g = load("G"); // G  = M⁻¹ Bᵀ  (n_u × n_p)
    let gt = load("GT"); // GT = B M⁻¹   (n_p × n_u)
    let l_mat = load("L"); // L  = B M⁻¹ Bᵀ (n_p × n_p)
    let l_lu = {
        let sym = SymbolicLu::try_new(l_mat.symbolic()).unwrap();
        Lu::try_new_with_symbolic(sym, l_mat.as_ref()).unwrap()
    };

    let n_u = problem.eqn().rhs().nstates();
    let n_p = gt.nrows();

    // Output storage
    let t_final = 5.0;
    let n_save = 201;
    let dt_save = t_final / (n_save as f64 - 1.0);
    let mut sol = ndarray::Array2::<f64>::zeros((n_u + n_p, n_save));
    let mut ts_save = Vec::with_capacity(n_save);

    // Initialise TR-BDF2 solver
    let mut solver = problem.tr_bdf2::<LS>().unwrap();
    save_vec(&mut sol, 0, solver.state().y.inner(), None);
    ts_save.push(0.0);

    // Time-stepping loop
    let mut save_idx = 1;
    let mut next_save = dt_save;
    let start = std::time::Instant::now();
    let mut t = solver.state().t;

    while t < t_final {
        let t0 = solver.state().t;
        solver.step().unwrap();
        t = solver.state().t;
        let h = t - t0;
        let inv_h = 1.0 / h;

        // div = GT * (m_lumped ⊙ u) / h
        let mut phi = &gt * zip!(&m_lumped, solver.state().y.inner()).map(|unzip!(m, u)| m * u * inv_h);

        // Project: u = u - h * G * phi  where  L * phi = div / h
        l_lu.solve_in_place(&mut phi);
        let tmp = &g * &phi;
        zip!(solver.state_mut().y.inner_mut(), &tmp).for_each(|unzip!(u, gphi)| *u -= h * gphi);

        // Save output
        if t > next_save - 1e-12 {
            next_save += dt_save;
            save_vec(&mut sol, save_idx, solver.state().y.inner(), Some(&phi));
            save_idx += 1;
            ts_save.push(t);
        }
    }

    println!(
        "tr_bdf2: {save_idx} saved steps, t = {:.6}, wall = {:.2}s",
        solver.state().t,
        start.elapsed().as_secs_f64()
    );

    let trimmed = sol.slice(ndarray::s![.., ..save_idx]).to_owned();
    write_npy("solution.npy", &trimmed).unwrap();
    write_npy("time.npy", &Array1::from_vec(ts_save[..save_idx].to_vec())).unwrap();
}
