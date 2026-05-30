use diffsol::{
    FaerSparseMat, LlvmModule,
    OdeBuilder, OdeSolverMethod, Vector, VectorCommon, VectorHost,
};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};
use faer::{unzip, zip, Col};
use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use std::fs;

type Mat = FaerSparseMat<f64>;
type CG = LlvmModule;

fn load(prefix: &str) -> SparseColMat<usize, f64> {
    let d = read_npy::<_, Array1<i64>>(format!("{prefix}_dims.npy")).unwrap();
    let i: Array1<i64> = read_npy::<_, _>(format!("{prefix}_i.npy")).unwrap();
    let j: Array1<i64> = read_npy::<_, _>(format!("{prefix}_j.npy")).unwrap();
    let v: Array1<f64> = read_npy::<_, _>(format!("{prefix}_v.npy")).unwrap();
    SparseColMat::try_new_from_triplets(
        d[0] as usize,
        d[1] as usize,
        &(0..i.len())
            .map(|k| Triplet::new(i[k] as usize, j[k] as usize, v[k]))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn load_vec(name: &str) -> Col<f64> {
    Col::from_iter(read_npy::<_, Array1<f64>>(format!("{name}.npy")).unwrap())
}

pub(crate) fn main() {
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("meta.json").unwrap()).unwrap();
    let nu = meta["nu"].as_f64().unwrap();
    let n_free = meta["n_free"].as_u64().unwrap() as usize;
    let n_u = meta["n_u"].as_u64().unwrap() as usize;
    let n_p = meta["n_p"].as_u64().unwrap() as usize;

    // For explicit solver: absorb M_lumped^{-1} into RHS
    let code = format!(
        "in_i {{ nu = {nu:.16e} }}\n\
         H_ij    {{ (0:{n_free}, 0:{n_free}): read('H.tns') }}\n\
         Fmom_i  {{ (0:{n_free}): read('f_mom.tns') }}\n\
         InvM_i  {{ (0:{n_free}): read('inv_mass.tns') }}\n\
         I_i     {{ (0:{n_free}): read('init.tns') }}\n\
         u_i     {{ y = I_i }}\n\
         dudt_i  {{ (0:{n_free}): dydt = 0 }}\n\
         F1_i    {{ H_ij * y_j }}\n\
         F_i     {{ InvM_i * (F1_i + Fmom_i) }}\n\
         out_i   {{ u_i }}\n"
    );

    let m_lumped = load_vec("m_lumped");
    let f_div: Array1<f64> = read_npy::<_, _>("f_div.npy").unwrap();
    let free_mask: Array1<bool> = read_npy::<_, _>("free_mask.npy").unwrap();
    let bc_vals_full: Array1<f64> = read_npy::<_, _>("bc_vals_full.npy").unwrap();

    let t0 = std::time::Instant::now();
    let problem = OdeBuilder::<Mat>::new()
        .rtol(1e-6)
        .atol([1e-8])
        .build_from_diffsl::<CG>(&code)
        .unwrap();
    println!("DiffSL init: {:.2}s", t0.elapsed().as_secs_f64());

    let t0 = std::time::Instant::now();
    let g = load("G");
    let gt = load("GT");
    let l_mat = load("L");
    let l_lu = {
        let s = SymbolicLu::try_new(l_mat.symbolic()).unwrap();
        Lu::try_new_with_symbolic(s, l_mat.as_ref()).unwrap()
    };
    println!("Load + factor L: {:.2}s", t0.elapsed().as_secs_f64());

    let t_final = 20.0;
    let n_save = 201;
    let dt_save = t_final / (n_save as f64 - 1.0);
    let mut sol = ndarray::Array2::<f64>::zeros((n_u + n_p, n_save));
    let mut ts_save = Vec::with_capacity(n_save);
    let mut solver = problem.tsit45().unwrap();
    {
        let mut fi = 0usize;
        let v: Vec<f64> = (0..n_u)
            .map(|i| {
                if free_mask[i] {
                    let x = solver.state().y.as_slice()[fi];
                    fi += 1;
                    x
                } else {
                    bc_vals_full[i]
                }
            })
            .collect();
        sol.column_mut(0)
            .slice_mut(ndarray::s![..n_u])
            .iter_mut()
            .zip(&v)
            .for_each(|(d, s)| *d = *s);
    }
    ts_save.push(0.0);

    let f_div_col = Col::from_fn(n_p, |i| f_div[i]);
    let mut save_idx = 1;
    let mut next_save = dt_save;
    let start = std::time::Instant::now();
    let mut t = solver.state().t;
    let mut total_step = 0.0f64;
    let mut total_proj = 0.0f64;
    let mut total_save = 0.0f64;
    let mut n_steps = 0usize;

    while t < t_final {
        let ts = std::time::Instant::now();
        let t0 = solver.state().t;
        solver.step().unwrap();
        t = solver.state().t;
        total_step += ts.elapsed().as_secs_f64();
        n_steps += 1;
        let h = t - t0;
        let inv_h = 1.0 / h;

        if n_steps.is_multiple_of(500) {
            println!("  step {}, t = {:.4}", n_steps, t);
        }

        let ts = std::time::Instant::now();
        let mut tmp = Col::zeros(n_free);
        zip!(tmp.as_mut(), &m_lumped, solver.state().y.inner())
            .for_each(|unzip!(t, m, u)| *t = m * u * inv_h);
        let mut phi = &gt * &tmp;
        zip!(phi.as_mut(), &f_div_col).for_each(|unzip!(p, f)| *p += f * inv_h);
        l_lu.solve_in_place(&mut phi);
        let grad = &g * &phi;
        zip!(solver.state_mut().y.inner_mut(), &grad).for_each(|unzip!(u, gphi)| *u -= h * gphi);
        total_proj += ts.elapsed().as_secs_f64();

        if t > next_save - 1e-12 {
            next_save += dt_save;
            let ts = std::time::Instant::now();
            let pv: Vec<f64> = phi.iter().copied().collect();
            let mut fi = 0usize;
            let v: Vec<f64> = (0..n_u)
                .map(|i| {
                    if free_mask[i] {
                        let x = solver.state().y.as_slice()[fi];
                        fi += 1;
                        x
                    } else {
                        bc_vals_full[i]
                    }
                })
                .collect();
            let mut col = sol.column_mut(save_idx);
            col.slice_mut(ndarray::s![..n_u])
                .iter_mut()
                .zip(&v)
                .for_each(|(d, s)| *d = *s);
            col.slice_mut(ndarray::s![n_u..])
                .iter_mut()
                .zip(&pv)
                .for_each(|(d, s)| *d = *s);
            save_idx += 1;
            ts_save.push(t);
            total_save += ts.elapsed().as_secs_f64();
        }
    }

    let wall = start.elapsed().as_secs_f64();
    let pct_step = 100.0 * total_step / wall;
    let pct_proj = 100.0 * total_proj / wall;
    let pct_save = 100.0 * total_save / wall;
    println!("\n=== Profile ({n_steps} steps, t_final={t_final}) ===");
    println!("  ODE step:     {total_step:.3}s  ({pct_step:.1}%)");
    println!("  Projection:   {total_proj:.3}s  ({pct_proj:.1}%)");
    println!("  Save/output:  {total_save:.3}s  ({pct_save:.1}%)");
    println!(
        "  Other:        {:.3}s",
        wall - total_step - total_proj - total_save
    );
    println!("  Total wall:   {:.3}s", wall);
    println!("  Avg ODE step: {:.3}s", total_step / n_steps as f64);
    println!("  Avg proj:     {:.3}s", total_proj / n_steps as f64);
    let stats = solver.get_statistics();
    println!("\nSolver statistics:");
    println!("  Steps:                {}", stats.number_of_steps);
    println!(
        "  Error test failures:  {}",
        stats.number_of_error_test_failures
    );
    println!(
        "  NLS iterations:       {}",
        stats.number_of_nonlinear_solver_iterations
    );
    println!(
        "  NLS failures:         {}",
        stats.number_of_nonlinear_solver_fails
    );
    println!(
        "  Linear solver setups: {}",
        stats.number_of_linear_solver_setups
    );
    write_npy(
        "solution.npy",
        &sol.slice(ndarray::s![.., ..save_idx]).to_owned(),
    )
    .unwrap();
    write_npy("time.npy", &Array1::from_vec(ts_save[..save_idx].to_vec())).unwrap();
}
