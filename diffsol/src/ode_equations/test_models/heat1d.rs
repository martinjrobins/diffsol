#[cfg(feature = "diffsl")]
use crate::{ode_solver::problem::OdeSolverSolution, Matrix, MatrixHost, OdeSolverProblem, Vector};

#[cfg(feature = "diffsl")]
#[allow(clippy::type_complexity)]
pub fn heat1d_diffsl_problem<
    M: MatrixHost<T = f64>,
    CG: crate::CodegenModuleJit + crate::CodegenModuleCompile,
    const MGRID: usize,
>() -> (
    OdeSolverProblem<impl crate::OdeEquationsImplicit<M = M, V = M::V, T = M::T, C = M::C>>,
    OdeSolverSolution<M::V>,
) {
    use crate::OdeBuilder;

    let mgridp1 = MGRID + 1;
    let h = 1.0 / (MGRID + 2) as f64;
    let y0 = (0..mgridp1)
        .map(|i| {
            let x = (i + 1) as f64 * h;
            if x < 0.5 {
                2.0 * x
            } else {
                2.0 * (1.0 - x)
            }
        })
        .collect::<Vec<_>>();
    let y0_str = y0
        .iter()
        .enumerate()
        .map(|(i, v)| format!("({i}): {v}"))
        .collect::<Vec<_>>()
        .join(", ");
    let code = format!(
        "
    D {{ 1.0 }}
    h {{ {h} }}
    A_ij {{
        (0..{MGRID}, 1..{mgridp1}): 1.0,
        (0..{mgridp1}, 0..{mgridp1}): -2.0,
        (1..{mgridp1}, 0..{MGRID}): 1.0,
    }}
    u_i {{
        {y0_str}
    }}
    heat_i {{ A_ij * u_j }}
    F_i {{
        D * heat_i / (h * h)
    }}
    out_i {{ u_i }}
    "
    );
    let problem = OdeBuilder::<M>::new()
        .rtol(1e-6)
        .atol([1e-6])
        .build_from_diffsl::<CG>(code.as_str())
        .unwrap();
    let soln = soln::<M>(problem.context().clone(), MGRID, h);
    (problem, soln)
}

#[cfg(feature = "diffsl")]
fn soln<M: Matrix<T = f64>>(ctx: M::C, mgrid: usize, h: f64) -> OdeSolverSolution<M::V> {
    // we'll put rather loose tolerances here, since the initial conditions have a discontinuity
    let mut soln = OdeSolverSolution {
        solution_points: Vec::new(),
        sens_solution_points: None,
        rtol: 1e-4,
        atol: M::V::from_element(mgrid + 1, 1e-4, ctx.clone()),
        negative_time: false,
    };
    let times = (0..5).map(|i| i as f64 * 0.01 + 0.5).collect::<Vec<_>>();
    let data: Vec<_> = times
        .iter()
        .map(|&t| {
            const PI: f64 = std::f64::consts::PI;
            let mut ret = vec![0.0; mgrid + 1];
            for (i, v) in ret.iter_mut().enumerate() {
                let x = (i + 1) as f64 * h;
                *v = 0.0;
                for n in 1..100 {
                    let two_n_minus_1: f64 = f64::from(2 * n - 1);
                    *v += (two_n_minus_1 * PI * x).sin()
                        * (-two_n_minus_1.powi(2) * PI.powi(2) * t).exp()
                        / two_n_minus_1.powi(2);
                }
                *v *= 8.0 / PI.powi(2);
            }
            M::V::from_vec(ret, ctx.clone())
        })
        .collect();

    for (values, time) in data.into_iter().zip(times.into_iter()) {
        soln.push(values, time);
    }
    soln
}
