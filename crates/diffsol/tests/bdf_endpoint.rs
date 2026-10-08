use diffsol::matrix::MatrixRef;
use diffsol::{
    Context, DefaultDenseMatrix, FaerLU, FaerMat, LinearSolver, Matrix, NalgebraLU, NalgebraMat,
    OdeBuilder, OdeSolverMethod, Vector, VectorRef, VectorView,
};

fn check_endpoint<M, LS>()
where
    M: Matrix<T = f64>,
    M::V: DefaultDenseMatrix,
    LS: LinearSolver<M>,
    for<'a> &'a M: MatrixRef<M>,
    for<'a> &'a M::V: VectorRef<M::V>,
{
    for nbatch in [1, 3] {
        for direction in [1.0, -1.0] {
            let problem = OdeBuilder::<M>::new()
                .context(M::C::default().clone_with_nbatch(nbatch).unwrap())
                .h0(direction * 1e-3)
                .rtol(1e-6)
                .atol([1e-6])
                .rhs_implicit(
                    |x, _p, _t, out| out[0] = x[0],
                    |_x, _p, _t, v, out| out[0] = v[0],
                )
                .init(|_p, _t, out| out[0] = 1.0, 1)
                .build()
                .unwrap();
            let mut method = problem.bdf::<LS>().unwrap();
            for _ in 0..20 {
                method.step().unwrap();
                let state = method.state();
                // Compare both the endpoint and the nearest interior point:
                // the accepted state must agree with its continuous extension.
                // The interior check also guards against an endpoint-only fix.
                let interior = if direction > 0.0 {
                    state.t.next_down()
                } else {
                    state.t.next_up()
                };
                for time in [state.t, interior] {
                    let sampled = method.interpolate(time).unwrap();
                    for batch in 0..nbatch {
                        let accepted = state.y.get_batch(batch).get_index(0);
                        let interpolated = sampled.get_batch(batch).get_index(0);
                        assert!(
                            (interpolated - accepted).abs() < 1e-12,
                            "at {time}, batch {batch}: state={accepted}, interpolated={interpolated}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn faer_bdf_endpoint_matches_its_continuous_extension() {
    check_endpoint::<FaerMat<f64>, FaerLU<f64>>();
}

#[test]
fn nalgebra_bdf_endpoint_matches_its_continuous_extension() {
    check_endpoint::<NalgebraMat<f64>, NalgebraLU<f64>>();
}
