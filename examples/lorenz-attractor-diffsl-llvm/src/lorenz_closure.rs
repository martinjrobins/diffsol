use diffsol::{NalgebraMat, OdeBuilder, OdeSolverMethod};

pub fn lorenz() -> Result<(), Box<dyn std::error::Error>> {
    let problem = OdeBuilder::<NalgebraMat<f64>>::new()
        .p([14.0, 10.0, 8.0 / 3.0])
        .rhs(|x, p, _t, y| {
            y[0] = p[1] * (x[1] - x[0]);
            y[1] = x[0] * (p[0] - x[2]) - x[1];
            y[2] = x[0] * x[1] - p[2] * x[2];
        })
        .init(
            |_p, _t, y| {
                y[0] = 1.0;
                y[1] = 0.0;
                y[2] = 0.0;
            },
            3,
        )
        .build()?;
    let mut solver = problem.tsit45()?;
    let (_ys, _ts, _stop_reason) = solver.solve(10.0)?;
    Ok(())
}
