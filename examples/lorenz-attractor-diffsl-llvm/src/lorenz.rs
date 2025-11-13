use diffsol::{LlvmModule, NalgebraLU, NalgebraMat, OdeBuilder, OdeSolverMethod};

pub fn lorenz() -> Result<(), Box<dyn std::error::Error>> {
    let problem = OdeBuilder::<NalgebraMat<f64>>::new().build_from_diffsl::<LlvmModule>(
        "
            a { 14.0 } b { 10.0 } c { 8.0 / 3.0 }
            u_i {
                x = 1.0,
                y = 0.0,
                z = 0.0,
            }
            F_i {
                b * (y - x);
                x * (a - z) - y;
                x * y - c * z;
            }
        ",
    )?;
    let mut solver = problem.bdf::<NalgebraLU<f64>>()?;
    let (_ys, _ts) = solver.solve(0.0)?;
    Ok(())
}
