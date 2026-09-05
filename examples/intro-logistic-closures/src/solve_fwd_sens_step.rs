use diffsol::{Context, OdeEquationsImplicitSens, OdeSolverMethod, Vector, VectorView};

pub fn solve_fwd_sens_step<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquationsImplicitSens<T = f64> + 'a,
{
    let t_o = 5.0;
    while solver.state().t < t_o {
        solver.step().unwrap();
    }
    // the sensitivities hold one parameter per batch lane
    let sens_at_t_o = solver.interpolate_sens(t_o).unwrap();
    let sens_at_internal_step = solver.state().s;
    for p in 0..sens_at_t_o.context().nbatch() {
        println!(
            "sensitivity wrt parameter {p} at t_o: {:?}",
            sens_at_t_o.get_batch(p).into_owned()
        );
        println!(
            "sensitivity wrt parameter {p} at internal step: {:?}",
            sens_at_internal_step.get_batch(p).into_owned()
        );
    }
}
