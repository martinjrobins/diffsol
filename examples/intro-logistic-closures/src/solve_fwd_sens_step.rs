use diffsol::{OdeEquationsImplicitSens, OdeSolverMethod};

pub fn solve_fwd_sens_step<'a, Solver, Eqn>(solver: &mut Solver)
where
    Solver: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquationsImplicitSens<T = f64> + 'a,
{
    let t_o = 5.0;
    while solver.state().t < t_o {
        solver.step().unwrap();
    }
    let sens_at_t_o = solver.interpolate_sens(t_o).unwrap();
    let sens_at_internal_step = &solver.state().s;
    println!("sensitivity at t_o: {sens_at_t_o:?}");
    println!("sensitivity at internal step: {sens_at_internal_step:?}");
}
