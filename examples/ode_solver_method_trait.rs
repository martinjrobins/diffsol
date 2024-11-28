use diffsol::{ NalgebraLU, OdeBuilder, OdeEquationsImplicit, OdeSolverMethod };

fn solve_ode<'a, Eqn>(solver: &mut impl OdeSolverMethod<'a, Eqn>, t: Eqn::T) -> Eqn::V
where
   Eqn: OdeEquationsImplicit + 'a,
{
    while solver.state().t <= t {
        solver.step().unwrap();
    }
    solver.interpolate(t).unwrap()
}

fn main() {
     type M = nalgebra::DMatrix<f64>;
     type LS = NalgebraLU<f64>;

     let problem = OdeBuilder::<M>::new()
       .rtol(1e-6)
       .p([0.1])
       .rhs_implicit(
         // dy/dt = -ay
         |x, p, t, y| {
           y[0] = -p[0] * x[0];
         },
         // Jv = -av
         |x, p, t, v, y| {
           y[0] = -p[0] * v[0];
         },
        )
        .init(
             // y(0) = 1
            |p, t| {
               nalgebra::DVector::from_vec(vec![1.0])
            },
        )
        .build()
        .unwrap();

     let mut solver = problem.bdf::<LS>().unwrap();
     let t = 0.4;
     while solver.state().t <= t {
         solver.step().unwrap();
     }
     let y = solver.interpolate(t);
}