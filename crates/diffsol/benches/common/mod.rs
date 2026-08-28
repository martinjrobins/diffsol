use diffsol::matrix::MatrixRef;
use diffsol::vector::VectorRef;
use diffsol::LinearSolver;
use diffsol::{
    DefaultDenseMatrix, DefaultSolver, Matrix, OdeEquationsImplicit, OdeSolverMethod,
    OdeSolverProblem,
};

/// Builds the problem and its evaluation times once, outside the timed loop.
///
/// The solvers take the problem by reference and construct their own state, so one problem
/// serves every iteration.  Building it inside `iter` instead makes the measurement mostly
/// allocation, which is the most machine-sensitive thing in the benchmark -- the CI runs
/// compare across a heterogeneous runner fleet and cannot afford the extra variance.
macro_rules! setup_problem {
    ($problem:expr) => {{
        let (problem, soln) = $problem;
        let t_evals = soln
            .solution_points
            .iter()
            .map(|sp| sp.t)
            .collect::<Vec<_>>();
        (problem, t_evals)
    }};
}
pub(crate) use setup_problem;

macro_rules! bench_implicit {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty) => {
        let (problem, t_evals) = $crate::common::setup_problem!($problem::<$m>(false));
        $g.bench_function(stringify!($name), |b| {
            b.iter(|| {
                $crate::common::$solver::<_, $ls<_>>(&problem, &t_evals);
            })
        });
    };
}
pub(crate) use bench_implicit;

macro_rules! bench_implicit_cg {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty, $($N:expr),+ $(,)?) => {
        $(
            {
                let (problem, t_evals) = $crate::common::setup_problem!($problem::<$m, $N>());
                $g.bench_function(concat!(stringify!($name), "_", $N), |b| {
                    b.iter(|| {
                        $crate::common::$solver::<_, $ls<_>>(&problem, &t_evals);
                    })
                });
            }
        )+
    };
}
pub(crate) use bench_implicit_cg;

macro_rules! bench_implicit_rt {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty, $($N:expr),+ $(,)?) => {
        $(
            {
                let (problem, t_evals) = $crate::common::setup_problem!($problem::<$m>(false, $N));
                $g.bench_function(concat!(stringify!($name), "_", $N), |b| {
                    b.iter(|| {
                        $crate::common::$solver::<_, $ls<_>>(&problem, &t_evals);
                    })
                });
            }
        )+
    };
}
pub(crate) use bench_implicit_rt;

macro_rules! bench_explicit {
    ($g:ident, $name:ident, $solver:ident, $problem:ident, $m:ty) => {
        let (problem, t_evals) = $crate::common::setup_problem!($problem::<$m>(false));
        $g.bench_function(stringify!($name), |b| {
            b.iter(|| {
                $crate::common::$solver::<_>(&problem, &t_evals);
            })
        });
    };
}
pub(crate) use bench_explicit;

pub(crate) fn bdf<Eqn, LS>(problem: &OdeSolverProblem<Eqn>, t_evals: &[Eqn::T])
where
    Eqn: OdeEquationsImplicit,
    Eqn::M: Matrix + DefaultSolver,
    Eqn::V: DefaultDenseMatrix,
    LS: LinearSolver<Eqn::M>,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    let mut s = problem.bdf::<LS>().unwrap();
    let _y = s.solve_dense(t_evals);
}

pub(crate) fn esdirk34<Eqn, LS>(problem: &OdeSolverProblem<Eqn>, t_evals: &[Eqn::T])
where
    Eqn: OdeEquationsImplicit,
    Eqn::M: Matrix + DefaultSolver,
    Eqn::V: DefaultDenseMatrix,
    LS: LinearSolver<Eqn::M>,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    let mut s = problem.esdirk34::<LS>().unwrap();
    let _y = s.solve_dense(t_evals);
}

pub(crate) fn tr_bdf2<Eqn, LS>(problem: &OdeSolverProblem<Eqn>, t_evals: &[Eqn::T])
where
    Eqn: OdeEquationsImplicit,
    Eqn::M: Matrix + DefaultSolver,
    Eqn::V: DefaultDenseMatrix,
    LS: LinearSolver<Eqn::M>,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    let mut s = problem.tr_bdf2::<LS>().unwrap();
    let _y = s.solve_dense(t_evals);
}

pub(crate) fn tsit45<Eqn>(problem: &OdeSolverProblem<Eqn>, t_evals: &[Eqn::T])
where
    Eqn: OdeEquationsImplicit,
    Eqn::M: Matrix + DefaultSolver,
    Eqn::V: DefaultDenseMatrix,
    for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
    for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
{
    let mut s = problem.tsit45().unwrap();
    let _y = s.solve_dense(t_evals);
}
