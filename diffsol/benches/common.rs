use diffsol::matrix::MatrixRef;
use diffsol::vector::VectorRef;
use diffsol::LinearSolver;
use diffsol::{
    DefaultDenseMatrix, DefaultSolver, Matrix, OdeEquationsImplicit, OdeSolverMethod,
    OdeSolverProblem,
};

macro_rules! bench_implicit {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty) => {
        $g.bench_function(stringify!($name), |b| {
            b.iter(|| {
                let (problem, soln) = $problem::<$m>(false);
                let t_evals = soln
                    .solution_points
                    .iter()
                    .map(|sp| sp.t)
                    .collect::<Vec<_>>();
                $crate::common::$solver::<_, $ls<_>>(&problem, &t_evals);
            })
        });
    };
}
pub(crate) use bench_implicit;

macro_rules! bench_implicit_cg {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty, $($N:expr),+ $(,)?) => {
        $(
            $g.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $problem::<$m, $N>();
                    let t_evals = soln
                        .solution_points
                        .iter()
                        .map(|sp| sp.t)
                        .collect::<Vec<_>>();
                    $crate::common::$solver::<_, $ls<_>>(&problem, &t_evals);
                })
            });
        )+
    };
}
pub(crate) use bench_implicit_cg;

macro_rules! bench_implicit_rt {
    ($g:ident, $name:ident, $solver:ident, $ls:ident, $problem:ident, $m:ty, $($N:expr),+ $(,)?) => {
        $(
            $g.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $problem::<$m>(false, $N);
                    let t_evals = soln
                        .solution_points
                        .iter()
                        .map(|sp| sp.t)
                        .collect::<Vec<_>>();
                    $crate::common::$solver::<_, $ls<_>>(&problem, &t_evals);
                })
            });
        )+
    };
}
pub(crate) use bench_implicit_rt;

macro_rules! bench_explicit {
    ($g:ident, $name:ident, $solver:ident, $problem:ident, $m:ty) => {
        $g.bench_function(stringify!($name), |b| {
            b.iter(|| {
                let (problem, soln) = $problem::<$m>(false);
                let t_evals = soln
                    .solution_points
                    .iter()
                    .map(|sp| sp.t)
                    .collect::<Vec<_>>();
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
