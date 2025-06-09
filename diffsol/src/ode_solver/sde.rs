use crate::{OdeEquationsStoch, OdeSolverMethod};

pub trait SdeSolverMethod<'a, Eqn>: OdeSolverMethod<'a, Eqn>
where
    Eqn: OdeEquationsStoch + 'a,
{
}
