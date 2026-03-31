// Solver type Python enum. This refers to the internal solver mechanism, either
// LU or KLU in diffsol, with default selecting whichever is most appropriate
// given the matrix type.

/// Enumerates the possible linear solver types for diffsol
///
/// :attr default: use the solver's default linear solver choice, typically LU
/// :attr lu: use LU decomposition linear solver (dense or sparse as appropriate)
/// :attr klu: use KLU sparse linear solver
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinearSolverType {
    Default,
    Lu,
    Klu,
}
