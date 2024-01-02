
pub trait OdeSolver<T: Scalar, V: Vector<T>, M: Matrix<T, V>, S: LinearSolver<T, V, M>> {
    fn new(f: Box<dyn Callable<T, V>>, x0: V, t0: T, dt: T, max_iter: usize, tol: T) -> Self;
    fn solve(&mut self) -> Result<V>;
}