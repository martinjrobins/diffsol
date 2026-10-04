#[cfg(feature = "diffsl-llvm")]
mod lorenz;
mod lorenz_closure;

fn main() {
    lorenz_closure::lorenz().unwrap();
    #[cfg(feature = "diffsl-llvm")]
    lorenz::lorenz().unwrap();
}
