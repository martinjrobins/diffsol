#[cfg(feature = "diffsl-llvm")]
mod lorenz;

fn main() {
    #[cfg(feature = "diffsl-llvm")]
    lorenz::lorenz().unwrap();
}
