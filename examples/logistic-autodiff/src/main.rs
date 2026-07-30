#[cfg(feature = "autodiff")]
mod main_autodiff;

fn main() {
    #[cfg(feature = "autodiff")]
    main_autodiff::main();
}
