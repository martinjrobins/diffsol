#[cfg(feature = "diffsl-llvm")]
mod main_llvm;

fn main() {
    #[cfg(feature = "diffsl-llvm")]
    main_llvm::main();
}
