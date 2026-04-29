#[cfg(feature = "diffsl-llvm")]
mod main_llvm;

fn main() {
    #[cfg(feature = "diffsl-llvm")]
    main_llvm::main();

    #[cfg(not(feature = "diffsl-llvm"))]
    eprintln!("Enable one of the diffsl-llvmXX features to run this example.");
}
