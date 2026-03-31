#[cfg(feature = "diffsl-cranelift")]
pub(crate) type JitBackend = diffsol::CraneliftJitModule;

#[cfg(feature = "diffsl-llvm")]
pub(crate) type JitBackend = diffsol::LlvmModule;
