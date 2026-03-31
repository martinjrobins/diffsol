#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitBackendType {
    #[cfg(feature = "diffsl-cranelift")]
    Cranelift,
    #[cfg(feature = "diffsl-llvm")]
    Llvm,
}

#[allow(dead_code)]
pub(crate) fn default_enabled_jit_backend() -> Option<JitBackendType> {
    #[cfg(all(feature = "diffsl-cranelift", not(feature = "diffsl-llvm")))]
    let backend = Some(JitBackendType::Cranelift);
    #[cfg(all(feature = "diffsl-llvm", not(feature = "diffsl-cranelift")))]
    let backend = Some(JitBackendType::Llvm);
    #[cfg(any(
        all(feature = "diffsl-cranelift", feature = "diffsl-llvm"),
        not(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))
    ))]
    let backend = None;

    backend
}
