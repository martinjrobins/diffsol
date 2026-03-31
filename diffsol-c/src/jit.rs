// Compile time JitModule type based on compilation settings.
// External module is used when no JIT backend is enabled.
pub type JitModule<T> = diffsl::ExternalModule<T>;
