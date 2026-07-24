/// get version of the library from Cargo.toml
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Inidicate whether Klu functions are available.
/// This depends on whether the library was built with suitesparse support.
pub fn is_klu_available() -> bool {
    cfg!(feature = "suitesparse")
}

/// Inidicate whether sensitivity analysis is available.
/// Sensitivity analysis is currently limited to Linux and macos, and not supported for Windows.
pub fn is_sens_available() -> bool {
    cfg!(not(target_os = "windows"))
}
