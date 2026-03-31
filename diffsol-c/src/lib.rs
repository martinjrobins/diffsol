// Configure dlmalloc as the global allocator for WASM targets
// dlmalloc is a maintained alternative to the unmaintained wee_alloc
#[cfg(target_arch = "wasm32")]
use dlmalloc::GlobalDlmalloc;

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOCATOR: GlobalDlmalloc = GlobalDlmalloc;

#[cfg(not(any(
    feature = "external",
    feature = "diffsl-cranelift",
    feature = "diffsl-llvm"
)))]
compile_error!(
    "diffsol-c requires exactly one backend mode: enable `external-*`, `diffsl-cranelift`, or one `diffsl-llvm*` feature."
);

#[cfg(all(
    feature = "external",
    any(feature = "diffsl-cranelift", feature = "diffsl-llvm")
))]
compile_error!("diffsol-c cannot be built with both external and JIT backends enabled.");

#[cfg(all(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
compile_error!("diffsol-c cannot enable both `diffsl-cranelift` and `diffsl-llvm*` backends.");

#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm16"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm17"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm18"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm19"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm15", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm17"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm18"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm19"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm16", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm17", feature = "diffsl-llvm18"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm17", feature = "diffsl-llvm19"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm17", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm17", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm18", feature = "diffsl-llvm19"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm18", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm18", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm19", feature = "diffsl-llvm20"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm19", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");
#[cfg(all(feature = "diffsl-llvm20", feature = "diffsl-llvm21"))]
compile_error!("diffsol-c cannot enable more than one `diffsl-llvm*` feature.");

pub mod c_api_utils;
pub mod error;
#[path = "error_c.rs"]
pub mod error_c;
pub mod host_array;
#[path = "host_array_c.rs"]
pub mod host_array_c;
pub mod initial_condition_options;
#[path = "initial_condition_options_c.rs"]
pub mod initial_condition_options_c;
pub mod jit;
pub mod linear_solver_type;
#[path = "linear_solver_type_c.rs"]
pub mod linear_solver_type_c;
pub mod matrix_type;
#[path = "matrix_type_c.rs"]
pub mod matrix_type_c;
pub mod ode;
#[path = "ode_c.rs"]
pub mod ode_c;
pub mod ode_options;
#[path = "ode_options_c.rs"]
pub mod ode_options_c;
pub mod ode_solver_type;
#[path = "ode_solver_type_c.rs"]
pub mod ode_solver_type_c;
pub mod scalar_type;
#[path = "scalar_type_c.rs"]
pub mod scalar_type_c;
pub mod solution;
pub mod solution_wrapper;
#[path = "solution_wrapper_c.rs"]
pub mod solution_wrapper_c;
pub mod solve;
pub mod solve_macros;
#[path = "string_c.rs"]
pub mod string_c;
pub mod utils;
pub mod valid_linear_solver;

#[cfg(test)]
mod test_support;

// dlmalloc already exports malloc/free, but we need to make sure they're available
// The global allocator handles allocation, dlmalloc will provide the exports

/// Core library functionality for diffsol-js
/// This will be compiled to WASM and exposed via raw bindings
pub mod solver {
    /// Example solver function - placeholder for future implementation
    pub fn initialize() -> i32 {
        42
    }
}

#[cfg(test)]
mod solver_tests {
    use super::solver;

    #[test]
    fn verify_initialization() {
        let output = solver::initialize();
        assert_eq!(output, 42);
    }
}
