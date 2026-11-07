use std::os::raw::c_char;
use std::ptr;

use crate::c_invalid_arg;
use crate::jit::JitBackendType;

#[cfg(feature = "diffsl-cranelift")]
const JIT_BACKEND_CRANELIFT: &[u8] = b"cranelift\0";
#[cfg(feature = "diffsl-llvm")]
const JIT_BACKEND_LLVM: &[u8] = b"llvm\0";

/// The C value of `JitBackendType::Cranelift`. The value is the same if the library does not
/// have this backend.
pub const DIFFSOL_JIT_BACKEND_TYPE_CRANELIFT: i32 = 0;
/// The C value of `JitBackendType::Llvm`. The value is the same if the library does not have
/// this backend.
pub const DIFFSOL_JIT_BACKEND_TYPE_LLVM: i32 = 1;

/// The JIT backend with the C value `value`, if the library has that backend.
pub(crate) fn jit_backend_from_i32(value: i32) -> Option<JitBackendType> {
    match value {
        #[cfg(feature = "diffsl-cranelift")]
        DIFFSOL_JIT_BACKEND_TYPE_CRANELIFT => Some(JitBackendType::Cranelift),
        #[cfg(feature = "diffsl-llvm")]
        DIFFSOL_JIT_BACKEND_TYPE_LLVM => Some(JitBackendType::Llvm),
        _ => None,
    }
}

#[allow(dead_code)]
pub(crate) fn jit_backend_to_i32(value: JitBackendType) -> i32 {
    match value {
        #[cfg(feature = "diffsl-cranelift")]
        JitBackendType::Cranelift => DIFFSOL_JIT_BACKEND_TYPE_CRANELIFT,
        #[cfg(feature = "diffsl-llvm")]
        JitBackendType::Llvm => DIFFSOL_JIT_BACKEND_TYPE_LLVM,
    }
}

/// Return the number of JIT backend enum values (2).
///
/// The values are `0` to `count - 1`, also if the library does not have all the backends. Use
/// `diffsol_jit_backend_type_is_valid` to find which backends the library has.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_jit_backend_type_count() -> usize {
    2
}

/// Return whether a JIT backend enum value is valid, and the library has that backend.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_jit_backend_type_is_valid(value: i32) -> i32 {
    if jit_backend_from_i32(value).is_some() {
        1
    } else {
        c_invalid_arg!("invalid jit_backend_type");
        0
    }
}

/// Return the name of a JIT backend enum value.
///
/// # Safety
/// The returned pointer is borrowed from static storage owned by this library
/// and must not be freed or mutated by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_jit_backend_type_name(value: i32) -> *const c_char {
    match jit_backend_from_i32(value) {
        #[cfg(feature = "diffsl-cranelift")]
        Some(JitBackendType::Cranelift) => JIT_BACKEND_CRANELIFT.as_ptr() as *const c_char,
        #[cfg(feature = "diffsl-llvm")]
        Some(JitBackendType::Llvm) => JIT_BACKEND_LLVM.as_ptr() as *const c_char,
        None => {
            c_invalid_arg!("invalid jit_backend_type");
            ptr::null()
        }
    }
}
