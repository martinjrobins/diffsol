use std::os::raw::c_char;
use std::ptr;

use crate::c_invalid_arg;
use crate::matrix_type::MatrixType;

const MATRIX_TYPE_NALGEBRA: &[u8] = b"nalgebra_dense\0";
const MATRIX_TYPE_FAER: &[u8] = b"faer_dense\0";
const MATRIX_TYPE_FAER_SPARSE: &[u8] = b"faer_sparse\0";

/// The C value of `MatrixType::NalgebraDense`.
pub const DIFFSOL_MATRIX_TYPE_NALGEBRA_DENSE: i32 = 0;
/// The C value of `MatrixType::FaerDense`.
pub const DIFFSOL_MATRIX_TYPE_FAER_DENSE: i32 = 1;
/// The C value of `MatrixType::FaerSparse`.
pub const DIFFSOL_MATRIX_TYPE_FAER_SPARSE: i32 = 2;

pub(crate) fn matrix_type_from_i32(value: i32) -> Option<MatrixType> {
    match value {
        DIFFSOL_MATRIX_TYPE_NALGEBRA_DENSE => Some(MatrixType::NalgebraDense),
        DIFFSOL_MATRIX_TYPE_FAER_DENSE => Some(MatrixType::FaerDense),
        DIFFSOL_MATRIX_TYPE_FAER_SPARSE => Some(MatrixType::FaerSparse),
        _ => None,
    }
}

pub(crate) fn matrix_type_to_i32(value: MatrixType) -> i32 {
    match value {
        MatrixType::NalgebraDense => DIFFSOL_MATRIX_TYPE_NALGEBRA_DENSE,
        MatrixType::FaerDense => DIFFSOL_MATRIX_TYPE_FAER_DENSE,
        MatrixType::FaerSparse => DIFFSOL_MATRIX_TYPE_FAER_SPARSE,
    }
}

/// Return the number of matrix type enum values.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_matrix_type_count() -> usize {
    3
}

/// Return whether a matrix type enum value is valid.
///
/// # Safety
/// This function is safe to call from C. It does not dereference any
/// caller-provided pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_matrix_type_is_valid(value: i32) -> i32 {
    if matrix_type_from_i32(value).is_some() {
        1
    } else {
        c_invalid_arg!("invalid matrix_type");
        0
    }
}

/// Return the name of a matrix type enum value.
///
/// # Safety
/// The returned pointer is borrowed from static storage owned by this library
/// and must not be freed or mutated by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_matrix_type_name(value: i32) -> *const c_char {
    match matrix_type_from_i32(value) {
        Some(MatrixType::NalgebraDense) => MATRIX_TYPE_NALGEBRA.as_ptr() as *const c_char,
        Some(MatrixType::FaerDense) => MATRIX_TYPE_FAER.as_ptr() as *const c_char,
        Some(MatrixType::FaerSparse) => MATRIX_TYPE_FAER_SPARSE.as_ptr() as *const c_char,
        None => {
            c_invalid_arg!("invalid matrix_type");
            ptr::null()
        }
    }
}
