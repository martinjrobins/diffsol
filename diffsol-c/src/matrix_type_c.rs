use std::os::raw::c_char;
use std::ptr;

use crate::c_invalid_arg;
use crate::matrix_type::MatrixType;

const MATRIX_TYPE_NALGEBRA: &[u8] = b"nalgebra_dense\0";
const MATRIX_TYPE_FAER: &[u8] = b"faer_dense\0";
const MATRIX_TYPE_FAER_SPARSE: &[u8] = b"faer_sparse\0";

pub(crate) fn matrix_type_from_i32(value: i32) -> Option<MatrixType> {
    match value {
        0 => Some(MatrixType::NalgebraDense),
        1 => Some(MatrixType::FaerDense),
        2 => Some(MatrixType::FaerSparse),
        _ => None,
    }
}

pub(crate) fn matrix_type_to_i32(value: MatrixType) -> i32 {
    match value {
        MatrixType::NalgebraDense => 0,
        MatrixType::FaerDense => 1,
        MatrixType::FaerSparse => 2,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_matrix_type_count() -> usize {
    3
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_matrix_type_is_valid(value: i32) -> i32 {
    if matrix_type_from_i32(value).is_some() {
        1
    } else {
        c_invalid_arg!("invalid matrix_type");
        0
    }
}

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
