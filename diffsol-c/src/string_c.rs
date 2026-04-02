use std::alloc::{Layout, alloc, dealloc};
use std::ffi::c_char;

/// Allocate memory for a string of given size (including null terminator)
/// Returns a pointer to writable memory that TypeScript can write to
///
/// # Safety
/// The returned pointer must be released with `diffsol_free_string` using the
/// same `size`. The caller must not read or write beyond the allocated buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_alloc_string(size: usize) -> *mut c_char {
    if size == 0 {
        return std::ptr::null_mut();
    }

    // Create a Vec with the requested capacity and length
    let mut buffer = vec![0u8; size];
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer); // Prevent the Vec from being dropped

    ptr as *mut c_char
}

/// Allocate aligned memory for arbitrary data
///
/// # Safety
/// The returned pointer must be released with `diffsol_free` using the same
/// `size` and `align`. `align` must be a valid alignment value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_alloc(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let align = if align == 0 { 1 } else { align };
    let layout = match Layout::from_size_align(size, align) {
        Ok(layout) => layout,
        Err(_) => return std::ptr::null_mut(),
    };
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    ptr
}

/// Free aligned memory allocated by diffsol_alloc
///
/// # Safety
/// `ptr` must be either null or a pointer returned by `diffsol_alloc` with the
/// same `size` and `align` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_free(ptr: *mut u8, size: usize, align: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    let align = if align == 0 { 1 } else { align };
    let layout = match Layout::from_size_align(size, align) {
        Ok(layout) => layout,
        Err(_) => return,
    };
    unsafe { dealloc(ptr, layout) };
}

/// Free memory allocated by diffsol_alloc_string
///
/// # Safety
/// `ptr` must be either null or a pointer returned by `diffsol_alloc_string`
/// with the same `size`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_free_string(ptr: *mut c_char, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }

    // Reconstruct the Vec from the raw pointer and drop it
    let _ = unsafe { Vec::from_raw_parts(ptr as *mut u8, size, size) };
    // Vec will be dropped here, freeing the memory
}
