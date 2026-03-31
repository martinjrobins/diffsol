use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

struct LastError {
    message: CString,
    file: CString,
    line: u32,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<LastError>> = RefCell::new(None);
}

fn cstring_from_str(value: &str) -> CString {
    if value.as_bytes().contains(&0) {
        let mut bytes = value.as_bytes().to_vec();
        for byte in &mut bytes {
            if *byte == 0 {
                *byte = b'?';
            }
        }
        CString::new(bytes).unwrap_or_else(|_| CString::new("error").unwrap())
    } else {
        CString::new(value).unwrap_or_else(|_| CString::new("error").unwrap())
    }
}

pub(crate) fn set_last_error(message: &str, file: &'static str, line: u32) {
    let message = cstring_from_str(message);
    let file = cstring_from_str(file);
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(LastError {
            message,
            file,
            line,
        });
    });
}

pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_error_code() -> i32 {
    LAST_ERROR.with(|slot| if slot.borrow().is_some() { 1 } else { 0 })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|err| err.message.as_ptr())
            .unwrap_or(ptr::null())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|err| err.message.as_ptr())
            .unwrap_or(ptr::null())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_last_error_file() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|err| err.file.as_ptr())
            .unwrap_or(ptr::null())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_last_error_line() -> u32 {
    LAST_ERROR.with(|slot| slot.borrow().as_ref().map(|err| err.line).unwrap_or(0))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_clear_last_error() {
    clear_last_error();
}
