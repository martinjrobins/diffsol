use crate::error_c::set_last_error;

pub const DIFFSOL_OK: i32 = 0;
pub const DIFFSOL_ERR: i32 = -1;
pub const DIFFSOL_BAD_ARG: i32 = -2;

pub trait CMapTo<Out> {
    fn c_map_to(self) -> Out;
}

pub trait CMapFrom<In> {
    fn c_map_from(value: In) -> Self;
}

macro_rules! impl_c_map_identity {
    ($t:ty) => {
        impl CMapTo<$t> for $t {
            #[inline]
            fn c_map_to(self) -> $t {
                self
            }
        }

        impl CMapFrom<$t> for $t {
            #[inline]
            fn c_map_from(value: $t) -> Self {
                value
            }
        }
    };
}

impl_c_map_identity!(i32);
impl_c_map_identity!(usize);
impl_c_map_identity!(u64);
impl_c_map_identity!(i64);
impl_c_map_identity!(f64);
impl_c_map_identity!(f32);
impl_c_map_identity!(bool);

impl CMapTo<i32> for bool {
    #[inline]
    fn c_map_to(self) -> i32 {
        if self {
            1
        } else {
            0
        }
    }
}

impl CMapFrom<i32> for bool {
    #[inline]
    fn c_map_from(value: i32) -> Self {
        value != 0
    }
}

#[inline]
pub fn map_get<In, Out>(value: In) -> Out
where
    In: CMapTo<Out>,
{
    value.c_map_to()
}

#[inline]
pub fn map_set<In, Out>(value: In) -> Out
where
    Out: CMapFrom<In>,
{
    Out::c_map_from(value)
}

#[inline]
pub fn invalid_arg_at(msg: &str, file: &'static str, line: u32) -> i32 {
    set_last_error(msg, file, line);
    DIFFSOL_BAD_ARG
}

#[inline]
pub fn error_at(msg: &str, file: &'static str, line: u32) -> i32 {
    set_last_error(msg, file, line);
    DIFFSOL_ERR
}

#[inline]
pub fn null_err_at<T>(ptr: *const T, msg: &str, file: &'static str, line: u32) -> bool {
    if ptr.is_null() {
        set_last_error(msg, file, line);
        true
    } else {
        false
    }
}

#[inline]
pub fn valid_f64_ptr(ptr: *const f64, len: usize) -> bool {
    len == 0 || !ptr.is_null()
}

#[macro_export]
macro_rules! c_invalid_arg {
    ($msg:expr) => {
        $crate::c_api_utils::invalid_arg_at($msg, file!(), line!())
    };
}

#[macro_export]
macro_rules! c_error {
    ($msg:expr) => {
        $crate::c_api_utils::error_at($msg, file!(), line!())
    };
}

#[macro_export]
macro_rules! c_null_err {
    ($ptr:expr, $msg:expr) => {
        $crate::c_api_utils::null_err_at($ptr, $msg, file!(), line!())
    };
}

#[macro_export]
macro_rules! c_getter_simple {
    ($prefix:ident, $opt_ty:ty, $out_ty:ty, $field:ident) => {
        ::paste::paste! {
            #[doc = "Get a solver option value."]
            #[doc = ""]
            #[doc = "# Safety"]
            #[doc = "`options` must be a valid pointer created by this library. `out_value` must"]
            #[doc = "be a valid, writable pointer for a single output value."]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<$prefix _get_ $field>](options: *const $opt_ty, out_value: *mut $out_ty) -> i32 {
                if options.is_null() || out_value.is_null() {
                    return $crate::c_invalid_arg!(concat!(
                        "invalid arguments to ",
                        stringify!([<$prefix _get_ $field>])
                    ));
                }
                let options = unsafe { &*options };
                match options.[<get_ $field>]() {
                    Ok(value) => {
                        let mapped: $out_ty = $crate::c_api_utils::map_get::<_, $out_ty>(value);
                        unsafe {
                            *out_value = mapped;
                        }
                        $crate::c_api_utils::DIFFSOL_OK
                    }
                    Err(err) => $crate::c_error!(&format!("{}", err)),
                }
            }
        }
    };
}

#[macro_export]
macro_rules! c_setter_simple {
    ($prefix:ident, $opt_ty:ty, $in_ty:ty, $field:ident) => {
        ::paste::paste! {
            #[doc = "Set a solver option value."]
            #[doc = ""]
            #[doc = "# Safety"]
            #[doc = "`options` must be a valid mutable pointer created by this library."]
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<$prefix _set_ $field>](options: *mut $opt_ty, value: $in_ty) -> i32 {
                if options.is_null() {
                    return $crate::c_invalid_arg!(concat!(
                        "invalid arguments to ",
                        stringify!([<$prefix _set_ $field>])
                    ));
                }
                let options = unsafe { &mut *options };
                let mapped = $crate::c_api_utils::map_set::<$in_ty, _>(value);
                match options.[<set_ $field>](mapped) {
                    Ok(()) => $crate::c_api_utils::DIFFSOL_OK,
                    Err(err) => $crate::c_error!(&format!("{}", err)),
                }
            }
        }
    };
}
