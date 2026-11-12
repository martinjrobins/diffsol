use crate::c_invalid_arg;
use crate::vjp_checkpoint::VjpCheckpointWrapper;

/// Free a VJP checkpoint previously returned by `diffsol_ode_solve_vjp_fwd`.
///
/// # Safety
/// `checkpoint` must be either null or a pointer returned by this library that has not
/// already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn diffsol_vjp_checkpoint_free(checkpoint: *mut VjpCheckpointWrapper) {
    if checkpoint.is_null() {
        c_invalid_arg!("VJP checkpoint is null");
        return;
    }
    unsafe {
        drop(Box::from_raw(checkpoint));
    }
}
