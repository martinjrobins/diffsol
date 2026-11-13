"""Load `libdiffsol_jax`, start the PJRT runtime, and register the XLA FFI targets."""

import ctypes
import os
import sys
from pathlib import Path

import jax

_ERR_LEN = 4096
_lib = None

FFI_TARGETS = {
    "diffsol_solve_dense": "DiffsolSolveDense",
    "diffsol_solve_fwd_sens": "DiffsolSolveFwdSens",
    "diffsol_solve_vjp_fwd": "DiffsolSolveVjpFwd",
    "diffsol_solve_vjp_bkwd": "DiffsolSolveVjpBkwd",
}


class DiffsolJaxError(RuntimeError):
    pass


def _default_lib_path() -> Path:
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    return Path(__file__).resolve().parents[2] / "build" / f"libdiffsol_jax{ext}"


def lib() -> ctypes.CDLL:
    """The library, loaded and set up on the first call.

    Environment variables:
    - `DIFFSOL_JAX_LIB`: the path of `libdiffsol_jax` (default: `examples/diffsol-jax/build`).
    - `DIFFSOL_JAX_PJRT_PLUGIN`: the path of the CPU PJRT plugin (`pjrt_c_api_cpu_plugin.so`).
    """
    global _lib
    if _lib is not None:
        return _lib
    path = os.environ.get("DIFFSOL_JAX_LIB", str(_default_lib_path()))
    plugin = os.environ.get("DIFFSOL_JAX_PJRT_PLUGIN")
    if not plugin:
        raise DiffsolJaxError("set DIFFSOL_JAX_PJRT_PLUGIN to the path of the CPU PJRT plugin")
    so = ctypes.CDLL(path)

    c = ctypes
    so.diffsol_jax_init.argtypes = [c.c_char_p, c.c_char_p, c.c_size_t]
    so.diffsol_jax_init.restype = c.c_int
    so.diffsol_jax_model_new.argtypes = [c.c_size_t] * 3 + [c.c_int32] * 3 + [
        c.c_size_t,
        c.c_char_p,
        c.c_size_t,
    ]
    so.diffsol_jax_model_new.restype = c.c_int64
    so.diffsol_jax_model_free.argtypes = [c.c_int64]
    so.diffsol_jax_model_free.restype = None
    so.diffsol_jax_model_add_variant.argtypes = [
        c.c_int64,
        c.c_size_t,
        c.c_int32,
        c.c_size_t,
        c.c_char_p,
        c.c_size_t,
        c.c_char_p,
        c.c_size_t,
        c.c_char_p,
        c.c_size_t,
    ]
    so.diffsol_jax_model_add_variant.restype = c.c_int
    so.diffsol_jax_model_has_variant.argtypes = [c.c_int64, c.c_size_t, c.c_int32, c.c_size_t]
    so.diffsol_jax_model_has_variant.restype = c.c_int
    so.diffsol_jax_model_clear_checkpoints.argtypes = [c.c_int64]
    so.diffsol_jax_model_clear_checkpoints.restype = None
    so.diffsol_jax_model_num_checkpoints.argtypes = [c.c_int64]
    so.diffsol_jax_model_num_checkpoints.restype = c.c_size_t
    so.diffsol_jax_model_num_recomputes.argtypes = [c.c_int64]
    so.diffsol_jax_model_num_recomputes.restype = c.c_uint64
    so.diffsol_jax_model_stats.argtypes = [
        c.c_int64,
        c.c_size_t,
        c.POINTER(c.c_uint64),
        c.POINTER(c.c_uint64),
        c.POINTER(c.c_uint64),
        c.c_int,
    ]
    so.diffsol_jax_model_stats.restype = c.c_int

    err = ctypes.create_string_buffer(_ERR_LEN)
    if not so.diffsol_jax_init(plugin.encode(), err, _ERR_LEN):
        raise DiffsolJaxError(err.value.decode())
    for target, symbol in FFI_TARGETS.items():
        jax.ffi.register_ffi_target(
            target, jax.ffi.pycapsule(getattr(so, symbol)), platform="cpu"
        )
    _lib = so
    return so


def call(fn, *args):
    """Call a library function whose last two arguments are `err, err_len`."""
    err = ctypes.create_string_buffer(_ERR_LEN)
    result = fn(*args, err, _ERR_LEN)
    if not result:
        raise DiffsolJaxError(err.value.decode())
    return result
