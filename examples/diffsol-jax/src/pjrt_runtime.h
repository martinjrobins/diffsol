// A small wrapper of the PJRT C API: load a plugin, compile StableHLO, and run an executable on
// views of caller memory.
//
// Only the PJRT C API header is used. There is no link to XLA: the plugin is loaded with
// dlopen at run time.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace diffsol_jax {

// The maximum number of arguments of a callback executable: (v, p, t, beta, y) for `mass`.
constexpr int kMaxArgs = 5;
// The maximum number of axes of an argument.
constexpr int kMaxDims = 3;

// A view of caller memory: a dense row-major f64 array.
struct ArgView {
  const void* ptr;
  int64_t dims[kMaxDims];
  int ndim;
};

// One PJRT client (one process, one platform), and its first device.
class PjrtRuntime {
 public:
  // Load the plugin at `plugin_path` and make a client. Return nullptr and set `err` on a
  // failure. The runtime is never freed (the plugin stays loaded until the process exits).
  static PjrtRuntime* Create(const std::string& plugin_path, std::string* err);

  // Compile `code` (StableHLO: MLIR text, MLIR bytecode or a portable artifact) with the
  // serialized `xla.CompileOptionsProto` in `options`.
  PJRT_LoadedExecutable* Compile(const std::string& code, const std::string& options,
                                 std::string* err);

  void Destroy(PJRT_LoadedExecutable* exe);

  // A PJRT buffer that is a view of `arg` (no copy). On the CPU client, `arg.ptr` must be
  // 64-byte aligned. The client reads the memory when an execution runs, so one view can be an
  // argument of many executions while its memory stays valid. Free it with `DestroyBuffer`.
  PJRT_Buffer* CreateView(const ArgView& arg, std::string* err);
  void DestroyBuffer(PJRT_Buffer* buffer);

  // Run `exe` with the `nargs` views `args` (never donated). Write the one output (a dense
  // row-major f64 array of `out_bytes` bytes) into `out`.
  //
  // There is no allocation of data memory. The PJRT C API makes a small `PJRT_Buffer` object
  // for the output, and `PJRT_Event` objects; these are the only allocations, and they are
  // freed before the return.
  bool Execute(PJRT_LoadedExecutable* exe, PJRT_Buffer* const* args, int nargs, void* out,
               size_t out_bytes, std::string* err);

  const PJRT_Api* api() const { return api_; }

 private:
  PjrtRuntime() = default;
  bool Check(PJRT_Error* error, const char* what, std::string* err);
  void AwaitAndDestroy(PJRT_Event* event, std::string* err, bool* ok);

  const PJRT_Api* api_ = nullptr;
  PJRT_Client* client_ = nullptr;
  PJRT_Device* device_ = nullptr;
};

}  // namespace diffsol_jax
