// The `OdeCallbacksDesc_f64` function pointers of diffsol-c, with PJRT executables behind them.
//
// Lane layout (from the diffsol-c test `lane_layout_rule`): diffsol calls a callback with views
// of the shape `[n, P, B]` (column-major). The `B` axis is never broadcast. If `P > 1`, the
// inputs `x` and `p` are broadcast on `P` (stride 0) and `v` is not. Thus, as row-major arrays
// with no copy:
//   x: [B, nstates]   p: [B, nparams]   v: [B, P, nv]   y: [B, P, ny]   t, beta: []
// and the only variable of the layout is `P`. Each callback has one executable for each `P`.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "diffsol.h"
#include "pjrt_runtime.h"

namespace diffsol_jax {

// The callbacks that the prototype supports (no `root` and `reset`: diffsol-c does not allow
// them with more than one lane). The values are the same as `CallbackId` in Python.
enum CallbackId : int {
  kRhs = 0,
  kRhsJac,
  kRhsSens,
  kRhsAdjoint,
  kRhsSensAdjoint,
  kInit,
  kInitSens,
  kInitSensAdjoint,
  kMass,
  kMassAdjoint,
  kOut,
  kOutJac,
  kOutSens,
  kOutAdjoint,
  kOutSensAdjoint,
  kNumCallbacks,
};

const char* CallbackName(int id);

// A PJRT view of one argument, kept between calls. diffsol often gives the same pointers again
// (`p` for the whole solve, the scalar slots and the staging buffers always), so the view is
// made only when the pointer or the shape changes. A view is only a pointer and a shape: the
// client reads the memory at execute time (spike), so a view of memory that diffsol has freed
// is harmless while it is not used, and it is used only in a call where the pointer is valid.
struct CachedView {
  const void* ptr = nullptr;
  int64_t dims[kMaxDims] = {};
  int ndim = -1;
  PJRT_Buffer* buffer = nullptr;
};

// The executables of one callback, one for each `P`. There are at most 2 in practice
// (`P == 1`, and `P == nparams` or `P == ndirections`). Each executable has its own views,
// because each `P` gives different shapes.
struct Variants {
  static constexpr int kMax = 4;
  size_t p[kMax] = {};
  PJRT_LoadedExecutable* exe[kMax] = {};
  CachedView views[kMax][kMaxArgs];
  int count = 0;
};

// A 64-byte aligned buffer that grows only when it must. The XLA CPU client accepts a view of
// caller memory only if the pointer is 64-byte aligned (spike 0.2); diffsol vectors are not
// always aligned, so an unaligned input is copied into this buffer first.
struct Staging {
  Staging() = default;
  Staging(const Staging&) = delete;
  Staging& operator=(const Staging&) = delete;
  ~Staging();
  // A buffer of at least `bytes` bytes.
  void* Reserve(size_t bytes);
  void* data = nullptr;
  size_t capacity = 0;
};

// A scalar argument, alone in its 64-byte block, for the same reason.
struct alignas(64) AlignedScalar {
  double value = 0.0;
};

// The `user_data` of the callbacks of one `OdeWrapper` (one model, one `B`).
struct CallbackSet {
  PjrtRuntime* runtime = nullptr;
  size_t nstates = 0;
  size_t nparams = 0;
  size_t nout = 0;
  size_t nbatch = 0;
  std::array<Variants, kNumCallbacks> variants;
  // The scalar arguments `t` and `beta`: PJRT gets a view of them, so they need an address.
  AlignedScalar scalars[2];
  // One staging buffer for each argument position.
  Staging staging[kMaxArgs];
  // Statistics: the number of callback calls, of inputs copied because of alignment, and of
  // PJRT views made (the misses of the view cache).
  uint64_t ncalls = 0;
  uint64_t nstaged = 0;
  uint64_t nviews = 0;
  // The first error of a callback since the last `TakeError`.
  std::string error;

  // Add the executable of callback `id` for `p` lanes. Replace an existing one.
  void Add(int id, size_t p, PJRT_LoadedExecutable* exe);
  bool Has(int id, size_t p) const;
  // Set the function pointers of the callbacks that have at least one executable.
  void Fill(OdeCallbacksDesc_f64* desc);
  std::string TakeError();
  void DestroyAll();
  // Destroy the cached views of executable `i` of callback `id`.
  void DestroyViews(int id, int i);
};

}  // namespace diffsol_jax
