#include "callbacks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace diffsol_jax {

namespace {

constexpr const char* kNames[kNumCallbacks] = {
    "rhs",  "rhs_jac",   "rhs_sens",         "rhs_adjoint", "rhs_sens_adjoint",
    "init", "init_sens", "init_sens_adjoint", "mass",       "mass_adjoint",
    "out",  "out_jac",   "out_sens",         "out_adjoint", "out_sens_adjoint",
};

// The callback return codes. diffsol fills the output with NaN and reports the callback name;
// the message is in `CallbackSet::error`.
constexpr int32_t kMissingVariant = 1;
constexpr int32_t kBadLayout = 2;
constexpr int32_t kPjrtError = 3;

constexpr size_t kF64 = sizeof(double);

// Record the first error of the solve.
int32_t Fail(CallbackSet* set, int id, int32_t code, const std::string& message) {
  if (set->error.empty()) set->error = std::string("callback '") + kNames[id] + "': " + message;
  return code;
}

// A view of an input that is constant over `P` (`x` or `p`) as a row-major `[B, n]` array.
bool LaneArg(const TensorView* v, ArgView* out) {
  size_t n = v->shape[0], b = v->shape[2];
  if (v->ndim != 3 || v->dtype != ScalarType_F64 || v->strides[0] != kF64) return false;
  if (v->shape[1] > 1 && v->strides[1] != 0) return false;
  if (b > 1 && v->strides[2] != n * kF64) return false;
  *out = ArgView{v->ptr, {static_cast<int64_t>(b), static_cast<int64_t>(n), 0}, 2};
  return true;
}

// A view of `v` or `y` as a row-major `[B, P, n]` array.
bool FullArg(const TensorView* v, ArgView* out) {
  size_t n = v->shape[0], p = v->shape[1], b = v->shape[2];
  if (v->ndim != 3 || v->dtype != ScalarType_F64 || v->strides[0] != kF64) return false;
  if (p > 1 && v->strides[1] != n * kF64) return false;
  if (b > 1 && v->strides[2] != n * p * kF64) return false;
  *out = ArgView{v->ptr,
                 {static_cast<int64_t>(b), static_cast<int64_t>(p), static_cast<int64_t>(n)},
                 3};
  return true;
}

ArgView Scalar(const AlignedScalar& s) { return ArgView{&s.value, {0, 0, 0}, 0}; }

constexpr uintptr_t kAlign = 64;

// If `arg` is not 64-byte aligned, copy it into `staging` and point `arg` at the copy.
bool AlignArg(ArgView* arg, Staging* staging, CallbackSet* set) {
  if ((reinterpret_cast<uintptr_t>(arg->ptr) & (kAlign - 1)) == 0) return true;
  size_t bytes = kF64;
  for (int d = 0; d < arg->ndim; d++) bytes *= static_cast<size_t>(arg->dims[d]);
  void* dst = staging->Reserve(bytes);
  if (dst == nullptr) return false;
  std::memcpy(dst, arg->ptr, bytes);
  arg->ptr = dst;
  set->nstaged++;
  return true;
}

// Find the executable for the `P` of `y`, check the layouts, run it, and write `y`.
template <int N>
int32_t Run(CallbackSet* set, int id, const TensorView* const (&lane)[N], const bool (&full)[N],
            const ArgView* scalars, int scalar_pos, int nscalars, TensorViewMut* y) {
  set->ncalls++;
  const size_t p = y->shape[1];
  Variants& v = set->variants[id];
  int slot = -1;
  for (int i = 0; i < v.count; i++) {
    if (v.p[i] == p) slot = i;
  }
  if (slot < 0) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "no executable for P = %zu, B = %zu; the lowering did not compile this layout",
                  p, static_cast<size_t>(y->shape[2]));
    return Fail(set, id, kMissingVariant, buf);
  }
  // the executable arguments: the views in order, with the scalars at `scalar_pos`
  ArgView args[kMaxArgs];
  int nargs = 0;
  for (int i = 0; i < N; i++) {
    if (i == scalar_pos) {
      for (int s = 0; s < nscalars; s++) args[nargs++] = scalars[s];
    }
    bool ok = full[i] ? FullArg(lane[i], &args[nargs]) : LaneArg(lane[i], &args[nargs]);
    if (!ok) return Fail(set, id, kBadLayout, "an input view has an unexpected layout");
    if (!AlignArg(&args[nargs], &set->staging[nargs], set)) {
      return Fail(set, id, kPjrtError, "cannot allocate a staging buffer");
    }
    nargs++;
  }
  if (scalar_pos >= N) {
    for (int s = 0; s < nscalars; s++) args[nargs++] = scalars[s];
  }
  ArgView out;
  if (!FullArg(y, &out)) return Fail(set, id, kBadLayout, "the output view is not dense");
  size_t out_bytes = y->shape[0] * y->shape[1] * y->shape[2] * kF64;
  // the PJRT views: reuse a cached view if the pointer and the shape are the same
  std::string err;
  PJRT_Buffer* buffers[kMaxArgs];
  for (int i = 0; i < nargs; i++) {
    CachedView& cv = v.views[slot][i];
    const ArgView& a = args[i];
    bool hit = cv.buffer != nullptr && cv.ptr == a.ptr && cv.ndim == a.ndim &&
               std::memcmp(cv.dims, a.dims, sizeof(int64_t) * static_cast<size_t>(a.ndim)) == 0;
    if (!hit) {
      set->runtime->DestroyBuffer(cv.buffer);
      cv = CachedView{};
      PJRT_Buffer* buffer = set->runtime->CreateView(a, &err);
      if (buffer == nullptr) return Fail(set, id, kPjrtError, err);
      cv.ptr = a.ptr;
      std::memcpy(cv.dims, a.dims, sizeof(cv.dims));
      cv.ndim = a.ndim;
      cv.buffer = buffer;
      set->nviews++;
    }
    buffers[i] = cv.buffer;
  }
  if (!set->runtime->Execute(v.exe[slot], buffers, nargs, const_cast<uint8_t*>(y->ptr),
                             out_bytes, &err)) {
    return Fail(set, id, kPjrtError, err);
  }
  return 0;
}

// The trampolines: one for each signature, with the callback id as a template parameter.
// The executable arguments have the order of the callback arguments.

template <int Id>
int32_t CbFn(const TensorView* x, const TensorView* p, double t, TensorViewMut* y, void* ud) {
  auto* set = static_cast<CallbackSet*>(ud);
  set->scalars[0].value = t;
  ArgView s[1] = {Scalar(set->scalars[0])};
  return Run<2>(set, Id, {x, p}, {false, false}, s, 2, 1, y);
}

template <int Id>
int32_t CbMulFn(const TensorView* x, const TensorView* p, double t, const TensorView* v,
                TensorViewMut* y, void* ud) {
  auto* set = static_cast<CallbackSet*>(ud);
  set->scalars[0].value = t;
  ArgView s[1] = {Scalar(set->scalars[0])};
  return Run<3>(set, Id, {x, p, v}, {false, false, true}, s, 2, 1, y);
}

template <int Id>
int32_t CbInitFn(const TensorView* p, double t, TensorViewMut* y, void* ud) {
  auto* set = static_cast<CallbackSet*>(ud);
  set->scalars[0].value = t;
  ArgView s[1] = {Scalar(set->scalars[0])};
  return Run<1>(set, Id, {p}, {false}, s, 1, 1, y);
}

template <int Id>
int32_t CbInitMulFn(const TensorView* p, double t, const TensorView* v, TensorViewMut* y,
                    void* ud) {
  auto* set = static_cast<CallbackSet*>(ud);
  set->scalars[0].value = t;
  ArgView s[1] = {Scalar(set->scalars[0])};
  return Run<2>(set, Id, {p, v}, {false, true}, s, 1, 1, y);
}

// `y` is an input and the output: the executable gets a view of `y`, and writes the result
// into the memory of `y` when it has finished. Thus the input and the output can be the same
// memory.
template <int Id>
int32_t CbMassFn(const TensorView* v, const TensorView* p, double t, double beta,
                 TensorViewMut* y, void* ud) {
  auto* set = static_cast<CallbackSet*>(ud);
  set->scalars[0].value = t;
  set->scalars[1].value = beta;
  ArgView s[2] = {Scalar(set->scalars[0]), Scalar(set->scalars[1])};
  const TensorView* y_in = y;
  return Run<3>(set, Id, {v, p, y_in}, {true, false, true}, s, 2, 2, y);
}

}  // namespace

Staging::~Staging() { std::free(data); }

void* Staging::Reserve(size_t bytes) {
  if (bytes <= capacity) return data;
  size_t rounded = (bytes + kAlign - 1) / kAlign * kAlign;
  void* fresh = std::aligned_alloc(kAlign, rounded);
  if (fresh == nullptr) return nullptr;
  std::free(data);
  data = fresh;
  capacity = rounded;
  return data;
}

const char* CallbackName(int id) {
  return id >= 0 && id < kNumCallbacks ? kNames[id] : "unknown";
}

void CallbackSet::DestroyViews(int id, int i) {
  for (auto& cv : variants[id].views[i]) {
    runtime->DestroyBuffer(cv.buffer);
    cv = CachedView{};
  }
}

void CallbackSet::Add(int id, size_t p, PJRT_LoadedExecutable* exe) {
  Variants& v = variants[id];
  for (int i = 0; i < v.count; i++) {
    if (v.p[i] == p) {
      DestroyViews(id, i);
      runtime->Destroy(v.exe[i]);
      v.exe[i] = exe;
      return;
    }
  }
  if (v.count == Variants::kMax) {
    // keep the first one (P == 1), replace the last one
    DestroyViews(id, v.count - 1);
    runtime->Destroy(v.exe[v.count - 1]);
    v.count--;
  }
  v.p[v.count] = p;
  v.exe[v.count] = exe;
  v.count++;
}

bool CallbackSet::Has(int id, size_t p) const {
  const Variants& v = variants[id];
  for (int i = 0; i < v.count; i++) {
    if (v.p[i] == p) return true;
  }
  return false;
}

void CallbackSet::Fill(OdeCallbacksDesc_f64* d) {
  auto has = [&](int id) { return variants[id].count > 0; };
  d->rhs = has(kRhs) ? CbFn<kRhs> : nullptr;
  d->rhs_jac = has(kRhsJac) ? CbMulFn<kRhsJac> : nullptr;
  d->rhs_sens = has(kRhsSens) ? CbMulFn<kRhsSens> : nullptr;
  d->rhs_adjoint = has(kRhsAdjoint) ? CbMulFn<kRhsAdjoint> : nullptr;
  d->rhs_sens_adjoint = has(kRhsSensAdjoint) ? CbMulFn<kRhsSensAdjoint> : nullptr;
  d->init = has(kInit) ? CbInitFn<kInit> : nullptr;
  d->init_sens = has(kInitSens) ? CbInitMulFn<kInitSens> : nullptr;
  d->init_sens_adjoint = has(kInitSensAdjoint) ? CbInitMulFn<kInitSensAdjoint> : nullptr;
  d->mass = has(kMass) ? CbMassFn<kMass> : nullptr;
  d->mass_adjoint = has(kMassAdjoint) ? CbMassFn<kMassAdjoint> : nullptr;
  d->out = has(kOut) ? CbFn<kOut> : nullptr;
  d->out_jac = has(kOutJac) ? CbMulFn<kOutJac> : nullptr;
  d->out_sens = has(kOutSens) ? CbMulFn<kOutSens> : nullptr;
  d->out_adjoint = has(kOutAdjoint) ? CbMulFn<kOutAdjoint> : nullptr;
  d->out_sens_adjoint = has(kOutSensAdjoint) ? CbMulFn<kOutSensAdjoint> : nullptr;
  d->user_data = this;
  d->free_user_data = nullptr;
}

std::string CallbackSet::TakeError() {
  std::string e;
  e.swap(error);
  return e;
}

void CallbackSet::DestroyAll() {
  for (int id = 0; id < kNumCallbacks; id++) {
    Variants& v = variants[id];
    for (int i = 0; i < v.count; i++) {
      DestroyViews(id, i);
      runtime->Destroy(v.exe[i]);
    }
    v.count = 0;
  }
}

}  // namespace diffsol_jax
