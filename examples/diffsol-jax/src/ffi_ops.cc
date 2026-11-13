// The diffsol-c solve functions as XLA FFI custom calls (CPU).
//
// Layouts (row-major, as JAX gives them; the leading axes `[*B]` are the lanes, `B` is their
// product). A row-major `[B, n]` array is a column-major `[n, B]` array: the diffsol-c layout,
// with no copy.
//
//   diffsol_solve_dense     params [*B, np], t_eval [nt]          -> ys [*B, nt, nout]
//   diffsol_solve_fwd_sens  params [*B, np], t_eval [nt]          -> ys, sens [*B, np, nt, nout]
//   diffsol_solve_vjp_fwd   params [*B, np], t_eval [nt]          -> ys, handle s64[*B]
//   diffsol_solve_vjp_bkwd  handle s64[*B], params [*B, np], t_eval [nt],
//                           dgdu [*B, *D, nt, nout]                -> grad [*B, *D, np]
//
// `nout` is the number of outputs of `out`, or `nstates` if the model has no `out`.
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include "diffsol.h"
#include "registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace diffsol_jax {
namespace {

// The product of the first `n` dimensions.
template <typename Dims>
size_t Product(const Dims& dims, size_t n) {
  size_t p = 1;
  for (size_t i = 0; i < n; i++) p *= static_cast<size_t>(dims[i]);
  return p;
}

// Copy a diffsol tensor into `dst` as a contiguous column-major array (which is the row-major
// array with the axes in the opposite order). Free the tensor.
ffi::Error CopyTensor(Tensor* tensor, double* dst, size_t expected) {
  size_t ndim = diffsol_tensor_ndim(tensor);
  size_t dims[3] = {1, 1, 1}, strides[3] = {0, 0, 0};
  for (size_t i = 0; i < ndim; i++) {
    dims[i] = diffsol_tensor_dim(tensor, i);
    strides[i] = diffsol_tensor_stride(tensor, i);
  }
  if (dims[0] * dims[1] * dims[2] != expected || diffsol_tensor_dtype(tensor) != ScalarType_F64) {
    diffsol_tensor_free(tensor);
    return ffi::Error::Internal("diffsol returned a tensor with an unexpected shape or type");
  }
  const uint8_t* src = diffsol_tensor_ptr(tensor);
  if (strides[0] == sizeof(double) && (dims[1] == 1 || strides[1] == dims[0] * sizeof(double)) &&
      (dims[2] == 1 || strides[2] == dims[0] * dims[1] * sizeof(double))) {
    std::memcpy(dst, src, expected * sizeof(double));
  } else {
    for (size_t c = 0; c < dims[2]; c++)
      for (size_t b = 0; b < dims[1]; b++)
        for (size_t a = 0; a < dims[0]; a++)
          std::memcpy(dst++, src + a * strides[0] + b * strides[1] + c * strides[2],
                      sizeof(double));
  }
  diffsol_tensor_free(tensor);
  return ffi::Error::Success();
}

// A locked solver for the lanes of `params`.
struct Solver {
  std::shared_ptr<Model> model;
  std::unique_lock<std::mutex> lock;
  Instance* instance = nullptr;
  size_t nbatch = 0;
  size_t nparams = 0;
};

ffi::Error Acquire(int64_t model_id, double rtol, double atol, size_t nbatch, Solver* s) {
  s->model = FindModel(model_id);
  if (!s->model) {
    return ffi::Error::InvalidArgument("unknown diffsol model " + std::to_string(model_id));
  }
  std::string err;
  s->instance = s->model->Acquire(nbatch, &s->lock, &err);
  if (s->instance == nullptr) return ffi::Error::Internal(err);
  s->nbatch = nbatch;
  s->nparams = s->model->config().nparams;
  if (diffsol_ode_set_rtol(s->instance->ode, rtol) != DIFFSOL_OK ||
      diffsol_ode_set_atol(s->instance->ode, atol) != DIFFSOL_OK) {
    return ffi::Error::Internal(DiffsolError(s->instance));
  }
  // clear the callback error of an earlier solve
  s->instance->callbacks.TakeError();
  return ffi::Error::Success();
}

// Check `params [*B, np]` and return `B`.
ffi::Error Lanes(const std::shared_ptr<Model>& model, ffi::Buffer<ffi::F64>& params,
                 size_t* nbatch) {
  auto dims = params.dimensions();
  size_t np = model ? model->config().nparams : 0;
  if (dims.size() < 1 || static_cast<size_t>(dims.back()) != np) {
    return ffi::Error::InvalidArgument("params must have the shape [..., nparams]");
  }
  *nbatch = Product(dims, dims.size() - 1);
  return ffi::Error::Success();
}

// Run a solve that returns a solution, and copy its values into `ys`.
template <typename SolveFn>
ffi::Error SolveAndCopyYs(int64_t model_id, double rtol, double atol,
                          ffi::Buffer<ffi::F64>& params, ffi::Buffer<ffi::F64>& t_eval,
                          double* ys, Solver* s, SolutionWrapper** solution, SolveFn solve) {
  size_t nbatch = 0;
  if (auto e = Lanes(FindModel(model_id), params, &nbatch); e.failure()) return e;
  if (auto e = Acquire(model_id, rtol, atol, nbatch, s); e.failure()) return e;
  size_t nt = t_eval.element_count();
  if (solve(s->instance->ode, params.typed_data(), s->nparams, nbatch, t_eval.typed_data(), nt,
            solution) != DIFFSOL_OK) {
    return ffi::Error::Internal(DiffsolError(s->instance));
  }
  Tensor* tensor = nullptr;
  if (diffsol_solution_wrapper_get_ys(*solution, &tensor) != DIFFSOL_OK) {
    return ffi::Error::Internal(DiffsolError(s->instance));
  }
  return CopyTensor(tensor, ys, s->model->nsolution() * nt * nbatch);
}

ffi::Error SolveDenseImpl(ffi::Buffer<ffi::F64> params, ffi::Buffer<ffi::F64> t_eval,
                          ffi::ResultBuffer<ffi::F64> ys, int64_t model_id, double rtol,
                          double atol) {
  Solver s;
  SolutionWrapper* solution = nullptr;
  auto e = SolveAndCopyYs(model_id, rtol, atol, params, t_eval, ys->typed_data(), &s, &solution,
                          diffsol_ode_solve_dense);
  if (solution) diffsol_solution_wrapper_free(solution);
  return e;
}

ffi::Error SolveFwdSensImpl(ffi::Buffer<ffi::F64> params, ffi::Buffer<ffi::F64> t_eval,
                            ffi::ResultBuffer<ffi::F64> ys, ffi::ResultBuffer<ffi::F64> sens,
                            int64_t model_id, double rtol, double atol) {
  Solver s;
  SolutionWrapper* solution = nullptr;
  auto e = SolveAndCopyYs(model_id, rtol, atol, params, t_eval, ys->typed_data(), &s, &solution,
                          diffsol_ode_solve_fwd_sens);
  if (e.success()) {
    Tensor** list = nullptr;
    size_t len = 0;
    if (diffsol_solution_wrapper_get_sens(solution, &list, &len) != DIFFSOL_OK) {
      e = ffi::Error::Internal(DiffsolError(s.instance));
    } else if (len != s.nparams) {
      e = ffi::Error::Internal("diffsol returned the wrong number of sensitivities");
    } else {
      // sens j is a column-major [nout, nt, B] array; the result is row-major [B, np, nt, nout]
      size_t block = s.model->nsolution() * t_eval.element_count();
      auto tmp = std::make_unique<double[]>(block * s.nbatch);
      for (size_t j = 0; j < len && e.success(); j++) {
        e = CopyTensor(list[j], tmp.get(), block * s.nbatch);
        list[j] = nullptr;
        for (size_t b = 0; b < s.nbatch && e.success(); b++) {
          std::memcpy(sens->typed_data() + (b * s.nparams + j) * block, tmp.get() + b * block,
                      block * sizeof(double));
        }
      }
      diffsol_tensor_list_free(list, len);
    }
  }
  if (solution) diffsol_solution_wrapper_free(solution);
  return e;
}

ffi::Error SolveVjpFwdImpl(ffi::Buffer<ffi::F64> params, ffi::Buffer<ffi::F64> t_eval,
                           ffi::ResultBuffer<ffi::F64> ys, ffi::ResultBuffer<ffi::S64> handle,
                           int64_t model_id, double rtol, double atol) {
  Solver s;
  SolutionWrapper* solution = nullptr;
  VjpCheckpointWrapper* checkpoint = nullptr;
  auto solve = [&checkpoint](OdeWrapper* ode, const double* p, size_t np, size_t nbatch,
                             const double* t, size_t nt, SolutionWrapper** out) {
    return diffsol_ode_solve_vjp_fwd(ode, p, np, nbatch, t, nt, out, &checkpoint);
  };
  auto e = SolveAndCopyYs(model_id, rtol, atol, params, t_eval, ys->typed_data(), &s, &solution,
                          solve);
  if (solution) diffsol_solution_wrapper_free(solution);
  if (e.failure()) {
    if (checkpoint) diffsol_vjp_checkpoint_free(checkpoint);
    return e;
  }
  CheckpointKey key;
  key.params.assign(params.typed_data(), params.typed_data() + params.element_count());
  key.t_eval.assign(t_eval.typed_data(), t_eval.typed_data() + t_eval.element_count());
  key.rtol = rtol;
  key.atol = atol;
  int64_t id = s.model->AddCheckpoint(s.instance, checkpoint, std::move(key));
  for (size_t b = 0; b < s.nbatch; b++) handle->typed_data()[b] = id;
  return e;
}

// The backward solve. It uses the checkpoint of `handle` only if the registry still has it, all
// the lanes of `handle` have the same id, and the checkpoint was made from exactly the same
// `params`, `t_eval`, `rtol` and `atol`. If not (for example the registry freed it, or the handle
// comes from a different process), it runs the forward solve again into a temporary
// checkpoint. Thus the gradient never depends on what the registry holds.
ffi::Error SolveVjpBkwdImpl(ffi::Buffer<ffi::S64> handle, ffi::Buffer<ffi::F64> params,
                            ffi::Buffer<ffi::F64> t_eval, ffi::Buffer<ffi::F64> dgdu,
                            ffi::ResultBuffer<ffi::F64> grad, int64_t model_id, double rtol,
                            double atol) {
  auto model = FindModel(model_id);
  if (!model) return ffi::Error::InvalidArgument("unknown diffsol model");
  size_t nbatch = 0;
  if (auto e = Lanes(model, params, &nbatch); e.failure()) return e;
  if (handle.element_count() != nbatch) {
    return ffi::Error::InvalidArgument("handle and params have different lanes");
  }
  auto dims = dgdu.dimensions();
  size_t nout = model->nsolution();
  size_t nt = t_eval.element_count();
  if (dims.size() < 2 || static_cast<size_t>(dims.back()) != nout ||
      static_cast<size_t>(dims[dims.size() - 2]) != nt) {
    return ffi::Error::InvalidArgument("dgdu must have the shape [..., nt, nout]");
  }
  size_t lanes = Product(dims, dims.size() - 2);
  if (lanes % nbatch != 0) {
    return ffi::Error::InvalidArgument("the lanes of dgdu are not a multiple of the handle lanes");
  }

  const int64_t* ids = handle.typed_data();
  bool same_ids = true;
  for (size_t b = 1; b < nbatch; b++) same_ids = same_ids && ids[b] == ids[0];
  auto checkpoint = same_ids ? model->FindCheckpoint(ids[0]) : nullptr;
  if (checkpoint && (checkpoint->instance->callbacks.nbatch != nbatch ||
                     !checkpoint->key.Matches(params.typed_data(), params.element_count(),
                                              t_eval.typed_data(), nt, rtol, atol))) {
    checkpoint = nullptr;
  }

  Solver s;
  VjpCheckpointWrapper* fresh = nullptr;
  const VjpCheckpointWrapper* use = nullptr;
  if (checkpoint) {
    s.model = model;
    s.instance = checkpoint->instance;
    s.lock = std::unique_lock<std::mutex>(s.instance->mutex);
    s.instance->callbacks.TakeError();
    use = checkpoint->checkpoint;
  } else {
    // recompute: the same forward solve as `diffsol_solve_vjp_fwd`
    if (auto e = Acquire(model_id, rtol, atol, nbatch, &s); e.failure()) return e;
    SolutionWrapper* solution = nullptr;
    int rc = diffsol_ode_solve_vjp_fwd(s.instance->ode, params.typed_data(), s.nparams, nbatch,
                                       t_eval.typed_data(), nt, &solution, &fresh);
    if (solution) diffsol_solution_wrapper_free(solution);
    if (rc != DIFFSOL_OK) return ffi::Error::Internal(DiffsolError(s.instance));
    model->CountRecompute();
    use = fresh;
  }
  Tensor* tensor = nullptr;
  int rc = diffsol_ode_solve_vjp_bkwd(s.instance->ode, use, dgdu.typed_data(), nout, nt, lanes,
                                      &tensor);
  if (fresh) diffsol_vjp_checkpoint_free(fresh);
  if (rc != DIFFSOL_OK) return ffi::Error::Internal(DiffsolError(s.instance));
  return CopyTensor(tensor, grad->typed_data(), model->config().nparams * lanes);
}

}  // namespace
}  // namespace diffsol_jax

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveDense, diffsol_jax::SolveDenseImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("model_id")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveFwdSens, diffsol_jax::SolveFwdSensImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("model_id")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveVjpFwd, diffsol_jax::SolveVjpFwdImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::S64>>()
                                  .Attr<int64_t>("model_id")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveVjpBkwd, diffsol_jax::SolveVjpBkwdImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<int64_t>("model_id")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol"));
