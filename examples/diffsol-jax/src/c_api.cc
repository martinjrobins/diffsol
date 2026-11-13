// The C functions that the Python package calls with ctypes. Each function that can fail writes
// a message into `err` (`err_len` bytes, NUL-terminated) and returns 0 on failure.
#include <cstdint>
#include <cstring>
#include <string>

#include "registry.h"

namespace {

int Fail(const std::string& message, char* err, size_t err_len) {
  if (err && err_len > 0) {
    std::strncpy(err, message.c_str(), err_len - 1);
    err[err_len - 1] = '\0';
  }
  return 0;
}

}  // namespace

extern "C" {

// Load the PJRT plugin and make the client. Only the first call has an effect.
int diffsol_jax_init(const char* plugin_path, char* err, size_t err_len) {
  std::string message;
  if (!diffsol_jax::InitRuntime(plugin_path, &message)) return Fail(message, err, err_len);
  return 1;
}

// Return the id of a new model (> 0), or 0 on failure.
int64_t diffsol_jax_model_new(size_t nstates, size_t nparams, size_t nout, int32_t matrix_type,
                              int32_t linear_solver, int32_t ode_solver, size_t max_checkpoints,
                              char* err, size_t err_len) {
  diffsol_jax::ModelConfig config;
  config.nstates = nstates;
  config.nparams = nparams;
  config.nout = nout;
  config.matrix_type = matrix_type;
  config.linear_solver = linear_solver;
  config.ode_solver = ode_solver;
  config.max_checkpoints = max_checkpoints > 0 ? max_checkpoints : 1;
  std::string message;
  int64_t id = diffsol_jax::NewModel(config, &message);
  if (id == 0) Fail(message, err, err_len);
  return id;
}

void diffsol_jax_model_free(int64_t model_id) { diffsol_jax::FreeModel(model_id); }

// Compile StableHLO `code` as callback `callback_id` of the model, for `nbatch` lanes and `p`
// lanes on the `P` axis.
int diffsol_jax_model_add_variant(int64_t model_id, size_t nbatch, int32_t callback_id, size_t p,
                                  const char* code, size_t code_len, const char* options,
                                  size_t options_len, char* err, size_t err_len) {
  auto model = diffsol_jax::FindModel(model_id);
  if (!model) return Fail("unknown model", err, err_len);
  std::string message;
  if (!model->AddVariant(nbatch, callback_id, p, std::string(code, code_len),
                         std::string(options, options_len), &message)) {
    return Fail(message, err, err_len);
  }
  return 1;
}

int diffsol_jax_model_has_variant(int64_t model_id, size_t nbatch, int32_t callback_id,
                                  size_t p) {
  auto model = diffsol_jax::FindModel(model_id);
  return model && model->HasVariant(nbatch, callback_id, p) ? 1 : 0;
}

void diffsol_jax_model_clear_checkpoints(int64_t model_id) {
  if (auto model = diffsol_jax::FindModel(model_id)) model->ClearCheckpoints();
}

size_t diffsol_jax_model_num_checkpoints(int64_t model_id) {
  auto model = diffsol_jax::FindModel(model_id);
  return model ? model->NumCheckpoints() : 0;
}

uint64_t diffsol_jax_model_num_recomputes(int64_t model_id) {
  auto model = diffsol_jax::FindModel(model_id);
  return model ? model->NumRecomputes() : 0;
}

// The number of callback calls, of inputs copied because they were not 64-byte aligned, and of
// PJRT views made (view cache misses), for the solver with `nbatch` lanes. With `reset != 0`,
// set them to 0. Return 0 if there is no such solver.
int diffsol_jax_model_stats(int64_t model_id, size_t nbatch, uint64_t* ncalls, uint64_t* nstaged,
                            uint64_t* nviews, int reset) {
  auto model = diffsol_jax::FindModel(model_id);
  return model && model->Stats(nbatch, ncalls, nstaged, nviews, reset != 0) ? 1 : 0;
}

}  // extern "C"
