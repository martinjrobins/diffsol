// Models, the diffsol `OdeWrapper` of each model and lane count, and VJP checkpoint handles.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "callbacks.h"
#include "diffsol.h"
#include "pjrt_runtime.h"

namespace diffsol_jax {

struct ModelConfig {
  size_t nstates = 0;
  size_t nparams = 0;
  // 0 if the model has no `out` callback (then the solution is the state)
  size_t nout = 0;
  int32_t matrix_type = 0;
  int32_t linear_solver = 0;
  int32_t ode_solver = 0;
  size_t max_checkpoints = 8;
};

// The solver of one model for `nbatch` lanes. One solve at a time: diffsol does not allow two
// calls of the callbacks at the same time.
struct Instance {
  std::mutex mutex;
  CallbackSet callbacks;
  OdeWrapper* ode = nullptr;  // made at the first solve
  size_t ncallbacks_at_construction = 0;
};

// The inputs of the forward solve that made a checkpoint. A backward solve uses a checkpoint only
// if its own inputs are exactly the same (bit for bit); if not, it runs the forward solve again.
// Thus a wrong or stale handle costs time, but cannot give a wrong gradient: the registry is
// only a cache.
struct CheckpointKey {
  std::vector<double> params;  // [B, nparams], row-major
  std::vector<double> t_eval;
  double rtol = 0.0;
  double atol = 0.0;
  // TODO(GPU): the device ordinal, when there is more than one device.

  bool Matches(const double* p, size_t np, const double* t, size_t nt, double rt,
               double at) const;
};

// A VJP checkpoint of one solve. It is freed when the last user drops it: an eviction from the
// registry does not free a checkpoint that a backward solve still uses.
struct Checkpoint {
  Checkpoint(int64_t id, Instance* instance, VjpCheckpointWrapper* checkpoint, CheckpointKey key)
      : id(id), instance(instance), checkpoint(checkpoint), key(std::move(key)) {}
  ~Checkpoint() { diffsol_vjp_checkpoint_free(checkpoint); }
  Checkpoint(const Checkpoint&) = delete;
  Checkpoint& operator=(const Checkpoint&) = delete;
  int64_t id;
  Instance* instance;
  VjpCheckpointWrapper* checkpoint;
  CheckpointKey key;
};

class Model {
 public:
  Model(int64_t id, PjrtRuntime* runtime, ModelConfig config)
      : id_(id), runtime_(runtime), config_(config) {}
  ~Model();

  int64_t id() const { return id_; }
  const ModelConfig& config() const { return config_; }
  // The number of values of the solution at each time.
  size_t nsolution() const { return config_.nout > 0 ? config_.nout : config_.nstates; }

  // Compile and add the executable of callback `id` for `nbatch` lanes and `p` lanes on `P`.
  bool AddVariant(size_t nbatch, int id, size_t p, const std::string& code,
                  const std::string& options, std::string* err);
  bool HasVariant(size_t nbatch, int id, size_t p);
  // The callback statistics of the instance for `nbatch` lanes; with `reset`, set them to 0.
  bool Stats(size_t nbatch, uint64_t* ncalls, uint64_t* nstaged, uint64_t* nviews, bool reset);

  // The instance for `nbatch` lanes, locked, with its `OdeWrapper` made. Return nullptr and set
  // `err` if there is no instance or the `OdeWrapper` cannot be made.
  Instance* Acquire(size_t nbatch, std::unique_lock<std::mutex>* lock, std::string* err);

  // Keep a checkpoint and return its handle. Free the oldest if there are too many: then a
  // backward solve with that handle runs the forward solve again.
  int64_t AddCheckpoint(Instance* instance, VjpCheckpointWrapper* checkpoint, CheckpointKey key);
  // The checkpoint of `handle`, or nullptr.
  std::shared_ptr<Checkpoint> FindCheckpoint(int64_t handle);
  void ClearCheckpoints();
  size_t NumCheckpoints();
  // The number of backward solves that ran the forward solve again (no checkpoint).
  void CountRecompute() { nrecomputes_++; }
  uint64_t NumRecomputes() const { return nrecomputes_; }

 private:
  int64_t id_;
  PjrtRuntime* runtime_;
  ModelConfig config_;
  std::mutex mutex_;  // for `instances_` and `checkpoints_`
  std::map<size_t, std::unique_ptr<Instance>> instances_;
  std::deque<std::shared_ptr<Checkpoint>> checkpoints_;
  std::atomic<uint64_t> nrecomputes_{0};
};

// The process-wide registry.
PjrtRuntime* GetRuntime();
bool InitRuntime(const std::string& plugin_path, std::string* err);
int64_t NewModel(const ModelConfig& config, std::string* err);
std::shared_ptr<Model> FindModel(int64_t id);
void FreeModel(int64_t id);

// The last diffsol-c error message, and the callback error, as one message.
std::string DiffsolError(Instance* instance);

}  // namespace diffsol_jax
