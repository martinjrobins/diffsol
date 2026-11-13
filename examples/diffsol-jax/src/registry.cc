#include "registry.h"

#include <atomic>
#include <cstring>
#include <random>
#include <unordered_map>

namespace diffsol_jax {

namespace {

std::mutex g_mutex;
PjrtRuntime* g_runtime = nullptr;
std::unordered_map<int64_t, std::shared_ptr<Model>> g_models;
std::atomic<int64_t> g_next_model{1};
std::atomic<uint64_t> g_next_checkpoint{1};

// A random value for each process, so that two processes do not make the same handles.
uint64_t ProcessSalt() {
  static const uint64_t salt = [] {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }();
  return salt;
}

// splitmix64: a bijection, so different counters in one process give different handles.
uint64_t Mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

}  // namespace

Model::~Model() {
  ClearCheckpoints();
  for (auto& [nbatch, instance] : instances_) {
    if (instance->ode) diffsol_ode_free(instance->ode);
    instance->callbacks.DestroyAll();
  }
}

bool Model::AddVariant(size_t nbatch, int id, size_t p, const std::string& code,
                       const std::string& options, std::string* err) {
  if (id < 0 || id >= kNumCallbacks) {
    *err = "unknown callback id " + std::to_string(id);
    return false;
  }
  PJRT_LoadedExecutable* exe = runtime_->Compile(code, options, err);
  if (exe == nullptr) {
    *err = std::string("cannot compile callback '") + CallbackName(id) + "': " + *err;
    return false;
  }
  Instance* instance;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& slot = instances_[nbatch];
    if (!slot) {
      slot = std::make_unique<Instance>();
      CallbackSet& cbs = slot->callbacks;
      cbs.runtime = runtime_;
      cbs.nstates = config_.nstates;
      cbs.nparams = config_.nparams;
      cbs.nout = config_.nout;
      cbs.nbatch = nbatch;
    }
    instance = slot.get();
  }
  std::lock_guard<std::mutex> guard(instance->mutex);
  instance->callbacks.Add(id, p, exe);
  return true;
}

bool Model::HasVariant(size_t nbatch, int id, size_t p) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = instances_.find(nbatch);
  return it != instances_.end() && it->second->callbacks.Has(id, p);
}

bool Model::Stats(size_t nbatch, uint64_t* ncalls, uint64_t* nstaged, uint64_t* nviews,
                  bool reset) {
  Instance* instance;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = instances_.find(nbatch);
    if (it == instances_.end()) return false;
    instance = it->second.get();
  }
  std::lock_guard<std::mutex> guard(instance->mutex);
  *ncalls = instance->callbacks.ncalls;
  *nstaged = instance->callbacks.nstaged;
  *nviews = instance->callbacks.nviews;
  if (reset) {
    instance->callbacks.ncalls = instance->callbacks.nstaged = instance->callbacks.nviews = 0;
  }
  return true;
}

Instance* Model::Acquire(size_t nbatch, std::unique_lock<std::mutex>* lock, std::string* err) {
  Instance* instance;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = instances_.find(nbatch);
    if (it == instances_.end()) {
      *err = "the model has no callbacks compiled for " + std::to_string(nbatch) + " lanes";
      return nullptr;
    }
    instance = it->second.get();
  }
  *lock = std::unique_lock<std::mutex>(instance->mutex);
  // The set of callbacks is fixed when the `OdeWrapper` is made. If a later lowering added a
  // callback (for example `rhs_sens` for a first forward-sensitivity solve), make it again.
  size_t ncallbacks = 0;
  for (const auto& v : instance->callbacks.variants) ncallbacks += v.count > 0;
  if (instance->ode && ncallbacks != instance->ncallbacks_at_construction) {
    // the checkpoints of the old `OdeWrapper` cannot be used with the new one
    {
      std::lock_guard<std::mutex> guard(mutex_);
      std::erase_if(checkpoints_, [&](const auto& c) { return c->instance == instance; });
    }
    diffsol_ode_free(instance->ode);
    instance->ode = nullptr;
  }
  if (instance->ode == nullptr) {
    OdeCallbacksDesc_f64 desc = {};
    desc.nstates = config_.nstates;
    desc.nparams = config_.nparams;
    desc.nout = config_.nout;
    instance->callbacks.Fill(&desc);
    instance->ode = diffsol_ode_new_callbacks_f64(&desc, config_.matrix_type,
                                                  config_.linear_solver, config_.ode_solver,
                                                  nbatch);
    if (instance->ode == nullptr) {
      *err = "cannot make the diffsol ODE: " + DiffsolError(instance);
      lock->unlock();
      return nullptr;
    }
    instance->ncallbacks_at_construction = ncallbacks;
  }
  return instance;
}

bool CheckpointKey::Matches(const double* p, size_t np, const double* t, size_t nt, double rt,
                            double at) const {
  // compare bits, not values: NaN params must not match, and -0.0 is not 0.0
  return np == params.size() && nt == t_eval.size() &&
         std::memcmp(&rt, &rtol, sizeof(double)) == 0 &&
         std::memcmp(&at, &atol, sizeof(double)) == 0 &&
         std::memcmp(p, params.data(), np * sizeof(double)) == 0 &&
         std::memcmp(t, t_eval.data(), nt * sizeof(double)) == 0;
}

int64_t Model::AddCheckpoint(Instance* instance, VjpCheckpointWrapper* checkpoint,
                             CheckpointKey key) {
  std::lock_guard<std::mutex> guard(mutex_);
  int64_t id = static_cast<int64_t>(Mix(ProcessSalt() + g_next_checkpoint++));
  checkpoints_.push_back(
      std::make_shared<Checkpoint>(id, instance, checkpoint, std::move(key)));
  while (checkpoints_.size() > config_.max_checkpoints) checkpoints_.pop_front();
  return id;
}

std::shared_ptr<Checkpoint> Model::FindCheckpoint(int64_t handle) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (const auto& c : checkpoints_) {
    if (c->id == handle) return c;
  }
  return nullptr;
}

void Model::ClearCheckpoints() {
  std::lock_guard<std::mutex> guard(mutex_);
  checkpoints_.clear();
}

size_t Model::NumCheckpoints() {
  std::lock_guard<std::mutex> guard(mutex_);
  return checkpoints_.size();
}

PjrtRuntime* GetRuntime() {
  std::lock_guard<std::mutex> guard(g_mutex);
  return g_runtime;
}

bool InitRuntime(const std::string& plugin_path, std::string* err) {
  std::lock_guard<std::mutex> guard(g_mutex);
  if (g_runtime) return true;
  g_runtime = PjrtRuntime::Create(plugin_path, err);
  return g_runtime != nullptr;
}

int64_t NewModel(const ModelConfig& config, std::string* err) {
  PjrtRuntime* runtime = GetRuntime();
  if (runtime == nullptr) {
    *err = "the PJRT runtime is not initialized";
    return 0;
  }
  if (config.nstates == 0) {
    *err = "nstates must be greater than 0";
    return 0;
  }
  int64_t id = g_next_model++;
  std::lock_guard<std::mutex> guard(g_mutex);
  g_models[id] = std::make_shared<Model>(id, runtime, config);
  return id;
}

std::shared_ptr<Model> FindModel(int64_t id) {
  std::lock_guard<std::mutex> guard(g_mutex);
  auto it = g_models.find(id);
  return it == g_models.end() ? nullptr : it->second;
}

void FreeModel(int64_t id) {
  std::shared_ptr<Model> model;
  {
    std::lock_guard<std::mutex> guard(g_mutex);
    auto it = g_models.find(id);
    if (it == g_models.end()) return;
    model = std::move(it->second);
    g_models.erase(it);
  }
  // the model is freed here, or when the last running solve releases it
}

std::string DiffsolError(Instance* instance) {
  const char* message = diffsol_error();
  std::string out = message ? message : "unknown error";
  if (instance) {
    std::string cb = instance->callbacks.TakeError();
    if (!cb.empty()) out += " [" + cb + "]";
  }
  return out;
}

}  // namespace diffsol_jax
