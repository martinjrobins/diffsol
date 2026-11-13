#include "pjrt_runtime.h"

#include <dlfcn.h>

#include <cstring>

namespace diffsol_jax {

namespace {

// Zero a PJRT argument struct and set its `struct_size`, as the C API needs.
template <typename T>
T Args() {
  T args;
  std::memset(&args, 0, sizeof(T));
  args.struct_size = sizeof(T);
  return args;
}

}  // namespace

bool PjrtRuntime::Check(PJRT_Error* error, const char* what, std::string* err) {
  if (error == nullptr) return true;
  auto message = Args<PJRT_Error_Message_Args>();
  message.error = error;
  api_->PJRT_Error_Message(&message);
  if (err) {
    *err = std::string(what) + ": " + std::string(message.message, message.message_size);
  }
  auto destroy = Args<PJRT_Error_Destroy_Args>();
  destroy.error = error;
  api_->PJRT_Error_Destroy(&destroy);
  return false;
}

void PjrtRuntime::AwaitAndDestroy(PJRT_Event* event, std::string* err, bool* ok) {
  if (event == nullptr) return;
  auto await = Args<PJRT_Event_Await_Args>();
  await.event = event;
  if (!Check(api_->PJRT_Event_Await(&await), "PJRT_Event_Await", err)) *ok = false;
  auto destroy = Args<PJRT_Event_Destroy_Args>();
  destroy.event = event;
  Check(api_->PJRT_Event_Destroy(&destroy), "PJRT_Event_Destroy", nullptr);
}

void PjrtRuntime::DestroyBuffer(PJRT_Buffer* buffer) {
  if (buffer == nullptr) return;
  auto destroy = Args<PJRT_Buffer_Destroy_Args>();
  destroy.buffer = buffer;
  Check(api_->PJRT_Buffer_Destroy(&destroy), "PJRT_Buffer_Destroy", nullptr);
}

PjrtRuntime* PjrtRuntime::Create(const std::string& plugin_path, std::string* err) {
  void* lib = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    *err = std::string("cannot load the PJRT plugin: ") + dlerror();
    return nullptr;
  }
  using GetPjrtApiFn = const PJRT_Api* (*)();
  auto get_api = reinterpret_cast<GetPjrtApiFn>(dlsym(lib, "GetPjrtApi"));
  if (get_api == nullptr) {
    *err = "the PJRT plugin " + plugin_path + " does not export GetPjrtApi";
    return nullptr;
  }
  auto* rt = new PjrtRuntime();
  rt->api_ = get_api();
  if (rt->api_->pjrt_api_version.major_version != PJRT_API_MAJOR) {
    *err = "the PJRT plugin has C API major version " +
           std::to_string(rt->api_->pjrt_api_version.major_version) + ", expected " +
           std::to_string(PJRT_API_MAJOR);
    delete rt;
    return nullptr;
  }
  auto init = Args<PJRT_Plugin_Initialize_Args>();
  if (!rt->Check(rt->api_->PJRT_Plugin_Initialize(&init), "PJRT_Plugin_Initialize", err)) {
    delete rt;
    return nullptr;
  }
  auto create = Args<PJRT_Client_Create_Args>();
  if (!rt->Check(rt->api_->PJRT_Client_Create(&create), "PJRT_Client_Create", err)) {
    delete rt;
    return nullptr;
  }
  rt->client_ = create.client;
  auto devices = Args<PJRT_Client_AddressableDevices_Args>();
  devices.client = rt->client_;
  if (!rt->Check(rt->api_->PJRT_Client_AddressableDevices(&devices),
                 "PJRT_Client_AddressableDevices", err)) {
    delete rt;
    return nullptr;
  }
  if (devices.num_addressable_devices == 0) {
    *err = "the PJRT client has no addressable device";
    delete rt;
    return nullptr;
  }
  rt->device_ = devices.addressable_devices[0];
  return rt;
}

PJRT_LoadedExecutable* PjrtRuntime::Compile(const std::string& code, const std::string& options,
                                            std::string* err) {
  auto program = Args<PJRT_Program>();
  program.code = const_cast<char*>(code.data());
  program.code_size = code.size();
  program.format = "mlir";
  program.format_size = 4;
  auto compile = Args<PJRT_Client_Compile_Args>();
  compile.client = client_;
  compile.program = &program;
  compile.compile_options = options.data();
  compile.compile_options_size = options.size();
  if (!Check(api_->PJRT_Client_Compile(&compile), "PJRT_Client_Compile", err)) return nullptr;
  return compile.executable;
}

void PjrtRuntime::Destroy(PJRT_LoadedExecutable* exe) {
  if (exe == nullptr) return;
  auto destroy = Args<PJRT_LoadedExecutable_Destroy_Args>();
  destroy.executable = exe;
  Check(api_->PJRT_LoadedExecutable_Destroy(&destroy), "PJRT_LoadedExecutable_Destroy", nullptr);
}

PJRT_Buffer* PjrtRuntime::CreateView(const ArgView& arg, std::string* err) {
  auto view = Args<PJRT_Client_CreateViewOfDeviceBuffer_Args>();
  view.client = client_;
  view.device_buffer_ptr = const_cast<void*>(arg.ptr);
  view.dims = arg.dims;
  view.num_dims = static_cast<size_t>(arg.ndim);
  view.element_type = PJRT_Buffer_Type_F64;
  view.device = device_;
  // the memory belongs to diffsol; some client paths call this callback without a null check
  view.on_delete_callback = [](void*, void*) {};
  if (!Check(api_->PJRT_Client_CreateViewOfDeviceBuffer(&view),
             "PJRT_Client_CreateViewOfDeviceBuffer", err)) {
    return nullptr;
  }
  return view.buffer;
}

bool PjrtRuntime::Execute(PJRT_LoadedExecutable* exe, PJRT_Buffer* const* args, int nargs,
                          void* out, size_t out_bytes, std::string* err) {
  // all the argument structs are on the stack: no allocation here
  // the inputs are views of diffsol memory: never donate them
  int64_t non_donatable[kMaxArgs];
  for (int i = 0; i < nargs; i++) non_donatable[i] = i;
  auto options = Args<PJRT_ExecuteOptions>();
  options.non_donatable_input_indices = non_donatable;
  options.num_non_donatable_input_indices = static_cast<size_t>(nargs);
  PJRT_Buffer* output = nullptr;
  PJRT_Buffer** output_list = &output;
  PJRT_Event* done = nullptr;
  auto execute = Args<PJRT_LoadedExecutable_Execute_Args>();
  execute.executable = exe;
  execute.options = &options;
  execute.argument_lists = &args;
  execute.num_devices = 1;
  execute.num_args = static_cast<size_t>(nargs);
  execute.output_lists = &output_list;
  execute.device_complete_events = &done;
  bool ok = Check(api_->PJRT_LoadedExecutable_Execute(&execute), "PJRT_LoadedExecutable_Execute",
                  err);
  if (ok) AwaitAndDestroy(done, err, &ok);
  if (ok) {
    // CPU: the output buffer is host memory and the execute event has completed, so copy it
    // directly. This is about 0.1 us; `PJRT_Buffer_ToHostBuffer` is about 9 us for a small
    // output (it goes through the async work queue of the client).
    // TODO(GPU): device memory; use ToHostBuffer, or better a device-to-device copy into `y`.
    auto pointer = Args<PJRT_Buffer_UnsafePointer_Args>();
    pointer.buffer = output;
    if (Check(api_->PJRT_Buffer_UnsafePointer(&pointer), "PJRT_Buffer_UnsafePointer", nullptr) &&
        pointer.buffer_pointer != 0) {
      std::memcpy(out, reinterpret_cast<const void*>(pointer.buffer_pointer), out_bytes);
    } else {
      auto to_host = Args<PJRT_Buffer_ToHostBuffer_Args>();
      to_host.src = output;
      to_host.dst = out;
      to_host.dst_size = out_bytes;
      ok = Check(api_->PJRT_Buffer_ToHostBuffer(&to_host), "PJRT_Buffer_ToHostBuffer", err);
      if (ok) AwaitAndDestroy(to_host.event, err, &ok);
    }
  }
  DestroyBuffer(output);
  return ok;
}

}  // namespace diffsol_jax
