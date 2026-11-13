cmake_minimum_required(VERSION 3.20)
project(diffsol_jax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# diffsol-c: `cargo build -p diffsol-c --release` makes the library and the header in
# `target/release` and `target/release/include`.
set(DIFFSOL_C_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release"
    CACHE PATH "Directory with libdiffsol_c and include/diffsol.h")

# The XLA FFI header comes with jax: `python -c "import jax.ffi; print(jax.ffi.include_dir())"`.
if(NOT XLA_FFI_INCLUDE_DIR)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
endif()
message(STATUS "XLA FFI headers: ${XLA_FFI_INCLUDE_DIR}")

find_library(DIFFSOL_C_LIB NAMES diffsol_c PATHS "${DIFFSOL_C_DIR}" NO_DEFAULT_PATH REQUIRED)

add_library(diffsol_jax SHARED
  src/pjrt_runtime.cc
  src/callbacks.cc
  src/registry.cc
  src/ffi_ops.cc
  src/c_api.cc)
target_include_directories(diffsol_jax SYSTEM PRIVATE "${XLA_FFI_INCLUDE_DIR}")
target_include_directories(diffsol_jax PRIVATE
  src
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
  "${DIFFSOL_C_DIR}/include")
target_link_libraries(diffsol_jax PRIVATE "${DIFFSOL_C_LIB}" ${CMAKE_DL_LIBS})
target_compile_options(diffsol_jax PRIVATE -Wall -Wextra)
# find libdiffsol_c next to libdiffsol_jax, or in DIFFSOL_C_DIR
set_target_properties(diffsol_jax PROPERTIES
  BUILD_RPATH "${DIFFSOL_C_DIR}"
  INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>")
