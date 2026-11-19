# Using diffsol from other languages

diffsol is a Rust library, but it can be used from other languages through various means. Three aspects that enable the portability of diffsol are:

- **The `diffsol-c` crate**: A companion crate providing a dynamic dispatch API and a C FFI that allows diffsol to be called from C, Python (via `cffi`/`ctypes`), and other languages with C bindings. Runtime configuration enums let you choose the solver method, matrix backend, scalar type, and linear solver without compile-time generic types.
- **The DiffSL DSL**: The DiffSL Domain Specific Language (DSL) allows users to specify ODEs, DAEs, and discretised PDEs in a way that is independent of the Rust language, but still can be JIT compiled to efficient native code.
- **WebAssembly**: As with many Rust libraries, diffsol can be easily compiled to WebAssembly (Wasm), allowing it to be used in web applications or other environments that support Wasm.

Note that the DiffSL JIT compiler cannot run inside WebAssembly, due to the complexity of compiling LLVM or Cranelift to Wasm. You can still use DiffSL from a Wasm application if a server does the JIT compilation, see [WebAssembly with JIT](./webassembly-jit.md).
