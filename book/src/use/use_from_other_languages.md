# Using Diffsol from other languages

Diffsol is a Rust library, but it can be used from other languages through various means. Two aspects that enable the portability of Diffsol are:

- **The DiffSL DSL**: The DiffSL Domain Specific Language (DSL) allows users to specify ODEs, DAEs, and discretised PDEs in a way that is independent of the Rust language, but still can be JIT compiled to efficient native code.
- **WebAssembly**: As with many Rust libraries, Diffsol can be easily compiled to WebAssembly (Wasm), allowing it to be used in web applications or other environments that support Wasm.

Note that a limitation is that these two aspects are not yet compatible with each other. DiffSL is the only part of Diffsol that is not yet available in WebAssembly, due to the complexity of compiling LLVM or Cranelift to Wasm. However, this is a planned feature for the future.