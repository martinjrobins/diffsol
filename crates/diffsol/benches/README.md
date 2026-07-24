# Benchmarks

To run the benchmarks, use the following command:

```bash
cargo bench
```

If you want to run the benchmarks using the diffsl-llvm backend,
you need to set the correct environment variables and feature flags.
For example, on a linux system you might have something like:

```bash
LLVM_DIR=/usr/lib/llvm-21 LLVM_SYS_211_PREFIX=/usr/lib/llvm-21 cargo bench --features diffsl-llvm21
```

If you want to also run the sundials benchmarks, you need to have sundials
installed and turn on the `sundials` feature. If you want to specify a particular version of sundials, please
set the

```bash
SUNDIALS_LIBRARY_DIR=~/.local/lib SUNDIALS_INCLUDE_DIR=~/.local/include cargo bench --features sundials
```
