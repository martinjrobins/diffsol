# Build and test the `cuda-oxide` GPU backend.
#
# Prerequisites (`cargo oxide doctor` checks them): 
# the pinned nightly with `rust-src`/`rustc-dev`/`llvm-tools`,
# CUDA Toolkit 13+, LLVM 21+ with NVPTX, driver 580+, an Ampere+ GPU,
# cuda-oxide git dependencies enabled in crates/diffsol-la/Cargo.toml:
#   sed -i -e '/^cuda-oxide = /d' -e 's/^#oxide# //' crates/diffsol-la/Cargo.toml
#
# RUSTFLAGS works around a rustc ICE on this nightly:
# https://github.com/rust-lang/rust/issues/162323
#
# `oxide_rev` is the commit crates/diffsol-la/Cargo.toml pins for
# cuda-device/cuda-host, and the driver and the codegen backend must come from
# it too, so kernels and the backend that lowers them agree. Install the driver
# with:
#
#   cargo +nightly-2026-08-28 install --locked \
#     --git https://github.com/NVlabs/cuda-oxide.git --rev <oxide_rev> cargo-oxide
#
# and build the backend with `just oxide-backend`. 
#
# cargo-oxide itself uses `cargo metadata --all-features` to find backend,
# but diffsol has conflicting features so use CUDA_OXIDE_BACKEND directly 
# as below.
#
# Bump oxide_nightly/oxide_rev here, in crates/diffsol-la/Cargo.toml and in
# OXIDE_NIGHTLY/OXIDE_REV (.github/workflows/rust.yml) together.
oxide_nightly := "nightly-2026-08-28"
oxide_rev := "26754ae52c26c097dc1c465a1e42c4c5d05a3d40"
oxide_flags := "-Znext-solver=coherence"
oxide_arch := "sm_86"
oxide_backend := justfile_directory() / "target/cuda-oxide-backend/librustc_codegen_cuda.so"

# Build the codegen backend from the pinned cuda-oxide rev. Needed once per
# rev/nightly: the .so links that toolchain's librustc_driver.
oxide-backend:
    rm -rf target/cuda-oxide-src
    git clone --filter=blob:none --no-checkout \
        https://github.com/NVlabs/cuda-oxide.git target/cuda-oxide-src
    git -C target/cuda-oxide-src checkout {{oxide_rev}}
    cd target/cuda-oxide-src && cargo +{{oxide_nightly}} oxide setup
    mkdir -p target/cuda-oxide-backend
    cp "$(find target/cuda-oxide-src/crates/rustc-codegen-cuda/target \
        -name librustc_codegen_cuda.so | head -1)" {{oxide_backend}}

# Check the cuda-oxide toolchain is installed and usable.
oxide-doctor:
    CUDA_OXIDE_BACKEND="{{oxide_backend}}" cargo +{{oxide_nightly}} oxide doctor

oxide-test *ARGS:
    CARGO_PROFILE_DEV_DEBUG=0 CUDA_OXIDE_BACKEND="{{oxide_backend}}" \
        RUSTFLAGS="{{oxide_flags}}" \
        cargo +{{oxide_nightly}} oxide test --arch {{oxide_arch}} -- \
        -p diffsol-la --features cuda-oxide {{ARGS}}

# Run the cuda-oxide tests in the `diffsol` crate (the recipe above is pinned to
# `diffsol-la`; `just` would read a leading `--` as a package argument, so this is
# a second recipe rather than a parameter).
#
# `diffsol` monomorphizes the generic lane-closure kernels, so this build needs the
# CARGO_PROFILE_DEV_DEBUG=0 workaround described on `oxide-bench` below.
oxide-test-diffsol *ARGS:
    CARGO_PROFILE_DEV_DEBUG=0 CUDA_OXIDE_BACKEND="{{oxide_backend}}" \
        RUSTFLAGS="{{oxide_flags}}" \
        cargo +{{oxide_nightly}} oxide test --arch {{oxide_arch}} -- \
        -p diffsol --features cuda-oxide {{ARGS}}

# Run the GPU ensemble example (examples/performance-gpu-ensemble) on the device.
#
# `cargo oxide run` takes the package after `--`, and the workspace release profile enables
# thin LTO, which the CUDA codegen backend cannot read bitcode through ("Can't find section
# .llvmbc"). So this builds the dev profile with optimisations turned up instead.
gpu-ensemble *ARGS:
    CARGO_PROFILE_DEV_OPT_LEVEL=3 CARGO_PROFILE_DEV_DEBUG=0 \
        CUDA_OXIDE_BACKEND="{{oxide_backend}}" RUSTFLAGS="{{oxide_flags}}" \
        cargo +{{oxide_nightly}} oxide build --arch {{oxide_arch}} -- \
        -p performance-gpu-ensemble --features cuda-oxide {{ARGS}}
    ./target/debug/performance-gpu-ensemble

oxide-build *ARGS:
    CUDA_OXIDE_BACKEND="{{oxide_backend}}" RUSTFLAGS="{{oxide_flags}}" \
        cargo +{{oxide_nightly}} oxide build --arch {{oxide_arch}} -- \
        -p diffsol --features cuda-oxide {{ARGS}}

# Run the cuda-oxide benchmarks.
#
# `cargo oxide` has no `bench` subcommand, so build the bench target and run it
# with criterion's `--bench` flag.
#
# CARGO_PROFILE_DEV_DEBUG=0: a generic kernel monomorphized in a consuming crate lands in that
# crate's own PTX bundle, and `load_all_ptx_bundles_merged` merges the bundles by stripping only
# `.version`/`.target`/`.address_size`. Debug builds also carry `.file 1..N` line tables, so the
# merged module has duplicate file indices and the driver rejects it ("a PTX JIT compilation
# failed" / ptxas "Duplicate file index #1"). Dropping debug info drops the `.file` directives.
oxide-bench *ARGS:
    CARGO_PROFILE_DEV_DEBUG=0 CUDA_OXIDE_BACKEND="{{oxide_backend}}" \
        RUSTFLAGS="{{oxide_flags}}" \
        cargo +{{oxide_nightly}} oxide build --arch {{oxide_arch}} -- \
        -p diffsol --features cuda-oxide --bench lin_alg_ops
    "$(ls -t target/debug/build/diffsol/*/out/lin_alg_ops-* | grep -v '\.d$' | head -1)" \
        --bench {{ARGS}}

# Run the element-parallel ODE model benchmarks (see the note on `oxide-bench`).
oxide-bench-ode *ARGS:
    CARGO_PROFILE_DEV_DEBUG=0 CUDA_OXIDE_BACKEND="{{oxide_backend}}" \
        RUSTFLAGS="{{oxide_flags}}" \
        cargo +{{oxide_nightly}} oxide build --arch {{oxide_arch}} -- \
        -p diffsol --features cuda-oxide --bench ode_solvers_oxide
    "$(ls -t target/debug/build/diffsol/*/out/ode_solvers_oxide-* | grep -v '\.d$' | head -1)" \
        --bench {{ARGS}}

# Run the cuda-oxide tests under compute-sanitizer (memcheck or racecheck).
#
# `cargo oxide sanitize` only drives executable targets, so build the test
# binary and hand that to compute-sanitizer directly.
oxide-sanitize tool="memcheck" *ARGS:
    CUDA_OXIDE_BACKEND="{{oxide_backend}}" RUSTFLAGS="{{oxide_flags}}" \
        cargo +{{oxide_nightly}} oxide test --arch {{oxide_arch}} -- \
        -p diffsol-la --features cuda-oxide --no-run
    compute-sanitizer --tool {{tool}} --error-exitcode 1 \
        "$(ls -t target/debug/build/diffsol-la/*/out/diffsol_la-* | grep -v '\.d$' | head -1)" \
        --test-threads=1 {{ARGS}}
