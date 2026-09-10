# Build and test the `cuda-oxide` GPU backend.
#
# cuda-oxide compiles `#[kernel]` functions through a custom rustc codegen
# backend, which `cargo oxide` injects via CARGO_ENCODED_RUSTFLAGS -- so plain
# `cargo test` cannot build this feature. Prerequisites (`cargo oxide doctor`
# checks them): the pinned nightly with `rust-src`/`rustc-dev`/`llvm-tools`,
# CUDA Toolkit 13+, LLVM 21+ with NVPTX, driver 580+, an Ampere+ GPU.
#
# RUSTFLAGS works around a rustc ICE on this nightly:
# https://github.com/rust-lang/rust/issues/162323
oxide_nightly := "nightly-2026-08-28"
oxide_flags := "-Znext-solver=coherence"
oxide_arch := "sm_86"

# Check the cuda-oxide toolchain is installed and usable.
oxide-doctor:
    cargo +{{oxide_nightly}} oxide doctor

oxide-test *ARGS:
    RUSTFLAGS="{{oxide_flags}}" cargo +{{oxide_nightly}} oxide test --arch {{oxide_arch}} -- \
        -p diffsol-la --features cuda-oxide {{ARGS}}

oxide-build *ARGS:
    RUSTFLAGS="{{oxide_flags}}" cargo +{{oxide_nightly}} oxide build --arch {{oxide_arch}} -- \
        -p diffsol --features cuda-oxide {{ARGS}}

# Run the cuda-oxide tests under compute-sanitizer (memcheck or racecheck).
#
# `cargo oxide sanitize` only drives executable targets, so build the test
# binary and hand that to compute-sanitizer directly.
oxide-sanitize tool="memcheck" *ARGS:
    RUSTFLAGS="{{oxide_flags}}" cargo +{{oxide_nightly}} oxide test --arch {{oxide_arch}} -- \
        -p diffsol-la --features cuda-oxide --no-run
    compute-sanitizer --tool {{tool}} --error-exitcode 1 \
        "$(ls -t target/debug/build/diffsol-la/*/out/diffsol_la-* | grep -v '\.d$' | head -1)" \
        --test-threads=1 {{ARGS}}
