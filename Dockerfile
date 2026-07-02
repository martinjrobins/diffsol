FROM rust:1.89-bookworm

WORKDIR /usr/src/diffsol

# Layer 1: Copy workspace manifests and create minimal source for dep pre-compilation
# This layer is only rebuilt when Cargo.toml files change.
COPY Cargo.toml ./
COPY diffsol/Cargo.toml diffsol/
COPY diffsol-c/Cargo.toml diffsol-c/

RUN mkdir -p diffsol/src diffsol-c/src diffsol/benches && \
    touch diffsol/src/lib.rs && \
    touch diffsol-c/src/lib.rs && \
    echo "fn main() {}" > diffsol/benches/ode_solvers_ci.rs && \
    echo "fn main() {}" > diffsol/benches/ode_solvers.rs && \
    echo "fn main() {}" > diffsol/benches/lin_alg_ops.rs && \
    echo "#[cfg(feature = \"diffsl-llvm\")] fn main() {}" > diffsol/benches/pybamm_dfn.rs && \
    touch diffsol/benches/common.rs && \
    touch diffsol/benches/sundials_benches.rs && \
    mkdir -p examples/dummy && \
    printf '[package]\nname = "dummy"\nversion = "0.1.0"\nedition = "2021"\n' > examples/dummy/Cargo.toml && \
    mkdir -p examples/dummy/src && \
    echo "fn main() {}" > examples/dummy/src/main.rs

RUN cargo fetch
RUN cargo build --bench ode_solvers_ci || true

# Layer 2: Copy real source and build bench binary
# This layer rebuilds on every commit, but only changed source is recompiled
# since deps are cached from Layer 1.
COPY . .
RUN cargo build --bench ode_solvers_ci

CMD ["cargo", "bench", "--bench", "ode_solvers_ci"]
